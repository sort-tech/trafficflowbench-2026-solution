"""Feature rows for train target cells: a sampled fit fold and a full eval fold.

eval fold = train days with index % 9 == 4 (about 30 days per panel, all regimes).
fit fold  = the remaining days, sampled to N_PER_REGIME rows per regime and panel.
"""
import sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import common, features as F

OUT = Path(__file__).resolve().parent / os.environ.get("DS_DIR", "ds")
N_PER_REGIME = int(os.environ.get("N_PER_REGIME", 80000))
BLOCK = 21
SEED = 20261025


def eval_days(nd):
    return np.arange(nd) % 9 == 4


def gap_spans(nd, rng, per_day=10 / 31):
    """Simulated Task 2 gaps: 18 slots blanked on every link, starting 07:00-18:00."""
    eb = np.zeros((nd, common.T), bool)
    for day in range(nd):
        n = rng.poisson(per_day)
        for _ in range(n):
            s = rng.integers(84, 216)
            eb[day, s:s + 18] = True
    return eb


def panel_rows(p):
    t0 = time.time()
    d = common.load(p, "train")
    net = common.network(p)
    lanes = net.lanes.to_numpy(np.float32)
    prof = F.profiles([d], lanes)
    nd = len(d["dates"]); nl = len(lanes)
    ev = eval_days(nd)
    rng = np.random.default_rng(SEED + common.PANELS.index(p))
    # fit-fold sampling probability per regime
    fit_counts = {r: d["target"][(~ev) & (d["regime"] == r)].sum() for r in common.REGIMES}
    prob = {r: min(1.0, N_PER_REGIME / max(c, 1)) for r, c in fit_counts.items()}
    gmode = os.environ.get("GAPS", "1")
    wins = []
    if gmode == "win":
        wins = common.train_windows(p, d, np.random.default_rng(SEED + 200 + common.PANELS.index(p)))
        extra = float(os.environ.get("EXTRA_WIN", 0))
        if extra > 0:
            # more simulated windows for a gap specialist, only on days whose
            # neighbours are not eval days, so the eval rows stay identical
            far = ~(ev | np.roll(ev, 1) | np.roll(ev, -1))
            xw = common.train_windows(p, d, np.random.default_rng(SEED + 300 + common.PANELS.index(p)), extra)
            wins = wins + [w for w in xw if far[w[0]]]
        gaps = np.zeros((nd, common.T), bool)
        flat = gaps.reshape(-1)
        for di0, s0, _ in wins:
            a = di0 * common.T + s0
            flat[a + 1:a + 19] = True
    elif gmode == "1":
        gaps = gap_spans(nd, np.random.default_rng(SEED + 100 + common.PANELS.index(p)))
    else:
        gaps = None
    fv = int(os.environ.get("FEAT", 3))
    rmp = None
    if fv >= 6:
        rmp = common.load_ramps(p, "train", d["dates"], d["links"])
        rmp["prof"] = F.ramp_profile([(rmp["flow"], d["dates"])])
    # FEAT >= 8: bottleneck thresholds for the gap features, fitted on the fit days only
    thr = F.fit_bottlenecks(d, net, ~ev) if fv >= 8 and gmode == "win" and wins else None
    fit_parts, eval_parts = [], []
    for s in range(0, nd, BLOCK):
        core = np.arange(s, min(nd, s + BLOCK))
        lo, hi = max(0, s - 1), min(nd, s + BLOCK + 1)
        days = np.arange(lo, hi)
        eb = None if gaps is None else np.broadcast_to(gaps[days][:, :, None], (len(days), common.T, nl))
        bw = [((di0 - lo) * common.T + s0, c) for di0, s0, c in wins if lo <= di0 < hi]
        rb = None if rmp is None else dict(rmp, flow=rmp["flow"][days])
        blk = F.Block(d, days, net, prof, ramps=rb, extra_blank=eb, version=fv, windows=bw, thr=thr)
        tg = d["target"][days].reshape(len(days) * common.T, nl)
        ti, li = np.nonzero(tg)
        dd = days[ti // common.T]
        incore = (dd >= core[0]) & (dd <= core[-1])
        reg = d["regime"][dd]
        is_ev = ev[dd]
        u = rng.random(len(ti))
        pr = np.vectorize(prob.get)(reg)
        if gaps is not None:
            pr = pr * np.where(gaps[dd, ti % common.T], float(os.environ.get("GAP_OS", 1.0)), 1.0)
        keep_fit = incore & ~is_ev & (u < pr)
        if os.environ.get("GAP_ONLY") == "1":
            keep_fit &= gaps[dd, ti % common.T]
        keep_ev = incore & is_ev
        if os.environ.get("EVAL_GAP_ONLY") == "1":
            keep_ev &= gaps[dd, ti % common.T]
        for keep, parts in ((keep_fit, fit_parts), (keep_ev, eval_parts)):
            if not keep.any():
                continue
            X = blk.features(ti[keep], li[keep], net)
            X["panel"] = p
            X["regime"] = reg[keep]
            X["day"] = dd[keep].astype(np.int16)
            X["t"] = (ti[keep] % common.T).astype(np.int16)
            X["li"] = li[keep].astype(np.int16)
            tv = d["true_speed"][dd[keep], ti[keep] % common.T, li[keep]]
            tq = d["true_flow"][dd[keep], ti[keep] % common.T, li[keep]] / lanes[li[keep]]
            X["y_v"] = tv.astype(np.float32)
            X["y_q"] = tq.astype(np.float32)
            parts.append(X)
        del blk
    fit = pd.concat(fit_parts, ignore_index=True)
    evl = pd.concat(eval_parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fit.to_parquet(OUT / f"fit_{p}.parquet", index=False)
    evl.to_parquet(OUT / f"eval_{p}.parquet", index=False)
    print(p, "fit", len(fit), "eval", len(evl), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    panels = sys.argv[1].split(",") if len(sys.argv) > 1 else common.PANELS
    for p in panels:
        panel_rows(p)
