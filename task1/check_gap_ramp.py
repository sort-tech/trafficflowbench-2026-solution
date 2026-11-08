"""Mac checks for the ramp gap specialist wiring (no stage-2 models needed).

1. apply_gap with one v4-only model set gives the same numbers as the previous code.
2. pipeline.gap_features on the train eval days (same blocks, windows and blanks as
   make_dataset) reproduces the ds11 eval feature columns, and apply_gap with the four
   ramp models at 0.75 on the x3 stage-2 eval predictions reproduces the proxy
   (0.44290 -> 0.44339, +0.000498).

    python -I check_gap_ramp.py
"""
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightgbm as lgb
import numpy as np
import pandas as pd

import common
import metrics
import pipeline
import train_gbm as TG
from make_dataset import SEED, BLOCK, eval_days

HERE = Path(__file__).resolve().parent
TAGS = ["gap8", "gap8_s11", "gap8b", "gap8b_s11"]


def old_apply_gap(X, pv, pq, pk, gap, gap_w):
    """apply_gap before the 10/30 change, for the regression check."""
    g = X.win_cond.to_numpy() > 0
    Xg = X.loc[g, gap["y_v"].feature_name()]
    gv = X.base_v.to_numpy()[g] + gap["y_v"].predict(Xg, num_threads=common.THREADS)
    gq = X.base_q.to_numpy()[g] + gap["y_q"].predict(Xg, num_threads=common.THREADS)
    gk = pipeline.k_base(X.loc[g]) + gap["y_k"].predict(Xg, num_threads=common.THREADS)
    pv, pq, pk = pv.copy(), pq.copy(), pk.copy()
    pv[g] = gap_w * gv + (1 - gap_w) * pv[g]
    pq[g] = gap_w * gq + (1 - gap_w) * pq[g]
    pk[g] = gap_w * gk + (1 - gap_w) * pk[g]
    return pv, pq, pk


def train_wins(p, d):
    """Windows and blanks exactly as make_dataset (GAPS=win, EXTRA_WIN=1)."""
    nd = len(d["dates"])
    ev = eval_days(nd)
    i = common.PANELS.index(p)
    wins = common.train_windows(p, d, np.random.default_rng(SEED + 200 + i))
    far = ~(ev | np.roll(ev, 1) | np.roll(ev, -1))
    xw = common.train_windows(p, d, np.random.default_rng(SEED + 300 + i), 1.0)
    wins = wins + [w for w in xw if far[w[0]]]
    gaps = np.zeros(nd * common.T, bool)
    for di0, s0, _ in wins:
        a = di0 * common.T + s0
        gaps[a + 1:a + 19] = True
    return wins, gaps.reshape(nd, common.T)


def block_features(p, ev):
    """gap_features per make_dataset block for the eval gap rows of one panel."""
    d = common.load(p, "train")
    nl = len(d["links"])
    nd = len(d["dates"])
    wins, gaps = train_wins(p, d)
    parts = []
    for s in range(0, nd, BLOCK):
        core = np.arange(s, min(nd, s + BLOCK))
        lo, hi = max(0, s - 1), min(nd, s + BLOCK + 1)
        days = np.arange(lo, hi)
        sel = ev.day.between(core[0], core[-1]).to_numpy()
        if not sel.any():
            continue
        e = ev[sel]
        eb = np.broadcast_to(gaps[days][:, :, None], (len(days), common.T, nl))
        bw = [((di0 - lo) * common.T + s0, c) for di0, s0, c in wins if lo <= di0 < hi]
        tt = (e.day.to_numpy().astype(np.int64) - lo) * common.T + e.t.to_numpy()
        X = pipeline.gap_features(p, "train", tt, e.li.to_numpy(), wins=bw, extra_blank=eb, days=days)
        X.index = e.index
        parts.append(X)
    return pd.concat(parts).sort_index()


def main():
    common.THREADS = 8
    sets = [{t: lgb.Booster(model_file=str(HERE / "models" / f"{tg}_{t}.txt")) for t in pipeline.TARGETS} for tg in TAGS]
    v4 = {t: lgb.Booster(model_file=str(HERE / "models" / f"gbm4_{t}.txt")) for t in pipeline.TARGETS}
    rows = {"x3g05 (preds)": [], "ramp x4 @0.75 via apply_gap": []}
    worst = 0.0
    for p in common.PANELS:
        ev = pd.read_parquet(HERE / "ds4" / f"eval_{p}.parquet")
        ev["rate"] = ev.regime.map(TG.RATE).astype(np.float32)
        base = pd.read_parquet(HERE / "preds" / f"gbm8s2_x3_{p}.parquet")
        x3g = pd.read_parquet(HERE / "preds" / f"x3g05_{p}.parquet")
        pv, pq, pk = (base[c].to_numpy() for c in ("pv", "pq", "pk"))
        g = ev.win_cond.to_numpy() > 0
        if g.any():
            # 1. default path unchanged (one v4-only set)
            a = old_apply_gap(ev, pv, pq, pk, v4, 0.5)
            b = pipeline.apply_gap(ev, pv, pq, pk, v4, 0.5)
            assert all(np.array_equal(x, y) for x, y in zip(a, b)), p
            # 2. features through gap_features == ds11
            e = ev[g]
            Xf = block_features(p, e)
            D = pd.read_parquet(HERE / "ds11" / f"eval_{p}.parquet")
            cols = [c for c in D.columns if c.startswith(("g5_", "g6_", "g7_", "g8_"))] + ["base_v", "base_q"]
            dif = np.nanmax(np.abs(Xf[cols].to_numpy(np.float64) - D[cols].to_numpy(np.float64)))
            nan_ok = (np.isnan(Xf[cols].to_numpy()) == np.isnan(D[cols].to_numpy())).all()
            worst = max(worst, dif)
            print(p, "default path identical; max |feature diff|", f"{dif:.2e}", "NaN pattern same", nan_ok, flush=True)
            orig = pipeline.gap_features
            pipeline.gap_features = lambda *a, **k: Xf.reset_index(drop=True)
            try:
                tt = ev.day.to_numpy().astype(np.int64) * common.T + ev.t.to_numpy()
                rv, rq, rk = pipeline.apply_gap(ev, pv, pq, pk, sets, 0.75, cells=(p, "train", tt, ev.li.to_numpy()))
            finally:
                pipeline.gap_features = orig
        else:
            rv, rq, rk = pv, pq, pk
        pe = TG.PanelEval(p, ev)
        for name, (a, b, c) in (("x3g05 (preds)", (x3g.pv.to_numpy(), x3g.pq.to_numpy(), x3g.pk.to_numpy())),
                                ("ramp x4 @0.75 via apply_gap", (rv, rq, rk))):
            v, q = pipeline.reconcile(a, b, c, 1.0, 0.2, 0.8, 80.0)
            rows[name].append(pe(v, q))
    S = {k: metrics.contribution(metrics.summarize(pd.concat(r))) for k, r in rows.items()}
    for k, v in S.items():
        print(f"{k}: {v:.6f}")
    print(f"gain {S['ramp x4 @0.75 via apply_gap'] - S['x3g05 (preds)']:+.6f}; worst feature diff {worst:.2e}")


if __name__ == "__main__":
    main()
