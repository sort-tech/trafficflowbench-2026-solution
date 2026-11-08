"""Stage 2: re-predict target cells with every missing cell pre-filled by stage 1.

On R3 days half of the eligible cells are targets, so many same-time neighbours
are missing and the stage-1 features fall back to wider kernels. Stage 2 fills
every missing cell (targets, outages, Task 2 gaps) with a stage-1 prediction,
recomputes neighbour features on the filled field, and fits the remaining
residual truth - p1.

Out-of-fold: stage-1 models are fitted per day parity, and a day is filled by
the model of the other parity, so stage 2 never sees in-sample stage-1 values.

    python -u stage2.py fit1  <ds>          # stage-1 models per parity
    python -u stage2.py rows  <ds> <panel>  # filled-field features for that ds's rows
    python -u stage2.py fit2  <ds>          # stage-2 models + eval
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightgbm as lgb
import numpy as np
import pandas as pd

import common
import features as F
import metrics
import train_gbm as TG
from make_dataset import SEED

HERE = Path(__file__).resolve().parent
TARGETS = ("y_v", "y_q", "y_k")
ROUNDS1 = int(os.environ.get("ROUNDS1", 800))
LR1 = float(os.environ.get("LR1", 0.1))


def stage1_dir(dsn):
    return HERE / "models" / f"s1_{dsn}"


def lean_rows(files, keep=None):
    """Feature matrix of the rows of several parquet tables (one list of files per panel,
    joined side by side), built straight into one float32 array.

    Columns: the feature columns (not META, not DROP) in file order, then "rate", the same
    order TG.feat_cols gives on the concatenated frame. LightGBM would turn the mixed
    float32 / int16 / int8 frame into exactly this float32 matrix, so the models are the
    same; the peak memory is about one copy of the matrix instead of three.
    keep(df) -> boolean row mask (applied per panel). Returns the frame (one float32 block)
    and the label columns y_v, y_q, base_v, base_q, p1_v, p1_q, p1_k, lanes, len as arrays."""
    import pyarrow.parquet as pq
    names = [[n for n in pq.read_schema(f).names] for f in files[0]]
    fcols = [[c for c in nm if c not in TG.META and not any(c.startswith(x) for x in TG.DROP)] for nm in names]
    cols = [c for fc in fcols for c in fc] + ["rate"]
    lab = ["y_v", "y_q", "base_v", "base_q", "p1_v", "p1_q", "p1_k", "lanes", "len"]
    # first pass: row masks from the day column only; second pass: one panel at a time
    masks = []
    for fs in files:
        dd = pd.read_parquet(fs[0], columns=["day"])
        masks.append(np.ones(len(dd), bool) if keep is None else keep(dd).to_numpy())
    n = int(sum(m.sum() for m in masks))
    X = np.empty((n, len(cols)), np.float32)
    L = {}
    o = 0
    for fs, m in zip(files, masks):
        dfs = [pd.read_parquet(f, columns=[c for c in nm if c in fc or c in lab or c in ("day", "regime")])
               for f, nm, fc in zip(fs, names, fcols)]
        assert all(len(df) == len(m) for df in dfs)
        k = int(m.sum())
        j = 0
        for df, fc in zip(dfs, fcols):
            X[o:o + k, j:j + len(fc)] = df.loc[m, fc].to_numpy(np.float32)
            j += len(fc)
            for c in lab:
                if c in df.columns:
                    L.setdefault(c, []).append(df.loc[m, c])
        X[o:o + k, j] = dfs[0].loc[m, "regime"].map(TG.RATE).astype(np.float32).to_numpy()
        o += k
        del dfs
    L = {c: pd.concat(v, ignore_index=True) for c, v in L.items()}
    return pd.DataFrame(X, columns=cols, copy=False), L


def fit1(dsn):
    ds = HERE / dsn
    files = [[ds / f"fit_{p}.parquet"] for p in common.PANELS if (ds / f"fit_{p}.parquet").exists()]
    out = stage1_dir(dsn)
    out.mkdir(parents=True, exist_ok=True)
    for par in (0, 1):
        X, L = lean_rows(files, keep=lambda df: df.day % 2 == par)
        for tgt in TARGETS:
            w = None
            if tgt == "y_k":
                y = (L["y_q"] / L["y_v"].clip(lower=1.0)).to_numpy(np.float32) - \
                    (L["base_q"] / L["base_v"].clip(lower=1.0)).to_numpy(np.float32)
                w = (L["lanes"] * L["len"]).to_numpy(np.float32)
            else:
                y = (L[tgt] - L["base_" + tgt[-1]]).to_numpy(np.float32)
            prm = TG.params()
            prm["learning_rate"] = LR1
            t0 = time.time()
            m = lgb.train(prm, lgb.Dataset(X, y, weight=w), ROUNDS1)
            m.save_model(str(out / f"p{par}_{tgt}.txt"))
            print(f"parity {par} {tgt} {time.time()-t0:.0f}s", flush=True)
        del X, L  # before the other parity's matrix is built


def gaps_and_windows(p, d):
    """The same simulated Task 2 windows make_dataset uses (GAPS=win)."""
    nd = len(d["dates"])
    wins = common.train_windows(p, d, np.random.default_rng(SEED + 200 + common.PANELS.index(p)))
    gaps = np.zeros(nd * common.T, bool)
    for di0, s0, _ in wins:
        a = di0 * common.T + s0
        gaps[a + 1:a + 19] = True
    return gaps.reshape(nd, common.T), wins


def predict_cells(blk, models, ti, li, net, rate, chunk=400000):
    out = {t: np.empty(len(ti), np.float32) for t in TARGETS}
    cols = models["y_v"].feature_name()
    for s in range(0, len(ti), chunk):
        X = blk.features(ti[s:s + chunk], li[s:s + chunk], net)
        X["rate"] = rate[s:s + chunk]
        out["y_v"][s:s + chunk] = X.base_v.to_numpy() + models["y_v"].predict(X[cols], num_threads=common.THREADS)
        out["y_q"][s:s + chunk] = X.base_q.to_numpy() + models["y_q"].predict(X[cols], num_threads=common.THREADS)
        out["y_k"][s:s + chunk] = TG.k_base(X) + models["y_k"].predict(X[cols], num_threads=common.THREADS)
    return out


FILLED_KEYS = None


def filled_features(fb: F.Block, ti, li, net, p1v, p1q, p1k, obs_v):
    """Features of the filled field at the requested cells."""
    G = {"p1_v": p1v, "p1_q": p1q, "p1_k": p1k}
    nl = fb.v.shape[1]
    nT = fb.v.shape[0]
    for o in (-3, -2, -1, 1, 2, 3):
        lj = li + o
        ok = (lj >= 0) & (lj < nl)
        ljc = np.clip(lj, 0, nl - 1)
        G[f"fv_{o}"] = np.where(ok, fb.v[ti, ljc], np.nan).astype(np.float32)
        G[f"fq_{o}"] = np.where(ok, fb.q[ti, ljc], np.nan).astype(np.float32)
        G[f"obs_{o}"] = np.where(ok, np.isfinite(obs_v[ti, ljc]), False).astype(np.int8)
    for dt in (-2, -1, 1, 2):
        tj = np.clip(ti + dt, 0, nT - 1)
        G[f"fv_t{dt}"] = fb.v[tj, li]
        G[f"fq_t{dt}"] = fb.q[tj, li]
    for key in ("rv_s0.5", "rv_s2.0", "rv_all", "rq_s0.5", "rq_s2.0", "rk_s0.5", "rk_s2.0", "rv_up2", "rv_dn2"):
        G["f" + key] = fb.kern[key][ti, li]
    for key in ("v_W2", "q_W2", "k_W2", "v_W6", "q_W6"):
        G["f" + key] = fb.f[key][ti, li]
    return pd.DataFrame(G)


def rows(dsn, p):
    """Stage-2 feature rows for the fit and eval rows of dataset dsn, one panel."""
    d = common.load(p, "train")
    net = common.network(p)
    lanes = net.lanes.to_numpy(np.float32)
    nd, nl = len(d["dates"]), len(lanes)
    prof = F.profiles([d], lanes)
    gaps, wins = gaps_and_windows(p, d)
    mdl = {par: {t: lgb.Booster(model_file=str(stage1_dir(dsn) / f"p{par}_{t}.txt")) for t in TARGETS}
           for par in (0, 1)}
    want = {k: pd.read_parquet(HERE / dsn / f"{k}_{p}.parquet", columns=["day", "t", "li"]) for k in ("fit", "eval")}
    # FEAT >= 6 (ramp flows) needs the ramp arrays in the stage-1 block; default 4 as before
    fver = int(os.environ.get("FEAT", 4))
    rmp = None
    if fver >= 6:
        rmp = common.load_ramps(p, "train", d["dates"], d["links"])
        rmp["prof"] = F.ramp_profile([(rmp["flow"], d["dates"])])
    parts = {k: [] for k in want}
    B = 21
    t0 = time.time()
    for s in range(0, nd, B):
        lo, hi = max(0, s - 1), min(nd, s + B + 1)
        days = np.arange(lo, hi)
        core_hi = min(nd, s + B)
        sel = {k: w[(w.day >= s) & (w.day < core_hi)] for k, w in want.items()}
        if all(len(x) == 0 for x in sel.values()):
            continue
        eb = np.broadcast_to(gaps[days][:, :, None], (len(days), common.T, nl))
        bw = [((di0 - lo) * common.T + s0, c) for di0, s0, c in wins if lo <= di0 < hi]
        rb = None if rmp is None else dict(rmp, flow=rmp["flow"][days])
        blk = F.Block(d, days, net, prof, ramps=rb, extra_blank=eb, version=fver, windows=bw)
        miss = ~np.isfinite(blk.v)
        ti, li = np.nonzero(miss)
        dd = days[ti // common.T]
        fv = blk.v.copy(); fq = blk.q.copy(); fk = blk.k.copy()
        for par in (0, 1):
            m = (dd % 2) == (1 - par)  # day parity 1-par is filled by the model fitted on parity par
            if not m.any():
                continue
            rate = np.vectorize(TG.RATE.get)(d["regime"][dd[m]]).astype(np.float32)
            pr = predict_cells(blk, mdl[par], ti[m], li[m], net, rate)
            fv[ti[m], li[m]] = np.clip(pr["y_v"], 1, common_cap())
            fq[ti[m], li[m]] = np.maximum(pr["y_q"], 1)
            fk[ti[m], li[m]] = np.maximum(pr["y_k"], 1e-3)
        dfill = {"obs_speed_kmh": fv.reshape(len(days), common.T, nl),
                 "obs_flow_vph": (fq * lanes).reshape(len(days), common.T, nl),
                 "obs_density_occ_linear_vehpkm": d["obs_density_occ_linear_vehpkm"][days],
                 "dates": d["dates"][days], "pct": d["pct"][days]}
        fb = F.Block(dfill, np.arange(len(days)), net, prof, version=2)
        for k, w in sel.items():
            if len(w) == 0:
                continue
            tt = (w.day.to_numpy() - lo) * common.T + w.t.to_numpy()
            ll = w.li.to_numpy()
            G = filled_features(fb, tt, ll, net, fv[tt, ll], fq[tt, ll], fk[tt, ll], blk.v)
            G["day"] = w.day.to_numpy(); G["t"] = w.t.to_numpy(); G["li"] = ll
            parts[k].append(G)
        print(f"{p} block {s} filled {miss.sum():,} {time.time()-t0:.0f}s", flush=True)
    (HERE / dsn).mkdir(parents=True, exist_ok=True)
    for k, ps in parts.items():
        pd.concat(ps, ignore_index=True).to_parquet(HERE / dsn / f"s2{k}_{p}.parquet", index=False)


def common_cap():
    return 120.75


def fit2(dsn, tag):
    ds = HERE / dsn
    def load(kind, p):
        a = pd.read_parquet(ds / f"{kind}_{p}.parquet")
        b = pd.read_parquet(ds / f"s2{kind}_{p}.parquet").drop(columns=["day", "t", "li"])
        assert len(a) == len(b)
        return pd.concat([a, b], axis=1)
    panels = [p for p in common.PANELS if (ds / f"s2fit_{p}.parquet").exists()]
    # one float32 feature matrix built in place (see lean_rows): same models, lower peak memory
    X, L = lean_rows([[ds / f"fit_{p}.parquet", ds / f"s2fit_{p}.parquet"] for p in panels])
    cols = list(X.columns)
    rng = np.random.default_rng(SEED)
    va = rng.random(len(X)) < 0.05
    ys = {"y_k": (L["y_q"] / L["y_v"].clip(lower=1.0)).to_numpy(np.float32) - L["p1_k"].to_numpy(np.float32)}
    for tgt in ("y_v", "y_q"):
        ys[tgt] = (L[tgt] - L["p1_" + tgt[-1]]).to_numpy(np.float32)
    wk = (L["lanes"] * L["len"]).to_numpy(np.float32)
    del L
    models = {}
    for tgt in TARGETS:
        m = TG.train_one(TG.params(), X, ys[tgt], wk if tgt == "y_k" else None, va, TG.ROUNDS, tgt)
        (HERE / "models").mkdir(parents=True, exist_ok=True)
        m.save_model(str(HERE / "models" / f"{tag}_{tgt}.txt"))
        models[tgt] = m
    del X
    rows_ = []
    (HERE / "preds").mkdir(parents=True, exist_ok=True)
    for p in panels:
        ev = load("eval", p)
        ev["rate"] = ev.regime.map(TG.RATE).astype(np.float32)
        X = ev[cols]
        pv = ev.p1_v.to_numpy() + models["y_v"].predict(X, num_iteration=models["y_v"].best_iteration)
        pq = ev.p1_q.to_numpy() + models["y_q"].predict(X, num_iteration=models["y_q"].best_iteration)
        pk = ev.p1_k.to_numpy() + models["y_k"].predict(X, num_iteration=models["y_k"].best_iteration)
        pd.DataFrame({"day": ev.day, "t": ev.t, "li": ev.li, "pv": pv.astype(np.float32), "pq": pq.astype(np.float32),
                      "pk": pk.astype(np.float32)}).to_parquet(HERE / "preds" / f"{tag}_{p}.parquet", index=False)
        r = TG.PanelEval(p, ev)(pv, pq)
        rows_.append(r)
        print(p, metrics.summarize(r), flush=True)
    s = metrics.summarize(pd.concat(rows_))
    print("STAGE2", s, metrics.contribution(s))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit1":
        fit1(sys.argv[2])
    elif cmd == "rows":
        rows(sys.argv[2], sys.argv[3])
    elif cmd == "fit2":
        fit2(sys.argv[2], sys.argv[3])
