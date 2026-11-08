"""Self-supervised split adaptation, small check on train.

A month of a scored split comes from its own scenario draw, but its visible
observations are public input for Task 1. This checks whether a month-specific
correction learned only from that month's visible cells helps:

  1. stage model M on train months 2030-06..2031-01 (actual release masks)
  2. in 2031-02, hide a further EXTRA share of the visible eligible cells
     (pseudo targets, labels = their published observations), predict them with M
  3. a small residual model R on those pseudo targets (features + M's output)
  4. score M and M+R on February's real release targets (truth from the unmasked layer)

No masked value is read or inferred in steps 1-3: R only models published
observations of the month, which the rules allow as input.
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
from pipeline import reconcile, RATE, SPEED_CAP

HERE = Path(__file__).resolve().parent
PANELS = os.environ.get("SS_PANELS", "D12_I5_N,D7_I10_W,D12_I405_N").split(",")
EXTRA = float(os.environ.get("SS_EXTRA", 0.15))
# control: take the pseudo targets from a month stage 1 already trained on
PSEUDO_MONTH = os.environ.get("SS_PSEUDO_MONTH", "2031-02")
SEED = 20261026
TARGETS = ("y_v", "y_q", "y_k")


def windows_gaps(p, d):
    wins = common.train_windows(p, d, np.random.default_rng(SEED + common.PANELS.index(p)))
    nd = len(d["dates"])
    g = np.zeros(nd * common.T, bool)
    for di0, s0, _ in wins:
        a = di0 * common.T + s0
        g[a + 1:a + 19] = True
    return g.reshape(nd, common.T), wins


def rows_for(d, p, net, prof, days, cells, gaps, wins, extra=None):
    """Features for given cells (day, t, li) inside a block of days."""
    lo, hi = days[0], days[-1] + 1
    nl = len(net)
    eb = np.broadcast_to(gaps[days][:, :, None], (len(days), common.T, nl))
    if extra is not None:
        eb = eb | extra
    bw = [((di0 - lo) * common.T + s0, c) for di0, s0, c in wins if lo <= di0 < hi]
    blk = F.Block(d, days, net, prof, extra_blank=eb, version=4, windows=bw)
    di, ti, li = cells
    X = blk.features((di - lo) * common.T + ti, li, net)
    X["rate"] = np.vectorize(RATE.get)(d["regime"][di]).astype(np.float32)
    return X


def label(d, lanes, di, ti, li, src="true"):
    v = d["true_speed"][di, ti, li] if src == "true" else d["obs_speed_kmh"][di, ti, li]
    q = (d["true_flow"][di, ti, li] if src == "true" else d["obs_flow_vph"][di, ti, li]) / lanes[li]
    return v.astype(np.float32), q.astype(np.float32)


def main():
    t0 = time.time()
    fit, feb_real, feb_pseudo = [], {}, {}
    rng = np.random.default_rng(SEED)
    for p in PANELS:
        d = common.load(p, "train")
        net = common.network(p)
        lanes = net.lanes.to_numpy(np.float32)
        prof = F.profiles([d], lanes)
        nd, nl = len(d["dates"]), len(lanes)
        month = np.array([x[:7] for x in d["dates"]])
        febd = np.where(month == "2031-02")[0]
        gaps, wins = windows_gaps(p, d)
        # 1. fit rows from the other months, in blocks
        for s in range(0, febd[0], 21):
            days = np.arange(max(0, s - 1), min(febd[0], s + 22))
            tg = d["target"][days]
            di, ti, li = np.nonzero(tg)
            di = days[di]
            keep = (di >= s) & (di < min(febd[0], s + 21)) & (rng.random(len(di)) < 0.06)
            cells = (di[keep], ti[keep], li[keep])
            X = rows_for(d, p, net, prof, days, cells, gaps, wins)
            X["y_v"], X["y_q"] = label(d, lanes, *cells)
            fit.append(X)
        # 2. February: real targets (original field) and pseudo targets (extra hidden)
        pmd = np.where(month == PSEUDO_MONTH)[0]
        pdays = np.arange(max(0, pmd[0] - 1), min(nd, pmd[-1] + 2))
        obs_el = d["elig"][pdays] & np.isfinite(d["obs_speed_kmh"][pdays]) & ~d["target"][pdays]
        obs_el &= (month[pdays] == PSEUDO_MONTH)[:, None, None]  # pad days are outside the month
        gapd = gaps[pdays][:, :, None]
        extra = obs_el & ~gapd & (rng.random(obs_el.shape) < EXTRA)
        di, ti, li = np.nonzero(extra)
        cells = (pdays[di], ti, li)
        Xp = rows_for(d, p, net, prof, pdays, cells, gaps, wins, extra=extra)
        days = np.arange(febd[0] - 1, nd)
        Xp["y_v"], Xp["y_q"] = label(d, lanes, *cells, src="obs")
        Xp["li_id"] = li; Xp["panel_id"] = common.PANELS.index(p)
        feb_pseudo[p] = Xp
        tg = d["target"][days]; tg[0] = False
        di, ti, li = np.nonzero(tg)
        cells = (days[di], ti, li)
        Xr = rows_for(d, p, net, prof, days, cells, gaps, wins)
        Xr["y_v"], Xr["y_q"] = label(d, lanes, *cells)
        Xr["li_id"] = li; Xr["panel_id"] = common.PANELS.index(p)
        Xr["day"], Xr["t"], Xr["li"] = days[di], ti, li
        feb_real[p] = Xr
        print(p, "fit", sum(len(x) for x in fit), "pseudo", len(Xp), "real", len(Xr), f"{time.time()-t0:.0f}s", flush=True)
    fit = pd.concat(fit, ignore_index=True)
    cols = [c for c in fit.columns if c not in ("y_v", "y_q")]
    prm = dict(objective="l2", learning_rate=0.1, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.7,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=common.THREADS, seed=SEED,
               deterministic=True, force_row_wise=True, verbose=-1)
    M = {}
    for tgt in TARGETS:
        if tgt == "y_k":
            y = (fit.y_q / fit.y_v.clip(lower=1)) - fit.base_q / fit.base_v.clip(lower=1)
            w = fit.lanes * fit.len
        else:
            y = fit[tgt] - fit["base_" + tgt[-1]]; w = None
        M[tgt] = lgb.train(prm, lgb.Dataset(fit[cols], y.to_numpy(np.float32), weight=w), 700)
        print("M", tgt, f"{time.time()-t0:.0f}s", flush=True)

    def predict_M(X):
        pv = X.base_v.to_numpy() + M["y_v"].predict(X[cols], num_threads=common.THREADS)
        pq = X.base_q.to_numpy() + M["y_q"].predict(X[cols], num_threads=common.THREADS)
        pk = (X.base_q / X.base_v.clip(lower=1)).to_numpy() + M["y_k"].predict(X[cols], num_threads=common.THREADS)
        return pv, pq, pk

    # 3. residual model R on the month's pseudo targets
    P = pd.concat(feb_pseudo.values(), ignore_index=True)
    pv, pq, pk = predict_M(P)
    P["m_v"], P["m_q"], P["m_k"] = pv, pq, pk
    rcols = ["m_v", "m_q", "m_k", "tod", "wd", "pos", "lanes", "vf", "cap_l", "len", "li_id", "panel_id",
             "gp", "gn", "rate", "cov_t", "pv", "pq", "rv_all", "rq_s2.0", "rv_s2.0"]
    rprm = dict(objective="l2", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=common.THREADS, seed=SEED,
                deterministic=True, force_row_wise=True, verbose=-1)
    R = {}
    ky = (P.y_q / P.y_v.clip(lower=1)) - P.m_k
    for tgt, y in (("v", P.y_v - P.m_v), ("q", P.y_q - P.m_q), ("k", ky)):
        R[tgt] = lgb.train(rprm, lgb.Dataset(P[rcols], y.to_numpy(np.float32), categorical_feature=["li_id", "panel_id"]), 300)
    print("pseudo residual sd v/q", float((P.y_v - P.m_v).std()), float((P.y_q - P.m_q).std()), flush=True)

    # 4. score February real targets
    out = {k: [] for k in ("M", "M+R", "M+R/2", "M+rc", "M+R+rc")}
    for p, X in feb_real.items():
        d = common.load(p, "train")
        lanes = common.network(p).lanes.to_numpy(np.float32)
        pv, pq, pk = predict_M(X)
        X["m_v"], X["m_q"], X["m_k"] = pv, pq, pk
        rv, rq, rk = (R[t].predict(X[rcols], num_threads=common.THREADS) for t in ("v", "q", "k"))
        days = np.unique(X.day.to_numpy())
        pos = {x: i for i, x in enumerate(days)}
        idx = (X.day.map(pos).to_numpy(), X.t.to_numpy(), X.li.to_numpy())
        sc = metrics.Scorer(p, d, days)
        shape = (len(days), common.T, len(lanes))
        li = X.li.to_numpy()

        def score(v, q):
            a = np.zeros(shape); b = np.zeros(shape)
            a[idx] = np.clip(v, 1, SPEED_CAP); b[idx] = np.maximum(q, 0) * lanes[li]
            return sc.score(a, b)
        out["M"].append(score(pv, pq))
        out["M+R"].append(score(pv + rv, pq + rq))
        out["M+R/2"].append(score(pv + rv / 2, pq + rq / 2))
        out["M+rc"].append(score(*reconcile(np.clip(pv, 1, SPEED_CAP), pq, pk, 1.0, 0.2, 0.8, 80.0)))
        out["M+R+rc"].append(score(*reconcile(np.clip(pv + rv, 1, SPEED_CAP), pq + rq, pk + rk, 1.0, 0.2, 0.8, 80.0)))
    for k, rows in out.items():
        df = pd.concat(rows)
        s = metrics.summarize(df)
        print(k, {a: round(b, 5) for a, b in s.items()}, "contrib", round(metrics.contribution(s), 5),
              "| per panel S_state", df.groupby("panel").S_state.mean().round(4).to_dict(), flush=True)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
