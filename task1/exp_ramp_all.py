"""Ramp flow + mainline conservation features (v6 g8_*, v7 g9_*) on every masked cell.

Same frame as exp_weights.py: ds4 fit/eval rows (ds12 = ds4 rows with FEAT=7, v4
columns identical), lr 0.1, 500 rounds, speed clip + density reconcile (0.2/0.8/80).
Arms differ only in the feature columns:
    base : v4
    g9   : v4 + conservation (g9_*)
    ramp : v4 + ramp (g8_*) + conservation (g9_*)
Scored per regime, also split into Task 2 gap cells and the rest.

    python -I exp_ramp_all.py [ds12]
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightgbm as lgb
import numpy as np
import pandas as pd

import common
import train_gbm as TG
from pipeline import reconcile, SPEED_CAP

HERE = Path(__file__).resolve().parent
DS = HERE / (sys.argv[1] if len(sys.argv) > 1 else "ds12")
TAG = os.environ.get("EXP_TAG", "r12")
ROUNDS, LR = int(os.environ.get("EXP_ROUNDS", 500)), 0.1
PANELS = [p for p in common.PANELS if (DS / f"eval_{p}.parquet").exists()]
NEW = ("g5_", "g6_", "g7_", "g8_", "g9_")
ARMS = {"base": (), "g9": ("g9_",), "ramp": ("g8_", "g9_")}


def cols_for(all_cols, keep):
    return [c for c in all_cols if not c.startswith(NEW) or c.startswith(keep)] if keep else \
        [c for c in all_cols if not c.startswith(NEW)]


def train():
    TG.DS = DS
    fit = TG.load_fit(PANELS)
    allc = TG.feat_cols(fit)
    print("fit rows", len(fit), flush=True)
    (HERE / "models").mkdir(parents=True, exist_ok=True)
    out = {}
    for arm, keep in ARMS.items():
        cols = cols_for(allc, keep)
        out[arm] = cols
        for tgt in ("y_v", "y_q", "y_k"):
            f = HERE / "models" / f"{TAG}_{arm}_{tgt}.txt"
            if f.exists():
                continue
            if tgt == "y_k":
                y = TG.k_true(fit) - TG.k_base(fit)
                w = (fit.lanes * fit.len).to_numpy(np.float32)
            else:
                y = (fit[tgt] - fit["base_" + tgt[-1]]).to_numpy(np.float32)
                w = None
            prm = TG.params()
            prm["learning_rate"] = LR
            t0 = time.time()
            m = lgb.train(prm, lgb.Dataset(fit[cols], y, weight=w), ROUNDS)
            m.save_model(str(f))
            print(arm, tgt, len(cols), "features", f"{time.time()-t0:.0f}s", flush=True)
    return out


def rmse(e):
    return float(np.sqrt(np.mean(e * e))) if len(e) else np.nan


def evaluate(arm_cols):
    M = {a: {t: lgb.Booster(model_file=str(HERE / "models" / f"{TAG}_{a}_{t}.txt")) for t in ("y_v", "y_q", "y_k")}
         for a in ARMS}
    rows, rm = [], []
    for p in PANELS:
        ev = pd.read_parquet(DS / f"eval_{p}.parquet")
        ev["rate"] = ev.regime.map(TG.RATE).astype(np.float32)
        pe = TG.PanelEval(p, ev)
        d = common.load(p, "train")
        li = ev.li.to_numpy()
        vmin = np.nanmin(d["true_speed"], axis=(0, 1))
        qmax = np.nanmax(d["true_flow"], axis=(0, 1)) / pe.lanes
        yv, yq = ev.y_v.to_numpy(), ev.y_q.to_numpy()
        cong = yv < 60
        gap = ev.win_cond.to_numpy() > 0
        for a, cols in arm_cols.items():
            X = ev[cols]
            pv = np.clip(ev.base_v.to_numpy() + M[a]["y_v"].predict(X, num_threads=common.THREADS), vmin[li], SPEED_CAP)
            pq = ev.base_q.to_numpy() + M[a]["y_q"].predict(X, num_threads=common.THREADS)
            pk = TG.k_base(ev) + M[a]["y_k"].predict(X, num_threads=common.THREADS)
            v, q = reconcile(pv, pq, pk, 1.0, 0.2, 0.8, 80.0)
            q = np.clip(q, 1.0, 1.1 * qmax[li])
            r = pe(v, q)
            r["arm"] = a
            rows.append(r)
            ev_, eq_ = v - yv, q - yq
            for part, m in (("all", np.ones_like(gap)), ("gap", gap), ("non-gap", ~gap)):
                rm.append((p, a, part, rmse(ev_[m & cong]), rmse(ev_[m & ~cong]), rmse(eq_[m & cong]), rmse(eq_[m & ~cong]),
                           rmse(ev_[m]), rmse(eq_[m]), int(m.sum())))
        print(p, flush=True)
    df = pd.concat(rows)
    df["family"] = df.panel.map(common.FAMILY)
    df["contrib"] = 0.35 * df.S_state + 0.15 * df.S_physics
    agg = df.groupby(["arm", "regime", "family"])[["contrib", "S_state", "S_LWR"]].mean().groupby(["arm", "regime"]).mean()
    tot = agg.contrib.unstack()
    tot["all"] = tot.mean(1)
    print(tot.round(5).to_string())
    print((tot - tot.loc["base"]).round(5).to_string())
    print(agg.round(5).to_string())
    R = pd.DataFrame(rm, columns=["panel", "arm", "part", "v_cong", "v_free", "q_cong", "q_free", "v", "q", "n"])
    # cell-weighted pooled RMSE over panels
    W = R.assign(**{c: R[c] ** 2 * R.n for c in ("v", "q")}).groupby(["part", "arm"])[["v", "q", "n"]].sum()
    W["v"] = np.sqrt(W.v / W.n); W["q"] = np.sqrt(W.q / W.n)
    print(W.round(3).to_string())
    print(R.groupby(["part", "arm"])[["v_cong", "v_free", "q_cong", "q_free"]].mean().round(3).to_string())
    (HERE / "logs").mkdir(parents=True, exist_ok=True)
    df.to_csv(HERE / "logs" / "exp_ramp_all.csv", index=False)
    R.to_csv(HERE / "logs" / "exp_ramp_all_rmse.csv", index=False)
    tot.to_csv(HERE / "logs" / "exp_ramp_all_contrib.csv")


if __name__ == "__main__":
    evaluate(train())
