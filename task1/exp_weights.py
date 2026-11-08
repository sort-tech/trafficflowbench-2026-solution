"""Guide 3-1 check: congestion sample weights for the speed and flow models.

Same ds4 fit/eval rows and features as gbm4, small fixed budget (lr 0.1, 500
rounds) so only the weights differ between arms:
    v: none | 2.5 where true speed < 60
    q: none | 1 / max(true speed, 15)
Scored after speed clip + density reconcile (0.2 / 0.8 / 80), per regime.

    python -I exp_weights.py
"""
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
DS = HERE / "ds4"
ROUNDS, LR = 500, 0.1


def weight(name, yv):
    if name == "v_w":
        return np.where(yv < 60, 2.5, 1.0).astype(np.float32)
    if name == "q_w":
        return (1.0 / np.maximum(yv, 15.0)).astype(np.float32)
    return None


def train():
    TG.DS = DS
    fit = TG.load_fit(common.PANELS)
    cols = TG.feat_cols(fit)
    yv = fit.y_v.to_numpy(np.float32)
    arms = {"v_b": ("y_v", None), "v_w": ("y_v", "v_w"), "q_b": ("y_q", None), "q_w": ("y_q", "q_w"),
            "k_b": ("y_k", "k")}
    (HERE / "models").mkdir(parents=True, exist_ok=True)
    for name, (tgt, wn) in arms.items():
        f = HERE / "models" / f"w31_{name}.txt"
        if f.exists():
            continue
        if tgt == "y_k":
            y = TG.k_true(fit) - TG.k_base(fit)
            w = (fit.lanes * fit.len).to_numpy(np.float32)
        else:
            y = (fit[tgt] - fit["base_" + tgt[-1]]).to_numpy(np.float32)
            w = weight(wn, yv)
        prm = TG.params()
        prm["learning_rate"] = LR
        t0 = time.time()
        m = lgb.train(prm, lgb.Dataset(fit[cols], y, weight=w), ROUNDS)
        m.save_model(str(f))
        print(name, f"{time.time()-t0:.0f}s", flush=True)
    return cols


def evaluate(cols):
    M = {n: lgb.Booster(model_file=str(HERE / "models" / f"w31_{n}.txt")) for n in ("v_b", "v_w", "q_b", "q_w", "k_b")}
    combos = {"base": ("v_b", "q_b"), "v_w": ("v_w", "q_b"), "q_w": ("v_b", "q_w"), "both": ("v_w", "q_w")}
    rows, rm = [], []
    for p in common.PANELS:
        ev = pd.read_parquet(DS / f"eval_{p}.parquet")
        ev["rate"] = ev.regime.map(TG.RATE).astype(np.float32)
        X = ev[cols]
        P = {n: m.predict(X, num_threads=common.THREADS) for n, m in M.items()}
        pk = TG.k_base(ev) + P["k_b"]
        pe = TG.PanelEval(p, ev)
        d = common.load(p, "train")
        li = ev.li.to_numpy()
        vmin = np.nanmin(d["true_speed"], axis=(0, 1))
        qmax = np.nanmax(d["true_flow"], axis=(0, 1)) / pe.lanes
        cong = ev.y_v.to_numpy() < 60
        for c, (vn, qn) in combos.items():
            pv = np.clip(ev.base_v.to_numpy() + P[vn], vmin[li], SPEED_CAP)
            pq = ev.base_q.to_numpy() + P[qn]
            v, q = reconcile(pv, pq, pk, 1.0, 0.2, 0.8, 80.0)
            q = np.clip(q, 1.0, 1.1 * qmax[li])
            r = pe(v, q)
            r["combo"] = c
            rows.append(r)
            ev_v, ev_q = v - ev.y_v.to_numpy(), q - ev.y_q.to_numpy()
            rm.append((p, c, np.sqrt((ev_v[cong] ** 2).mean()), np.sqrt((ev_v[~cong] ** 2).mean()),
                       np.sqrt((ev_q[cong] ** 2).mean()), np.sqrt((ev_q[~cong] ** 2).mean())))
        print(p, flush=True)
    df = pd.concat(rows)
    df["family"] = df.panel.map(common.FAMILY)
    df["contrib"] = 0.35 * df.S_state + 0.15 * df.S_physics
    agg = df.groupby(["combo", "regime", "family"])[["contrib", "S_state", "S_LWR"]].mean().groupby(["combo", "regime"]).mean()
    print(agg.round(5).to_string())
    tot = agg.contrib.unstack()
    tot["all"] = tot.mean(1)
    print((tot - tot.loc["base"]).round(5).to_string())
    R = pd.DataFrame(rm, columns=["panel", "combo", "v_cong", "v_free", "q_cong", "q_free"])
    print(R.groupby("combo").mean(numeric_only=True).round(3).to_string())
    (HERE / "logs").mkdir(parents=True, exist_ok=True)
    df.to_csv(HERE / "logs" / "exp_weights.csv", index=False)


if __name__ == "__main__":
    evaluate(train())
