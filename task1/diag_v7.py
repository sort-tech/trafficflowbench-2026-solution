"""Cells where v7 and ss3f flows differ by more than 2000 vph: what the v7 features saw.

For each such cell (validation / private template rows): Task 2 gap or not, link distance to
the nearest observed links u and d, ramps attached in (u, d] and their observed share,
the conservation estimate g9_est against base_q, and the v7 / ss3f values. The same
features over all template cells give the reference shares.

    python -I diag_v7.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import common
import features as F
import pipeline

HERE = Path(__file__).resolve().parent
A = HERE / "candidates" / "gbm_v8s2x3g_ss3f_fixed" / "state_submission.csv"
B = HERE / "candidates" / "gbm_v8s2rx3g_ss3f_ramp" / "state_submission.csv"
COLS = ["g9_lu", "g9_ld", "g9_ramp_obs", "g9_est", "g9_est_up", "g9_est_dn", "g9_q_up", "g9_q_dn", "g9_net_up",
        "g9_net_dn", "base_q", "base_v", "win_cond", "lanes", "cnt_W2", "gp", "gn"]


def main():
    use = ["panel", "timestamp", "link_id", "speed_kmh", "flow_vph"]
    a = pd.read_csv(A, usecols=use)
    b = pd.read_csv(B, usecols=use)
    assert (a.panel.values == b.panel.values).all() and (a.timestamp.values == b.timestamp.values).all()
    dq = b.flow_vph - a.flow_vph
    big = (dq.abs() > 2000).to_numpy()
    print("cells", len(a), "|dq|>2000", int(big.sum()), "|dq|>4000", int((dq.abs() > 4000).sum()), flush=True)
    rows = []
    ref = []
    for p in common.PANELS:
        for split in ("validation", "private"):
            d = common.load(p, split)
            tr = common.load(p, "train")
            net = common.network(p)
            lanes = net.lanes.to_numpy(np.float32)
            tpl = common.read_template(p, split)
            sel = (a.panel == p).to_numpy() & a.timestamp.str[:10].isin(set(map(str, d["dates"]))).to_numpy()
            if not sel.any():
                continue
            idx = np.nonzero(sel)[0]
            sub = a.iloc[idx]
            di, ti, li = common.template_index(sub, d["dates"], d["links"])
            wins = [(x * common.T + s0, c) for x, s0, c in common.split_windows(p, split, d["dates"])]
            blk = F.Block(d, np.arange(len(d["dates"])), net, F.profiles([tr, d], lanes),
                          ramps=pipeline.split_ramps(p, split, d, tr), version=7, windows=wins)
            X = blk.features(di * common.T + ti, li, net)[COLS]
            X["panel"], X["split"] = p, split
            X["timestamp"], X["link_id"] = sub.timestamp.to_numpy(), sub.link_id.to_numpy()
            X["q_ss3f"], X["q_v7"] = a.flow_vph.to_numpy()[idx], b.flow_vph.to_numpy()[idx]
            X["v_ss3f"], X["v_v7"] = a.speed_kmh.to_numpy()[idx], b.speed_kmh.to_numpy()[idx]
            X["big"] = big[idx]
            ref.append(X.drop(columns=["timestamp", "link_id"]))
            if X.big.any():
                rows.append(X[X.big])
            print(p, split, int(X.big.sum()), flush=True)
    R = pd.concat(ref, ignore_index=True)
    D = pd.concat(rows, ignore_index=True)
    for name, T_ in (("all template cells", R), ("|dq|>2000", D)):
        span = T_.g9_lu + T_.g9_ld
        ratio = T_.g9_est / T_.base_q.clip(lower=1)
        print(f"\n{name}: n={len(T_)}")
        print("  gap share", round(float((T_.win_cond > 0).mean()), 3))
        print("  u..d span (links) median", float(np.nanmedian(span)), "share > 10", round(float((span > 10).mean()), 3),
              "share no u or no d", round(float((T_.g9_lu.isna() | T_.g9_ld.isna()).mean()), 3))
        print("  ramp obs share median", float(np.nanmedian(T_.g9_ramp_obs)))
        print("  g9_est / base_q outside +-50%", round(float(((ratio < 0.5) | (ratio > 1.5)).mean()), 3),
              "median", round(float(np.nanmedian(ratio)), 3))
    D["q_v7_lane"] = D.q_v7 / D.lanes
    D["q_ss3f_lane"] = D.q_ss3f / D.lanes
    show = ["panel", "split", "timestamp", "link_id", "lanes", "win_cond", "g9_lu", "g9_ld", "g9_ramp_obs", "base_q",
            "g9_est", "g9_q_up", "g9_net_up", "g9_q_dn", "g9_net_dn", "q_ss3f_lane", "q_v7_lane", "v_ss3f", "v_v7"]
    D["absdq"] = (D.q_v7 - D.q_ss3f).abs()
    print(D.sort_values("absdq", ascending=False)[show].head(25).round(1).to_string())
    print(D.groupby(["panel", "split"]).size().to_string())
    (HERE / "logs").mkdir(exist_ok=True)
    D.to_csv(HERE / "logs" / "diag_v7_big.csv", index=False)


if __name__ == "__main__":
    main()
