"""Paired stage-2 check: baseline (FEAT=4) vs ramp + conservation (FEAT=7) stage-2 eval
predictions on the train eval days. Both get gbm9gap blended at 0.5 inside the Task 2
gaps (eval rows identical to ds4/ds7), speed clip, density reconcile 0.2/0.8/80 and the
flow cap, then the proxy per regime and RMSE split into gap / non-gap, congested / free.

    python -I exp_s2pair.py <ds_base>:<tag_base> <ds_ramp>:<tag_ramp>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import common
import train_gbm as TG
from pipeline import reconcile, SPEED_CAP

HERE = Path(__file__).resolve().parent
ARMS = {"base": sys.argv[1].split(":"), "ramp": sys.argv[2].split(":")}


def rmse(e):
    return float(np.sqrt(np.mean(e * e))) if len(e) else np.nan


rows, rm = [], []
PANELS = [p for p in common.PANELS if all((HERE / "preds" / f"{t}_{p}.parquet").exists() for _, t in ARMS.values())]
for p in PANELS:
    ref = None
    for arm, (dsn, tag) in ARMS.items():
        ev = pd.read_parquet(HERE / dsn / f"eval_{p}.parquet", columns=["day", "t", "li", "regime", "win_cond", "y_v", "y_q"])
        if ref is None:
            ref = ev
            old = pd.read_parquet(HERE / "preds" / f"gbm9gap_{p}.parquet")
            assert (old.day.to_numpy() == ev.day.to_numpy()).all() and (old.li.to_numpy() == ev.li.to_numpy()).all()
            pe = TG.PanelEval(p, ev)
            d = common.load(p, "train")
            li = ev.li.to_numpy()
            vmin = np.nanmin(d["true_speed"], axis=(0, 1))
            qmax = np.nanmax(d["true_flow"], axis=(0, 1)) / pe.lanes
        assert (ev.day.to_numpy() == ref.day.to_numpy()).all() and (ev.t.to_numpy() == ref.t.to_numpy()).all()
        pr = pd.read_parquet(HERE / "preds" / f"{tag}_{p}.parquet")
        g = ev.win_cond.to_numpy() > 0
        a = {c: pr[c].to_numpy().astype(np.float64) for c in ("pv", "pq", "pk")}
        for c in a:
            a[c][g] = 0.5 * old[c].to_numpy()[g] + 0.5 * a[c][g]
        pv = np.clip(a["pv"], vmin[li], SPEED_CAP)
        v, q = reconcile(pv, a["pq"], a["pk"], 1.0, 0.2, 0.8, 80.0)
        q = np.clip(q, 1.0, 1.1 * qmax[li])
        r = pe(v, q)
        r["arm"] = arm
        rows.append(r)
        yv, yq = ev.y_v.to_numpy(), ev.y_q.to_numpy()
        cong = yv < 60
        e_v, e_q = v - yv, q - yq
        for part, m in (("all", np.ones_like(g)), ("gap", g), ("non-gap", ~g)):
            rm.append((p, arm, part, int(m.sum()), int((m & cong).sum()),
                       *(float((x[mm] ** 2).sum()) for x in (e_v, e_q) for mm in (m, m & cong, m & ~cong))))
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
R = pd.DataFrame(rm, columns=["panel", "arm", "part", "n", "n_cong", "sv", "sv_c", "sv_f", "sq", "sq_c", "sq_f"])
G = R.groupby(["part", "arm"]).sum(numeric_only=True)
nf = G.n - G.n_cong
out = pd.DataFrame({"n": G.n, "v": np.sqrt(G.sv / G.n), "v_cong": np.sqrt(G.sv_c / G.n_cong), "v_free": np.sqrt(G.sv_f / nf),
                    "q": np.sqrt(G.sq / G.n), "q_cong": np.sqrt(G.sq_c / G.n_cong), "q_free": np.sqrt(G.sq_f / nf)})
print(out.round(3).to_string())
(HERE / "logs").mkdir(parents=True, exist_ok=True)
df.to_csv(HERE / "logs" / "exp_s2pair.csv", index=False)
out.to_csv(HERE / "logs" / "exp_s2pair_rmse.csv")
tot.to_csv(HERE / "logs" / "exp_s2pair_contrib.csv")
