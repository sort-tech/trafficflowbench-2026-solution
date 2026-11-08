"""Gap cells: x3 + gbm9gap + a v5 gap model mixed with separate weights for predicted
free flow and congestion (x3 speed >= / < 80 km/h). Same scoring as exp_gap5.py.

    python -I exp_gap5_mix.py <tag>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import common
import metrics
import train_gbm as TG
from pipeline import reconcile

HERE = Path(__file__).resolve().parent
tag = sys.argv[1]
# (gbm9gap, new) weights in free flow and in congestion; x3 takes the rest
MIX = {"x3g05": ((0.5, 0), (0.5, 0)), "new 0.5": ((0, 0.5), (0, 0.5)),
       "free old / cong new": ((0.5, 0), (0, 0.5)), "free old / cong new 0.7": ((0.5, 0), (0, 0.7)),
       "thirds": ((1 / 3, 1 / 3), (1 / 3, 1 / 3)), "0.25/0.25 free, 0.5 new cong": ((0.25, 0.25), (0, 0.5)),
       "0.3/0.3": ((0.3, 0.3), (0.3, 0.3)), "free 0.4/0.1, cong 0.1/0.5": ((0.4, 0.1), (0.1, 0.5))}
rows = {k: [] for k in MIX}
for p in common.PANELS:
    ev = pd.read_parquet(HERE / "ds4" / f"eval_{p}.parquet", columns=["day", "t", "li", "win_cond", "regime"])
    base = pd.read_parquet(HERE / "preds" / f"gbm8s2_x3_{p}.parquet")
    old = pd.read_parquet(HERE / "preds" / f"gbm9gap_{p}.parquet")
    g = ev.win_cond.to_numpy() > 0
    new = pd.read_parquet(HERE / "preds" / f"{tag}gaponly_{p}.parquet") if g.any() else None
    pe = TG.PanelEval(p, ev)
    cong = base.pv.to_numpy()[g] < 80
    for name, (wf, wc) in MIX.items():
        arr = {c: base[c].to_numpy().copy() for c in ("pv", "pq", "pk")}
        if g.any():
            wo = np.where(cong, wc[0], wf[0])
            wn = np.where(cong, wc[1], wf[1])
            for c, a in arr.items():
                a[g] = wo * old[c].to_numpy()[g] + wn * new[c].to_numpy() + (1 - wo - wn) * a[g]
        v, q = reconcile(arr["pv"], arr["pq"], arr["pk"], 1.0, 0.2, 0.8, 80.0)
        rows[name].append(pe(v, q))
    print(p, flush=True)
out = [(k, metrics.contribution(metrics.summarize(pd.concat(r)))) for k, r in rows.items()]
S = pd.DataFrame(out, columns=["mix", "contrib"]).set_index("mix")
S["d_vs_x3g05"] = S.contrib - S.loc["x3g05", "contrib"]
print(S.round(5).to_string())
S.to_csv(HERE / "logs" / f"exp_gap5_mix_{tag}.csv")
