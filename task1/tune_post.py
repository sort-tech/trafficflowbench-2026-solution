"""Post-processing fine-tuning on eval-fold predictions.

The overall contribution 0.35*S_state + 0.15*S_physics is a mean over regimes of
per-regime terms (panel -> family -> overall means are linear), so the
reconcile setting can be chosen per regime independently.

    python -I tune_post.py <ds> <tag[:w],tag[:w],...>      # weighted blend of eval predictions
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import common
from pipeline import reconcile, SPEED_CAP
from train_gbm import PanelEval

HERE = Path(__file__).resolve().parent
dsn = sys.argv[1]
blend = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in sys.argv[2].split(",")]
BF = (0.0, 0.1, 0.2, 0.3, 0.4)
BC = (0.6, 0.7, 0.8, 0.9, 1.0)
VT = (70.0, 80.0, 90.0)
rows = []
for p in common.PANELS:
    ev = pd.read_parquet(HERE / dsn / f"eval_{p}.parquet", columns=["day", "t", "li"])
    ws = sum(w for _, w in blend)
    prs = [(pd.read_parquet(HERE / "preds" / f"{t}_{p}.parquet"), w / ws) for t, w in blend]
    pv = sum(w * pr.pv.to_numpy() for pr, w in prs)
    pq = sum(w * pr.pq.to_numpy() for pr, w in prs)
    pk = sum(w * pr.pk.to_numpy() for pr, w in prs)
    pe = PanelEval(p, ev)
    lanes = pe.lanes
    d = common.load(p, "train")
    li = ev.li.to_numpy()
    vmin = np.nanmin(d["true_speed"], axis=(0, 1))
    qmax = np.nanmax(d["true_flow"], axis=(0, 1)) / lanes
    pvc = np.clip(pv, vmin[li], SPEED_CAP)
    for bf, bc, vt in itertools.product(BF, BC, VT):
        v, q = reconcile(pvc, pq, pk, 1.0, bf, bc, vt)
        r = pe(v, np.clip(q, 1.0, 1.1 * qmax[li]))
        r["cfg"] = f"{bf}:{bc}:{vt}"
        rows.append(r)
    print(p, flush=True)
df = pd.concat(rows)
df["family"] = df.panel.map(common.FAMILY)
df["contrib"] = 0.35 * df.S_state + 0.15 * (df.S_FD / 3 + 2 * df.S_LWR / 3)
agg = df.groupby(["cfg", "regime", "family"]).contrib.mean().groupby(["cfg", "regime"]).mean().unstack()
best = agg.idxmax()
print("per-regime best:", best.to_dict(), "values", agg.max().round(5).to_dict())
print("single best cfg:", agg.mean(1).idxmax(), round(agg.mean(1).max(), 5))
print("per-regime choice total:", round(agg.max().mean(), 5))
print("reference 0.2:0.8:80.0:", round(agg.loc["0.2:0.8:80.0"].mean(), 5))
