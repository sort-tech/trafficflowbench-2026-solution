"""How unusual is the LB drop of the network mixes? Each train month of the 719-window set is a validation-like
replica (5 ongoing windows per panel). Prints the per-month dS_q of go30/go50/go70 vs i5 (onset unchanged, so
dS_q = half the ongoing change) and a bootstrap of 5 ongoing windows per panel.

Usage: lb_gap.py <run dir> [...]
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task2")); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from mix_eval import combine
from mix_eval5 import H5_STEPS, load, thr_for
from model import iou
from combine_eval import BASES, load_nn

LB = {0.3: -0.00114 / 0.3, 0.5: -0.00143 / 0.3, 0.7: -0.00241 / 0.3}   # 11-02 LB total change / Task 2 weight
W = (0.3, 0.5, 0.7)
G, kt = load(); NN = load_nn([Path(a) for a in sys.argv[1:]])
rows = []
for i, k in enumerate(G.kg):
    r = G.go[k]; pl = combine({v: r[f"{v}_0"] for v in BASES["i5"]}, BASES["i5"]); th = thr_for(int(kt[i]), H5_STEPS)
    d = {"panel": k[0], "month": r["month"]}
    for w in (0.0,) + W:
        d[w] = iou((1 - w) * pl + w * NN[k]["nn_0"] >= th, r["y"], r["e"])
    rows.append(d)
d = pd.DataFrame(rows)


def dS(x):
    pm = x.groupby("panel")[[0.0, *W]].mean()
    f = pm.groupby(pm.index.map(lambda p: p.rsplit("_", 1)[0])).mean().mean()
    return [(f[w] - f[0.0]) / 2 for w in W]


o = pd.DataFrame([[m] + dS(x) for m, x in d.groupby("month")], columns=["month"] + [f"go{int(w * 100)}" for w in W])
print("dS_q per month:"); print(o.round(4).to_string(index=False))
print("LB", {f"go{int(w * 100)}": round(v, 4) for w, v in LB.items()})
rng = np.random.default_rng(0)
g = [x for _, x in d.groupby("panel")]
B = np.array([dS(pd.concat([x.iloc[rng.integers(0, len(x), 5)] for x in g])) for _ in range(4000)])
for j, w in enumerate(W):
    print(f"bootstrap go{int(w * 100)}: mean {B[:, j].mean():+.4f} sd {B[:, j].std():.4f} P(<= LB) {(B[:, j] <= LB[w]).mean():.4f}")
print("corr of the three deltas", np.corrcoef(B.T).round(2).tolist())
print("P(all three <= LB)", ((B[:, 0] <= LB[0.3]) & (B[:, 1] <= LB[0.5]) & (B[:, 2] <= LB[0.7])).mean())
