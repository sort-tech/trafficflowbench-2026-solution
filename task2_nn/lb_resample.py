"""Decision test (main session, 11-02): window-level out-of-fold differences (go30 / go50 / go70 minus i5) resampled
10,000 times into the public LB layout (validation: 8 panels x 5 onset + 5 ongoing windows), scored with the S_q
aggregation (window -> (panel, condition) -> panel -> family), times the Task 2 weight 0.30, against the LB changes.

Usage: lb_resample.py <run dir> [...]
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task2")); sys.path.insert(0, str(HERE))
import numpy as np
from mix_eval import combine
from mix_eval5 import H5_STEPS, load, thr_for
from model import best_k_set, iou
from combine_eval import BASES, load_nn
from common import FAMILY, H

LB = {0.3: -0.00114, 0.5: -0.00143, 0.7: -0.00241}
W = (0.0, 0.3, 0.5, 0.7)
G, kt = load(); NN = load_nn([Path(a) for a in sys.argv[1:]])
panels = sorted({k[0] for k in G.kg})
on = {p: [] for p in panels}; go = {p: [] for p in panels}
for k in G.ko:          # onset is LightGBM in every candidate: same IoU for all weights
    r = G.on[k]; pr = np.zeros(r["y"].shape, bool)
    pr[H - 1] = best_k_set(0.5 * r["v1_0"] + 0.5 * r["v1_thr_0"]); on[k[0]].append([iou(pr, r["y"], r["e"])] * len(W))
for i, k in enumerate(G.kg):
    r = G.go[k]; pl = combine({v: r[f"{v}_0"] for v in BASES["i5"]}, BASES["i5"]); th = thr_for(int(kt[i]), H5_STEPS)
    go[k[0]].append([iou((1 - w) * pl + w * NN[k]["nn_0"] >= th, r["y"], r["e"]) for w in W])
on = {p: np.array(v) for p, v in on.items()}; go = {p: np.array(v) for p, v in go.items()}
fams = sorted(set(FAMILY[p] for p in panels))
rng = np.random.default_rng(20261102)
N = 10000
tot = np.zeros((N, len(W)))
for n in range(N):
    pm = {}
    for p in panels:   # panel score = mean of the two condition means (5 windows each)
        a = on[p][rng.integers(0, len(on[p]), 5)].mean(0); b = go[p][rng.integers(0, len(go[p]), 5)].mean(0)
        pm[p] = (a + b) / 2
    tot[n] = np.mean([np.mean([pm[p] for p in panels if FAMILY[p] == f], 0) for f in fams], 0)
d = 0.30 * (tot[:, 1:] - tot[:, :1])
full = {}
for j, w in enumerate(W[1:]):
    full[w] = d[:, j]
    print(f"go{int(w * 100)}: total-score difference mean {d[:, j].mean():+.5f}, sd {d[:, j].std():.5f}, "
          f"5% quantile {np.quantile(d[:, j], .05):+.5f}, LB {LB[w]:+.5f}, P(<= LB) {(d[:, j] <= LB[w]).mean():.4f}")
print("P(all three <= LB)", float(np.mean((d[:, 0] <= LB[0.3]) & (d[:, 1] <= LB[0.5]) & (d[:, 2] <= LB[0.7]))))
print("windows per panel: onset", {p: len(on[p]) for p in panels}, "ongoing", {p: len(go[p]) for p in panels})
