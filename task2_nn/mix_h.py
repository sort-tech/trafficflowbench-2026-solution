"""Out-of-fold check of horizon-dependent network weights on ongoing windows (i5 base, onset LightGBM only):
p_h = (1 - w_h) p_lgb + w_h p_nn with w_h = w for h1-h4, w5 for h5, w6 for h6. Also per-panel ongoing IoU.

Usage: mix_h.py <run dir> [...]   (network probabilities averaged)
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task2")); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from common import H, aggregate
from mix_eval import combine
from mix_eval5 import H5_STEPS, load, thr_for
from model import best_k_set, iou
from combine_eval import BASES, SH, load_nn

G, kt = load(); NN = load_nn([Path(a) for a in sys.argv[1:]]); base = BASES["i5"]


def on(sh):
    out = []
    for k in G.ko:
        r = G.on[k]; pr = np.zeros(r["y"].shape, bool)
        pr[H - 1] = best_k_set(0.5 * r[f"v1_{sh}"] + 0.5 * r[f"v1_thr_{sh}"]); out.append(iou(pr, r["y"], r["e"]))
    return np.array(out)


def go(w, sh):
    out = []
    for i, k in enumerate(G.kg):
        r = G.go[k]; pl = combine({v: r[f"{v}_{sh}"] for v in base}, base)
        out.append(iou((1 - w[:, None]) * pl + w[:, None] * NN[k][f"nn_{sh}"] >= thr_for(int(kt[i]), H5_STEPS), r["y"], r["e"]))
    return np.array(out)


ON = {s: on(s) for s in (0,) + SH}
S = lambda a, b: aggregate(G.meta.assign(iou=np.concatenate([a, b])), "iou")
ref = {s: go(np.zeros(H), s) for s in (0,) + SH}
r50 = go(np.full(H, .5), 0)
rows = []
for w in (0.5, 0.7):
    for w5 in (0.0, 0.25, 0.5, 0.7):
        for w6 in (0.0, 0.25, 0.5, 0.7):
            wv = np.array([w] * 4 + [w5, w6])
            g = {s: go(wv, s) for s in (0,) + SH}
            d = g[0] - r50
            a = S(ON[0], g[0])
            rows.append({"w": w, "w5": w5, "w6": w6, "S_q": a["S_queue"], "ongoing": a["ongoing"],
                         "dS_vs_i5": a["S_queue"] - S(ON[0], ref[0])["S_queue"],
                         "dS_vs_go50": a["S_queue"] - S(ON[0], r50)["S_queue"], "se_vs_go50": d.std() / np.sqrt(len(d) + len(ON[0])),
                         "better": int((d > 1e-9).sum()), "worse": int((d < -1e-9).sum()),
                         "shift": np.mean([S(ON[s], g[s])["S_queue"] - S(ON[s], ref[s])["S_queue"] for s in SH]),
                         "D7_I210_E": a["per_panel"]["D7_I210_E"]})
res = pd.DataFrame(rows).sort_values("S_q", ascending=False)
pd.set_option("display.width", 220)
print(res.round(4).to_string(index=False))
tag = "+".join(Path(a).name for a in sys.argv[1:])
res.to_csv(HERE / "logs" / f"mix_h_{tag}.csv", index=False)
