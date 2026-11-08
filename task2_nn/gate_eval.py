"""Out-of-fold check of gating the network mix on ongoing windows: the mix may not add a queue (0 -> 1 against i5) on
links free at T where the train climatology of a queue at T+h is >= c (c = 1.1 means no gate; c = 0 blocks every
added queue on free links). Release: cells changed vs i5 after the gate.

Usage: gate_eval.py <dump.pkl from build_mix.py --dump> <i5 csv> <run dir> [...]
"""
import pickle
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task2")); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from common import aggregate
from mix_eval import combine
from mix_eval5 import H5_STEPS, load, thr_for
from model import best_k_set, iou
from combine_eval import BASES, SH, load_nn
from change_types import clim_at, panel, qT
from common import H

G, kt = load(); NN = load_nn([Path(a) for a in sys.argv[3:]]); base = BASES["i5"]
W = (0.3, 0.5, 0.7); C = (1.1, 0.5, 0.3, 0.0)
prep = []
for i, k in enumerate(G.kg):
    z = panel(k[0]); j = int(np.nonzero((z["ev_t"] == k[1]) & (z["ev_cond"] == 0))[0][0])
    free = ~qT(z["ev_speed"][j], z["vcut"]); cl = clim_at(z, int(z["ev_fold"][j]), k[1])
    prep.append((free, cl, thr_for(int(kt[i]), H5_STEPS)))


def go(w, c, sh):
    out = []
    for i, k in enumerate(G.kg):
        r = G.go[k]; free, cl, th = prep[i]
        pl = combine({v: r[f"{v}_{sh}"] for v in base}, base)
        a = pl >= th; b = (1 - w) * pl + w * NN[k][f"nn_{sh}"] >= th
        block = free[None, :] & (cl >= c) & b & ~a
        out.append(iou(b & ~block, r["y"], r["e"]))
    return np.array(out)


def on(sh):
    out = []
    for k in G.ko:
        r = G.on[k]; pr = np.zeros(r["y"].shape, bool)
        pr[H - 1] = best_k_set(0.5 * r[f"v1_{sh}"] + 0.5 * r[f"v1_thr_{sh}"]); out.append(iou(pr, r["y"], r["e"]))
    return np.array(out)


ON = {s: on(s) for s in (0,) + SH}
S = lambda a, b: aggregate(G.meta.assign(iou=np.concatenate([a, b])), "iou")["S_queue"]
ref = {s: go(0.0, 1.1, s) for s in (0,) + SH}
rows = []
for w in W:
    for c in C:
        g = {s: go(w, c, s) for s in (0,) + SH}
        d = g[0] - ref[0]
        rows.append({"w": w, "gate_clim": c, "dS_q": S(ON[0], g[0]) - S(ON[0], ref[0]), "se": d.std() / np.sqrt(len(d) + len(ON[0])),
                     "better": int((d > 1e-9).sum()), "worse": int((d < -1e-9).sum()),
                     "shift": np.mean([S(ON[s], g[s]) - S(ON[s], ref[s]) for s in SH])})
# release cell changes after the gate
D = pickle.load(open(sys.argv[1], "rb")); i5 = pd.read_csv(sys.argv[2]).groupby("window_id").queue_pred.sum()
rel = {}
for wid, v in D.items():
    p = wid[3:].rsplit("_", 2)[0]; s = "validation" if "_validation_" in wid else "private"; z = panel(p)
    j = int(np.nonzero(z[f"rel_{s}_wid"] == wid)[0][0]); t = int(z[f"rel_{s}_t"][j])
    th = min((0.31, 0.37), key=lambda x: abs(int((v["p"] >= x).sum()) - int(i5[wid])))
    free = ~qT(z[f"rel_{s}_speed"][j], z["vcut"])[v["link"]]; cl = clim_at(z, 3, t)[v["h"] - 1, v["link"]]
    a = v["p"] >= th
    for w in W:
        b = (1 - w) * v["p"] + w * v["q"] >= th
        for c in C:
            fin = b & ~(free & (cl >= c) & b & ~a)
            key = (w, c, s); rel[key] = rel.get(key, 0) + int((fin != a).sum())
res = pd.DataFrame(rows)
res["val_cells"] = [rel[(r.w, r.gate_clim, "validation")] for r in res.itertuples()]
res["prv_cells"] = [rel[(r.w, r.gate_clim, "private")] for r in res.itertuples()]
pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))
