"""Where the network loses to LightGBM on ongoing windows (out-of-fold 719-window set): per panel, error cells by
link position (upstream end = first 5 links) and horizon, and the queue state at T of the windows that lose most.

Usage: d7_diag.py <run dir> [...]   (network probabilities averaged)
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

G, kt = load(); NN = load_nn([Path(a) for a in sys.argv[1:]]); base = BASES["i5"]
Q = {}
rows, cells = [], []
for i, k in enumerate(G.kg):
    panel, t = k
    if panel not in Q:
        Q[panel] = np.load(HERE / "data" / f"{panel}.npz")["q"]
    r = G.go[k]; y, e = r["y"].astype(bool), r["e"].astype(bool)
    pl = combine({v: r[f"{v}_0"] for v in base}, base); pn = NN[k]["nn_0"]; th = thr_for(int(kt[i]), H5_STEPS)
    pm = 0.5 * pl + 0.5 * pn
    qT = Q[panel][t].astype(bool)
    L = y.shape[1]
    rows.append({"panel": panel, "t": t, "tod": t % 288 / 12, "lgb": iou(pl >= th, y, e), "nn": iou(pn >= th, y, e),
                 "mix": iou(pm >= th, y, e), "qT": int(qT.sum()), "qT_up5": int(qT[:5].sum()),
                 "y30": int((y[5] & e[5]).sum()), "y30_up5": int((y[5, :5] & e[5, :5]).sum())})
    for name, p in (("lgb", pl), ("nn", pn), ("mix", pm)):
        pr = p >= th
        for h in range(6):
            for zone, sl in (("up5", slice(0, 5)), ("rest", slice(5, L))):
                ee = e[h, sl]
                cells.append({"panel": panel, "model": name, "h": h + 1, "zone": zone,
                              "fp": int((pr[h, sl] & ~y[h, sl] & ee).sum()), "fn": int((~pr[h, sl] & y[h, sl] & ee).sum()),
                              "tp": int((pr[h, sl] & y[h, sl] & ee).sum())})
d = pd.DataFrame(rows); c = pd.DataFrame(cells)
pd.set_option("display.width", 200)
print(d.groupby("panel")[["lgb", "nn", "mix", "qT", "qT_up5", "y30_up5"]].mean().round(3))
print("\nerror cells by zone (sum over windows, horizons):")
print(c.groupby(["panel", "zone", "model"])[["tp", "fp", "fn"]].sum().unstack("model").loc[lambda x: x.index.get_level_values(0).str.startswith("D7_I210")])
print("\nD7_I210_E upstream-end errors by horizon:")
print(c[(c.panel == "D7_I210_E") & (c.zone == "up5")].groupby(["h", "model"])[["fp", "fn"]].sum().unstack("model"))
w = d[d.panel == "D7_I210_E"].assign(dn=lambda x: x.nn - x.lgb).sort_values("dn")
print("\nD7_I210_E windows, network - LightGBM:"); print(w.round(3).to_string(index=False))
