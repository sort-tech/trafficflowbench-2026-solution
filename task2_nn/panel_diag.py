"""Per panel ongoing IoU: LightGBM i5, network alone, mix .5 (no shift), plus window counts and queue sizes."""
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
rows = []
for i, k in enumerate(G.kg):
    r = G.go[k]; pl = combine({v: r[f"{v}_0"] for v in base}, base); pn = NN[k]["nn_0"]; th = thr_for(int(kt[i]), H5_STEPS)
    rows.append({"panel": k[0], "t": k[1], "lgb": iou(pl >= th, r["y"], r["e"]), "nn": iou(pn >= th, r["y"], r["e"]),
                 "mix": iou(0.5 * pl + 0.5 * pn >= th, r["y"], r["e"]), "npos": int((r["y"] & r["e"]).sum()),
                 "nn_sum": float(pn.sum()), "lgb_sum": float(pl.sum())})
d = pd.DataFrame(rows)
print(d.groupby("panel")[["lgb", "nn", "mix", "npos", "nn_sum", "lgb_sum"]].mean().round(3))
w = d[d.panel.str.startswith("D7_I210")].assign(dm=lambda x: x.mix - x.lgb).sort_values("dm")
print(w.head(12).round(3).to_string(index=False))
