"""Month and corridor breakdown of a network mix vs its LightGBM base (out-of-fold, no shift).
Usage: breakdown.py <base h5|i5> <w_onset> <w_ongoing> <run dir> [...]"""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task2")); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from common import H, aggregate, FAMILY
from mix_eval import combine
from mix_eval5 import H5_STEPS, load, thr_for
from model import best_k_set, iou
from combine_eval import BASES, load_nn

def main():
    bname, won, wgo = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    G, kt = load(); NN = load_nn([Path(a) for a in sys.argv[4:]]); base = BASES[bname]
    def scores(won, wgo):
        a = []
        for k in G.ko:
            r = G.on[k]; p = (1 - won) * (0.5 * r["v1_0"] + 0.5 * r["v1_thr_0"]) + won * NN[k]["nn_0"]
            pr = np.zeros(r["y"].shape, bool); pr[H - 1] = best_k_set(p); a.append(iou(pr, r["y"], r["e"]))
        for i, k in enumerate(G.kg):
            r = G.go[k]; p = (1 - wgo) * combine({v: r[f"{v}_0"] for v in base}, base) + wgo * NN[k]["nn_0"]
            a.append(iou(p >= thr_for(int(kt[i]), H5_STEPS), r["y"], r["e"]))
        return np.array(a)
    m = G.meta.assign(base=scores(0, 0), mix=scores(won, wgo))
    rows = []
    for mon, g in m.groupby("month"):
        rows.append({"month": mon, "base": aggregate(g, "base")["S_queue"], "mix": aggregate(g, "mix")["S_queue"]})
    r = pd.DataFrame(rows); r["d"] = r.mix - r.base
    print(r.round(4).to_string(index=False)); print("months improved", int((r.d > 0).sum()), "of", len(r))
    m["fam"] = m.panel.map(FAMILY)
    print(m.groupby(["fam", "condition"])[["base", "mix"]].mean().assign(d=lambda x: x.mix - x.base).round(4))
main()
