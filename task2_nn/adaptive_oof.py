"""Out-of-fold analogue of the adaptive gate: for each of the 719 windows, h from the same train month's complete
days before the window's day (unmasked train observations, climatology table without the window's fold), the same
shrinkage, ratio and gate rule as adaptive_gate.py. Scores vs cpu3g_go50 (fixed c 0.5) and i5.

Usage: adaptive_oof.py <run dir> [...]
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task2")); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import features_v1 as F1
from common import H, T0, aggregate
from mix_eval import combine
from mix_eval5 import H5_STEPS, load, thr_for
from model import best_k_set, iou
from combine_eval import BASES, load_nn
from change_types import clim_at, panel as panel_z, qT
from gate_eval2 import decide
import adaptive_gate as AG

FK = ["A", "B", "C"]


def row_counts(p):
    z = np.load(HERE / "data" / f"{p}.npz"); sp = z["speed"]; vcut = z["vcut"]
    ti = np.arange(sp.shape[0])
    mon = pd.DatetimeIndex(T0 + ti * pd.Timedelta(minutes=5)).strftime("%Y-%m").to_numpy()
    m2f = {m: i for i, (f, ms) in enumerate(F1.FOLDS.items()) for m in ms}
    fold = np.array([m2f[m] for m in mon]); dw = ((ti // 288 + T0.day_of_week) % 7 >= 5).astype(int)
    vis = ~np.isnan(sp); q = np.nan_to_num(sp, nan=1e9) <= vcut
    n = np.zeros(len(ti)); qq = np.zeros(len(ti))
    for k in range(3):
        m = fold == k; hi = (z[f"clim_{FK[k]}_q"][dw[m], ti[m] % 288] >= 0.5) & vis[m]
        n[m] = hi.sum(1); qq[m] = (hi & q[m]).sum(1)
    first = {}
    for i, m in enumerate(mon):
        first.setdefault(m, i)
    return np.concatenate([[0], np.cumsum(n)]), np.concatenate([[0], np.cumsum(qq)]), mon, first


def main() -> None:
    G, kt = load(); NN = load_nn([Path(a) for a in sys.argv[1:]]); base = BASES["i5"]
    RC = {}
    rows = []
    for i, k in enumerate(G.kg):
        p, t = k
        if p not in RC:
            RC[p] = row_counts(p)
        cn, cq, mon, first = RC[p]
        a = first[mon[t]]; b = a + 288 * ((t - a) // 288)
        n, q = cn[b] - cn[a], cq[b] - cq[a]
        ref = AG.H_REF[p]; hs = (q + AG.K * ref) / (n + AG.K); c, wide = AG.gate_from_ratio(hs / ref)
        z = panel_z(p); j = int(np.nonzero((z["ev_t"] == t) & (z["ev_cond"] == 0))[0][0])
        free = ~qT(z["ev_speed"][j], z["vcut"]); cl = clim_at(z, int(z["ev_fold"][j]), t); th = thr_for(int(kt[i]), H5_STEPS)
        r = G.go[k]; pl = combine({v: r[f"{v}_0"] for v in base}, base); pn = NN[k]["nn_0"]
        out = {"panel": p, "ratio": hs / ref, "gate": "c0+wide" if wide else f"c{c}"}
        for name, w in (("ga50", 0.5), ("ga70", 0.7)):
            out[name] = iou(decide(pl, pn, th, free, cl, w, c, "wide" if wide else "add"), r["y"], r["e"])
        out["g50"] = iou(decide(pl, pn, th, free, cl, 0.5, 0.5, "add"), r["y"], r["e"])
        out["g70"] = iou(decide(pl, pn, th, free, cl, 0.7, 0.5, "add"), r["y"], r["e"])
        out["i5"] = iou(pl >= th, r["y"], r["e"])
        rows.append(out)
    d = pd.DataFrame(rows)
    on = []
    for k in G.ko:
        r = G.on[k]; pr = np.zeros(r["y"].shape, bool)
        pr[H - 1] = best_k_set(0.5 * r["v1_0"] + 0.5 * r["v1_thr_0"]); on.append(iou(pr, r["y"], r["e"]))
    S = lambda col: aggregate(G.meta.assign(iou=np.concatenate([on, d[col].to_numpy()])), "iou")["S_queue"]
    print("gate counts (719-window ongoing):", d.gate.value_counts().to_dict())
    print(d.groupby("panel").gate.value_counts().unstack(fill_value=0).to_string())
    for col in ("i5", "g50", "g70", "ga50", "ga70"):
        print(f"{col}: S_q {S(col):.4f}")
    for a, b in (("ga50", "g50"), ("ga70", "g70")):
        x = d[a] - d[b]
        print(f"{a} - {b}: dS_q {S(a) - S(b):+.4f}, window SE {x.std() / np.sqrt(len(x) + len(on)):.4f}, better {(x > 1e-9).sum()}, worse {(x < -1e-9).sum()}")


if __name__ == "__main__":
    main()
