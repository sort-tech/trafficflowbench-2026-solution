"""Out-of-fold check of climatology-gate variants on ongoing windows (i5 base, onset LightGBM only).
Gate cells G = links free at T (T row, empty cells from T-5) with train climatology of a queue at T+h >= c.
  add   (build_mix --gate-clim c): in G the mix may not add a queue against i5      -> final = mix & ~(G & ~i5)
  wide  (--gate-clim c --gate-wide): in G a queue needs both i5 and the network alone -> final = i5 & (p_nn >= th)
Release: cells changed vs i5 and vs cpu3g_go50 (add, w 0.5, c 0.5), from build_mix.py --dump probabilities.

Usage: gate_eval2.py <dump.pkl> <i5 csv> <run dir> [...]
"""
import pickle
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
from change_types import clim_at, panel, qT

SPECS = [(0.5, None, "add"), (0.5, 0.5, "add"), (0.5, 0.3, "add"), (0.5, 0.7, "add"), (0.5, 0.0, "add"),
         (0.5, 0.5, "wide"), (0.5, 0.3, "wide"), (0.5, 0.7, "wide"), (0.3, 0.5, "add"), (0.7, 0.5, "add"),
         (0.7, 0.3, "add"), (0.3, 0.3, "add"), (0.0, 0.5, "wide")]


def decide(pl, pn, th, free, cl, w, c, mode):
    a = pl >= th; b = (1 - w) * pl + w * pn >= th
    if c is None:
        return b
    g = free[None, :] & (cl >= c) if free.ndim == 1 else free & (cl >= c)
    return np.where(g, a & (pn >= th), b) if mode == "wide" else b & ~(g & ~a)


def main() -> None:
    G, kt = load(); NN = load_nn([Path(a) for a in sys.argv[3:]]); base = BASES["i5"]
    prep = []
    for i, k in enumerate(G.kg):
        z = panel(k[0]); j = int(np.nonzero((z["ev_t"] == k[1]) & (z["ev_cond"] == 0))[0][0])
        prep.append((~qT(z["ev_speed"][j], z["vcut"]), clim_at(z, int(z["ev_fold"][j]), k[1]), thr_for(int(kt[i]), H5_STEPS)))

    def go(spec, sh):
        out = []
        for i, k in enumerate(G.kg):
            r = G.go[k]; free, cl, th = prep[i]
            pl = combine({v: r[f"{v}_{sh}"] for v in base}, base)
            out.append(iou(decide(pl, NN[k][f"nn_{sh}"], th, free, cl, *spec), r["y"], r["e"]))
        return np.array(out)

    def on(sh):
        out = []
        for k in G.ko:
            r = G.on[k]; pr = np.zeros(r["y"].shape, bool)
            pr[H - 1] = best_k_set(0.5 * r[f"v1_{sh}"] + 0.5 * r[f"v1_thr_{sh}"]); out.append(iou(pr, r["y"], r["e"]))
        return np.array(out)

    ON = {s: on(s) for s in (0,) + SH}
    S = lambda a, b: aggregate(G.meta.assign(iou=np.concatenate([a, b])), "iou")["S_queue"]
    ref = {s: go((0.0, None, "add"), s) for s in (0,) + SH}
    g50 = go((0.5, 0.5, "add"), 0)
    # release
    D = pickle.load(open(sys.argv[1], "rb")); i5 = pd.read_csv(sys.argv[2]).groupby("window_id").queue_pred.sum()
    R = []
    for wid, v in D.items():
        p = wid[3:].rsplit("_", 2)[0]; s = "validation" if "_validation_" in wid else "private"; z = panel(p)
        j = int(np.nonzero(z[f"rel_{s}_wid"] == wid)[0][0]); t = int(z[f"rel_{s}_t"][j])
        th = min((0.31, 0.37), key=lambda x: abs(int((v["p"] >= x).sum()) - int(i5[wid])))
        R.append((s, v["p"], v["q"], th, ~qT(z[f"rel_{s}_speed"][j], z["vcut"])[v["link"]], clim_at(z, 3, t)[v["h"] - 1, v["link"]]))
    rel_dec = lambda spec: [decide(p, q, th, free, cl, *spec) for s, p, q, th, free, cl in R]
    a5 = rel_dec((0.5, 0.5, "add")); a0 = [p >= th for s, p, q, th, free, cl in R]
    rows = []
    for spec in SPECS:
        g = {s: go(spec, s) for s in (0,) + SH}
        d = g[0] - g50
        dec = rel_dec(spec)
        cnt = lambda ref_, split: sum(int((x != y).sum()) for x, y, r in zip(dec, ref_, R) if r[0] == split)
        rows.append({"w": spec[0], "c": "-" if spec[1] is None else spec[1], "mode": spec[2] if spec[1] is not None else "none",
                     "dS_vs_i5": S(ON[0], g[0]) - S(ON[0], ref[0]), "dS_vs_g50": S(ON[0], g[0]) - S(ON[0], g50),
                     "se_vs_g50": d.std() / np.sqrt(len(d) + len(ON[0])),
                     "shift": np.mean([S(ON[s], g[s]) - S(ON[s], ref[s]) for s in SH]),
                     "val_vs_i5": cnt(a0, "validation"), "prv_vs_i5": cnt(a0, "private"),
                     "val_vs_g50": cnt(a5, "validation"), "prv_vs_g50": cnt(a5, "private")})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))
    res.to_csv(HERE / "logs" / "gate_eval2.csv", index=False)


if __name__ == "__main__":
    main()
