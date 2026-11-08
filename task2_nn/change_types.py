"""What kind of ongoing cells does the network mix (w = 0.5) change against i5, out of fold vs release?
Cell classes: observed queue at T on that link (T row, empty T cells from T-5) and the train climatology of a queue
at T+h on that link (clim < 0.1 low, 0.1-0.5 mid, >= 0.5 high). Out of fold also shows how often the change is right.

Usage: change_types.py <dump.pkl from build_mix.py --dump> <i5 csv>   (network runs r1 r2 r3 for out of fold)
"""
import pickle
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task2")); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import features_v1 as F1
from mix_eval import combine
from mix_eval5 import H5_STEPS, load, thr_for
from combine_eval import BASES, load_nn

T0_DOW = 5  # 2030-06-01 is a Saturday
FK = ["A", "B", "C", "all"]
Z = {}


def panel(p):
    if p not in Z:
        z = np.load(HERE / "data" / f"{p}.npz")
        Z[p] = {"vcut": z["vcut"], "clim": {k: z[f"clim_{k}_q"] for k in FK}, "ev_t": z["ev_t"], "ev_speed": z["ev_speed"],
                "ev_fold": z["ev_fold"], "ev_cond": z["ev_cond"],
                **{f"rel_{s}_{a}": z[f"rel_{s}_{a}"] for s in ("validation", "private") for a in ("wid", "t", "speed")}}
    return Z[p]


def qT(sp, vcut):
    row = np.where(np.isnan(sp[12]), sp[11], sp[12])
    return np.nan_to_num(row, nan=999) <= vcut


def clim_at(z, key, t):
    tod = t % 288; dw = int((t // 288 + T0_DOW) % 7 >= 5)
    return np.stack([z["clim"][FK[key]][dw, (tod + h) % 288] for h in range(1, 7)])    # [6, L]


def classes(q_t, cl):
    c = np.where(cl < 0.1, "low", np.where(cl < 0.5, "mid", "high"))
    return np.char.add(np.where(q_t[None, :], "qT|", "free|"), c)


def main() -> None:
    rows = []
    # out of fold
    G, kt = load(); NN = load_nn([HERE / "runs" / r for r in ("r1", "r2", "r3")])
    for i, k in enumerate(G.kg):
        z = panel(k[0]); j = int(np.nonzero((z["ev_t"] == k[1]) & (z["ev_cond"] == 0))[0][0])
        r = G.go[k]; pl = combine({v: r[f"{v}_0"] for v in BASES["i5"]}, BASES["i5"]); th = thr_for(int(kt[i]), H5_STEPS)
        a = pl >= th; b = 0.5 * pl + 0.5 * NN[k]["nn_0"] >= th; e = r["e"].astype(bool); y = r["y"].astype(bool)
        cls = classes(qT(z["ev_speed"][j], z["vcut"]), clim_at(z, int(z["ev_fold"][j]), k[1]))
        ch = (a != b) & e
        for c, d, ok in zip(cls[ch], np.where(b[ch], "0->1", "1->0"), b[ch] == y[ch]):
            rows.append(("oof", c, d, ok))
    # release
    D = pickle.load(open(sys.argv[1], "rb"))
    i5 = pd.read_csv(sys.argv[2]).set_index(["window_id"])
    for w, v in D.items():
        p = w[3:].rsplit("_", 2)[0]; s = "validation" if "_validation_" in w else "private"; z = panel(p)
        j = int(np.nonzero(z[f"rel_{s}_wid"] == w)[0][0]); t = int(z[f"rel_{s}_t"][j])
        got = i5.loc[[w]]; npos = int(got.queue_pred.sum())
        th = min((0.31, 0.37), key=lambda x: abs(int((v["p"] >= x).sum()) - npos))
        a = v["p"] >= th; b = 0.5 * v["p"] + 0.5 * v["q"] >= th
        cl = clim_at(z, 3, t)[v["h"] - 1, v["link"]]; qt = qT(z[f"rel_{s}_speed"][j], z["vcut"])[v["link"]]
        cls = np.char.add(np.where(qt, "qT|", "free|"), np.where(cl < 0.1, "low", np.where(cl < 0.5, "mid", "high")))
        ch = a != b
        for c, d in zip(cls[ch], np.where(b[ch], "0->1", "1->0")):
            rows.append((s, c, d, np.nan))
    d = pd.DataFrame(rows, columns=["split", "cls", "dir", "ok"])
    n = d.groupby(["cls", "dir", "split"]).size().unstack("split").fillna(0).astype(int)
    for s in ("oof", "validation", "private"):
        n[f"{s}%"] = (n[s] / n[s].sum() * 100).round(1)
    n["oof_right%"] = (d[d.split == "oof"].groupby(["cls", "dir"]).ok.mean() * 100).round(0)
    pd.set_option("display.width", 200)
    print(n[["oof", "oof%", "oof_right%", "validation", "validation%", "private", "private%"]].to_string())


if __name__ == "__main__":
    main()
