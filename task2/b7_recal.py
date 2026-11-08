"""B7 step 1: re-fit decision settings on the previous month, apply to the next (cached out-of-fold probabilities).

For each consecutive train month pair (m, m+1), the settings that maximise S_q on month m's official-style windows
are applied to month m+1 and compared with the fixed i5 settings (and, with the network, the fixed go50 mix).
Grid: onset v1_thr weight; ongoing mix (i5 / g5 / v1-phys .10 / aug-onphys .30), threshold, small-queue threshold
(and the network weight for the network base). Network out-of-fold probabilities: task2_nn/runs/r1..r3 (mean).
"""
from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd

from common import H, aggregate
from mix_eval import combine
from mix_eval5 import load
from model import best_k_set, iou

MIXES = {"i5": {"v1": .15, "phys": .15, "aug": .35, "onphys": .35}, "g5": {"v1": .2, "phys": .2, "aug": .6},
         "pao1010": {"v1": .1, "phys": .1, "aug": .4, "onphys": .4}, "pao33": {"v1": .2, "phys": .2, "aug": .3, "onphys": .3}}
W_THR = (0.3, 0.4, 0.5, 0.6, 0.7)
THRS = (0.33, 0.35, 0.37, 0.39, 0.41)
SMALLS = (0.28, 0.31, 0.34, None)
NN_W = (0.3, 0.5, 0.7)


def load_nn():
    runs = []
    for d in ("r1", "r2", "r3"):
        r = {}
        for k in "ABC":
            r.update(pickle.load(open(HERE.parent / "task2_nn" / "runs" / d / f"oof_{k}.pkl", "rb")))
        runs.append(r)
    return {key: {f: np.mean([r[key][f] for r in runs], 0) for f in runs[0][key]} for key in runs[0]}


def main() -> None:
    G, kt = load()
    NN = load_nn()
    om = np.array([G.on[k]["month"] for k in G.ko]); gm = np.array([G.go[k]["month"] for k in G.kg])
    months = sorted(set(om))
    meta = G.meta

    on_cache = {w: np.array([iou(_on(G.on[k], w), G.on[k]["y"], G.on[k]["e"]) for k in G.ko]) for w in W_THR}

    def go_scores(mix, thr, small, nn_w=None):
        out = []
        for i, k in enumerate(G.kg):
            r = G.go[k]
            p = combine({v: r[f"{v}_0"] for v in MIXES[mix]}, MIXES[mix])
            if nn_w is not None:
                p = (1 - nn_w) * p + nn_w * NN[k]["nn_0"]
            t = small if (small is not None and 1 <= kt[i] <= 5) else thr
            out.append(iou(p >= t, r["y"], r["e"]))
        return np.array(out)

    go_cache = {}
    for mix, thr, small in itertools.product(MIXES, THRS, SMALLS):
        go_cache[(mix, thr, small, None)] = go_scores(mix, thr, small)
    for thr, small, w in itertools.product(THRS, SMALLS, NN_W):
        go_cache[("i5", thr, small, w)] = go_scores("i5", thr, small, w)

    def S_month(m, on_s, go_s):
        mo, mg = om == m, gm == m
        df = pd.concat([meta.iloc[:len(G.ko)][mo].assign(iou=on_s[mo]), meta.iloc[len(G.ko):][mg].assign(iou=go_s[mg])])
        return aggregate(df, "iou")["S_queue"]

    rows = []
    for base_name, keys in (("i5", [k for k in go_cache if k[3] is None]), ("i5+nn", [k for k in go_cache if k[3] is not None])):
        fixed_go = ("i5", 0.37, 0.31, None if base_name == "i5" else 0.5)
        for m, m1 in zip(months[:-1], months[1:]):
            best_w = max(W_THR, key=lambda w: S_month(m, on_cache[w], go_cache[fixed_go]))
            best_go = max(keys, key=lambda g: S_month(m, on_cache[0.5], go_cache[g]))
            fixed = S_month(m1, on_cache[0.5], go_cache[fixed_go])
            adapted = S_month(m1, on_cache[best_w], go_cache[best_go])
            rows.append({"base": base_name, "fit_month": m, "test_month": m1, "fixed": fixed, "adapted": adapted,
                         "gain": adapted - fixed, "w_thr": best_w, "ongoing": best_go})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))
    for b, g in res.groupby("base"):
        print(f"{b}: mean gain {g.gain.mean():+.4f} (sd {g.gain.std():.4f}, se {g.gain.std() / np.sqrt(len(g)):.4f}), "
              f"up {int((g.gain > 1e-9).sum())} / down {int((g.gain < -1e-9).sum())} of {len(g)}")
    res.to_csv(HERE / "cache" / "b7_recal.csv", index=False)


def _on(r, w):
    pr = np.zeros(r["y"].shape, bool)
    pr[H - 1] = best_k_set((1 - w) * r["v1_0"] + w * r["v1_thr_0"])
    return pr


if __name__ == "__main__":
    main()
