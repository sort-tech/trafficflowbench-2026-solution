"""11/1 grid around h5 (onset v1 .5 + v1_thr .5; ongoing v1 .2 + phys .2 + aug .6, threshold 0.37, small-queue
windows (K_T 1..5 links queued at T, empty T cells from T-5) at 0.31). Cached out-of-fold probabilities only.
dS0 = no-shift dS_q vs h5; shift = mean over +-45/+-60 min of (setting - h5) at the same shift.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE, H
from mix_eval import Grid, combine
from model import best_k_set, iou

SH = (9, -9, 12, -12)


def main() -> None:
    G = Grid()
    th = pickle.load(open(CACHE / "v1_thr_oof.pkl", "rb"))["on"]
    c4 = pickle.load(open(CACHE / "cand_eval4.pkl", "rb"))["go"]
    for k in G.on:
        G.on[k].update(th[k])
    for k in G.go:
        G.go[k].update(c4[k])
    pp = {(r["panel"], r["t"]): r for r in pickle.load(open(CACHE / "pp_windows.pkl", "rb"))}
    kt = np.array([int(pp[k]["q_t5"].sum()) for k in G.kg])

    def onset(w_thr, sh):
        out = []
        for k in G.ko:
            r = G.on[k]
            p = (1 - w_thr) * r[f"v1_{sh}"] + w_thr * r[f"v1_thr_{sh}"]
            pr = np.zeros(r["y"].shape, bool); pr[H - 1] = best_k_set(p)
            out.append(iou(pr, r["y"], r["e"]))
        return np.array(out)

    def ongoing(wts, sh, thr=0.37, small=0.31, kmax=5):
        out = []
        for i, k in enumerate(G.kg):
            r = G.go[k]
            p = combine({v: r[f"{v}_{sh}"] for v in wts}, wts)
            t = small if (small is not None and 1 <= kt[i] <= kmax) else thr
            out.append(iou(p >= t, r["y"], r["e"]))
        return np.array(out)

    G5 = {"v1": .2, "phys": .2, "aug": .6}
    ref_on = {s: onset(0.5, s) for s in (0,) + SH}
    ref_go = {s: ongoing(G5, s) for s in (0,) + SH}
    ref = {s: G.S(ref_on[s], ref_go[s]) for s in (0,) + SH}
    rows = []

    def add(name, on_fn, go_fn):
        a = {s: G.S(on_fn(s), go_fn(s)) for s in (0,) + SH}
        d = np.concatenate([on_fn(0), go_fn(0)]) - np.concatenate([ref_on[0], ref_go[0]])
        rows.append({"setting": name, "dS0": a[0] - ref[0], "se": d.std() / np.sqrt(len(d)), "better": int((d > 0).sum()),
                     "worse": int((d < 0).sum()), "shift": np.mean([a[s] - ref[s] for s in SH])})

    for w in (0.4, 0.6, 0.7):
        add(f"(a) onset v1 {1 - w:.1f} + v1_thr {w:.1f}", lambda s, w=w: onset(w, s), lambda s: ref_go[s])
    for name, wts in [("(b) v1 .2 phys .2 aug .3 onphys .3", {"v1": .2, "phys": .2, "aug": .3, "onphys": .3}),
                      ("(b) v1 .2 phys .2 aug .4 onphys .2", {"v1": .2, "phys": .2, "aug": .4, "onphys": .2}),
                      ("(b) v1 .2 phys .1 aug .35 onphys .35", {"v1": .2, "phys": .1, "aug": .35, "onphys": .35})]:
        add(name, lambda s: ref_on[s], lambda s, w=wts: ongoing(w, s))
    for small, kmax in ((0.28, 5), (0.34, 5), (0.31, 3), (0.31, 8), (0.28, 3), (0.34, 8)):
        add(f"(c) small-queue thr {small}, K_T 1-{kmax}", lambda s: ref_on[s], lambda s, a=small, b=kmax: ongoing(G5, s, small=a, kmax=b))
    print("h5 local S_q %.4f; small-queue windows K_T 1-3/1-5/1-8: %d/%d/%d of %d" % (ref[0], ((kt >= 1) & (kt <= 3)).sum(),
          ((kt >= 1) & (kt <= 5)).sum(), ((kt >= 1) & (kt <= 8)).sum(), len(kt)))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    res.to_csv(CACHE / "mix_eval4.csv", index=False)


if __name__ == "__main__":
    main()
