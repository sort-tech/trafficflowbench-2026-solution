"""11/1 update grid around h5: four-way ongoing mixes, horizon-dependent weights, stepped small-queue thresholds.
Cached out-of-fold probabilities only. dS0 = no-shift dS_q vs h5; shift = mean over +-45/+-60 min vs h5 at the same shift.
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
G5 = {"v1": .2, "phys": .2, "aug": .6}


def load():
    G = Grid()
    th = pickle.load(open(CACHE / "v1_thr_oof.pkl", "rb"))["on"]
    c4 = pickle.load(open(CACHE / "cand_eval4.pkl", "rb"))["go"]
    for k in G.on:
        G.on[k].update(th[k])
    for k in G.go:
        G.go[k].update(c4[k])
    pp = {(r["panel"], r["t"]): r for r in pickle.load(open(CACHE / "pp_windows.pkl", "rb"))}
    kt = np.array([int(pp[k]["q_t5"].sum()) for k in G.kg])
    return G, kt


def thr_for(k_t: int, steps) -> float:
    """steps: [(k_max, thr), ...] ascending; windows with no queue at T (k_t = 0) use the last (default) threshold."""
    if k_t >= 1:
        for kmax, t in steps:
            if k_t <= kmax:
                return t
    return steps[-1][1]


H5_STEPS = [(5, 0.31), (10**9, 0.37)]


def main() -> None:
    G, kt = load()

    def onset(sh):
        out = []
        for k in G.ko:
            r = G.on[k]
            pr = np.zeros(r["y"].shape, bool); pr[H - 1] = best_k_set(0.5 * r[f"v1_{sh}"] + 0.5 * r[f"v1_thr_{sh}"])
            out.append(iou(pr, r["y"], r["e"]))
        return np.array(out)

    def ongoing(wts, sh, steps=H5_STEPS):
        """wts: one dict for all horizons, or a list of 6 dicts (T+5..T+30)."""
        out = []
        for i, k in enumerate(G.kg):
            r = G.go[k]
            if isinstance(wts, dict):
                p = combine({v: r[f"{v}_{sh}"] for v in wts}, wts)
            else:
                p = np.vstack([combine({v: r[f"{v}_{sh}"][h:h + 1] for v in wts[h]}, wts[h]) for h in range(H)])
            out.append(iou(p >= thr_for(int(kt[i]), steps), r["y"], r["e"]))
        return np.array(out)

    on = {s: onset(s) for s in (0,) + SH}
    ref_go = {s: ongoing(G5, s) for s in (0,) + SH}
    ref = {s: G.S(on[s], ref_go[s]) for s in (0,) + SH}
    rows = []

    def add(name, go_fn):
        a = {s: G.S(on[s], go_fn(s)) for s in (0,) + SH}
        d = go_fn(0) - ref_go[0]
        rows.append({"setting": name, "dS0": a[0] - ref[0], "se": d.std() / np.sqrt(len(d)), "better": int((d > 0).sum()),
                     "worse": int((d < 0).sum()), "shift": np.mean([a[s] - ref[s] for s in SH])})

    for name, w in [("1) v1 .15 phys .15 aug .35 onphys .35", {"v1": .15, "phys": .15, "aug": .35, "onphys": .35}),
                    ("1) v1 .2 phys .2 aug .3 onphys .3", {"v1": .2, "phys": .2, "aug": .3, "onphys": .3}),
                    ("1) v1 .15 phys .1 aug .4 onphys .35", {"v1": .15, "phys": .1, "aug": .4, "onphys": .35})]:
        add(name, lambda s, w=w: ongoing(w, s))
    hz = [{"v1": .10, "phys": .30, "aug": .20, "onphys": .40}] * 2 + [{"v1": .15, "phys": .20, "aug": .35, "onphys": .30}] * 2 \
        + [{"v1": .20, "phys": .15, "aug": .45, "onphys": .20}] * 2
    add("2) horizon weights (guide example)", lambda s: ongoing(hz, s))
    hz2 = [{"v1": .2, "phys": .3, "aug": .3, "onphys": .2}] * 2 + [{"v1": .2, "phys": .2, "aug": .4, "onphys": .2}] * 2 \
        + [{"v1": .2, "phys": .1, "aug": .5, "onphys": .2}] * 2
    add("2) horizon weights (phys .3 -> .1, aug .3 -> .5)", lambda s: ongoing(hz2, s))
    for steps in ([(3, 0.29), (8, 0.33), (15, 0.35), (10**9, 0.37)], [(3, 0.30), (8, 0.32), (15, 0.35), (10**9, 0.37)],
                  [(3, 0.28), (8, 0.34), (15, 0.35), (10**9, 0.38)], [(3, 0.30), (8, 0.33), (15, 0.36), (10**9, 0.37)]):
        add("3) stepped " + " ".join(f"<={k if k < 10**8 else 'inf'}:{t}" for k, t in steps), lambda s, st=steps: ongoing(G5, s, st))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"h5 local S_q {ref[0]:.4f}")
    print(res.round(4).to_string(index=False))
    res.to_csv(CACHE / "mix_eval5.csv", index=False)


if __name__ == "__main__":
    main()
