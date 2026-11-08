"""(2) Is there selector structure in ongoing windows (like onset's 'queue only at T+30')?

(a) Greedy train ongoing windows (cache/pseudo_windows.parquet, greedy): would the origin T-5 have met every ongoing
    condition except the persistence cap (spacing from the previous selected origin, >= 2 eligible history steps,
    queue in the horizon)? And how often was its persistence IoU > 0.9?
(b) Per horizon h, IoU of the eligible observed queue at T+h with the clean state at T and at T-5.
(c) Out of fold (official-style monthly windows, lgb models as cached): replace the first k horizons of the current
    ongoing decision (h5 / i5 mixes) by persistence of the queue at T (empty T cells from T-5).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE, H, NHIST, PANELS, SPACING, Panel
from mix_eval5 import load, thr_for, H5_STEPS
from mix_eval import combine
from model import best_k_set, iou


def iou_cells(a, b, e):
    a = a & e; b = b & e
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


def part_ab() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    rows = []
    for panel in PANELS:
        P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        g = W[(W.panel == panel) & W.greedy].sort_values("t")
        qv = P.q & P.elig
        anyq = qv.any(1)
        cs = np.concatenate([[0], np.cumsum(anyq)])
        prev_t = None
        for w in g.itertuples(index=False):
            t = int(w.t)
            if w.condition == "queue_ongoing":
                s = t - 1
                spacing_ok = prev_t is None or s - prev_t >= SPACING
                nsteps = cs[s] - cs[s - NHIST]
                hor = qc[s + 1:s + 1 + H]
                pers = qc[s]
                u = (hor | pers).sum(); piou = (hor & pers).sum() / u if u else 1.0
                r = {"panel": panel, "t": t, "spacing_ok": spacing_ok, "hist_ok": nsteps >= 2, "hor_ok": bool(hor.any()),
                     "piou_prev": piou}
                y = P.q[t + 1:t + 1 + H] & P.elig[t + 1:t + 1 + H]
                e = P.elig[t + 1:t + 1 + H]
                for h in range(H):
                    r[f"iouT_h{h + 1}"] = iou_cells(qc[t], y[h], e[h])
                    r[f"iouT5_h{h + 1}"] = iou_cells(qc[t - 1], y[h], e[h])
                rows.append(r)
            prev_t = t
    d = pd.DataFrame(rows)
    other = d.spacing_ok & d.hist_ok & d.hor_ok
    print(f"(a) greedy ongoing windows {len(d)}: T-5 met spacing+history+horizon {other.mean():.3f}; "
          f"of those, persistence IoU at T-5 > 0.9: {(d.piou_prev[other] > 0.9).mean():.3f}; spacing was the binding rule: {(~d.spacing_ok).mean():.3f}")
    print("(b) mean IoU of the eligible queue at T+h with the clean state at T / at T-5:")
    print("   ", " ".join(f"h{h + 1}: {d[f'iouT_h{h + 1}'].mean():.3f}/{d[f'iouT5_h{h + 1}'].mean():.3f}" for h in range(H)))
    sub = d[other & (d.piou_prev > 0.9)]
    print("    windows whose T-5 failed only on persistence:", " ".join(f"h{h + 1}: {sub[f'iouT_h{h + 1}'].mean():.3f}" for h in range(H)))
    d.to_csv(CACHE / "selector_diag_ab.csv", index=False)


def part_c() -> None:
    G, kt = load()
    pp = {(r["panel"], r["t"]): r for r in pickle.load(open(CACHE / "pp_windows.pkl", "rb"))}

    def onset():
        out = []
        for k in G.ko:
            r = G.on[k]; pr = np.zeros(r["y"].shape, bool); pr[H - 1] = best_k_set(.5 * r["v1_0"] + .5 * r["v1_thr_0"])
            out.append(iou(pr, r["y"], r["e"]))
        return np.array(out)

    on = onset()
    for name, w in (("h5", {"v1": .2, "phys": .2, "aug": .6}), ("i5", {"v1": .15, "phys": .15, "aug": .35, "onphys": .35})):
        base = None
        for kpers in (0, 1, 2, 3, 5):
            out = []
            for i, k in enumerate(G.kg):
                r = G.go[k]
                p = combine({v: r[f"{v}_0"] for v in w}, w) >= thr_for(int(kt[i]), H5_STEPS)
                if kpers:
                    p = p.copy(); p[:kpers] = pp[k]["q_t5"][None, :]
                out.append(iou(p, r["y"], r["e"]))
            out = np.array(out)
            S = G.S(on, out)
            if base is None:
                base = S
            print(f"(c) {name}: persistence for h1..h{kpers} + model for the rest: S_q {S:.4f} ({S - base:+.4f})")


if __name__ == "__main__":
    part_ab()
    part_c()
