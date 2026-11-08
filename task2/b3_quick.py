"""B3 quick check: does the selector regime ("T-5 met every ongoing rule but the persistence cap") carry information
the i5 ongoing probabilities miss? Logistic recalibration logit(p') = a[reg, h] + b[reg, h] * logit(p) fitted on two
folds' windows and applied to the third (out of fold), same decision rule; compared with the same recalibration
without the regime split (to separate generic recalibration from the regime effect).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import features_v1 as F1
from common import CACHE, H, NHIST, PANELS, SPACING, STEP, T0, Panel, select_windows
from mix_eval import combine
from mix_eval5 import load
from model import best_k_set, iou

W5 = {"v1": .15, "phys": .15, "aug": .35, "onphys": .35}


def regimes() -> dict:
    out = {}
    for panel in PANELS:
        P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        cs = np.concatenate([[0], np.cumsum((P.q & P.elig).any(1))])
        for mon in [m for ms in F1.FOLDS.values() for m in ms]:
            start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
            end = int((pd.Timestamp(f"{mon}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            prev = None
            for w in select_windows(P, qc, max_per_condition=5, start=max(start, 12), end=end).sort_values("t").itertuples():
                t = int(w.t); s = t - 1
                if w.condition == "queue_ongoing":
                    out[(panel, t)] = int((prev is None or s - prev >= SPACING) and cs[s] - cs[s - NHIST] >= 2 and s - NHIST >= start)
                prev = t
    return out


def main() -> None:
    G, kt = load()
    reg = regimes()
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    rows = []
    for i, k in enumerate(G.kg):
        r = G.go[k]
        p = combine({v: r[f"{v}_0"] for v in W5}, W5)
        rows.append((k, m2f[r["month"]], reg[k], p, r["y"] & r["e"], r["e"], kt[i]))

    def lg(p):
        p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

    def run(use_reg: bool):
        out = np.zeros(len(rows))
        for f in "ABC":
            tr = [x for x in rows if x[1] != f]
            models = {}
            for g in ((0, 1) if use_reg else (None,)):
                for h in range(H):
                    X, Y = [], []
                    for _, _, rg, p, y, e, _ in tr:
                        if g is not None and rg != g:
                            continue
                        m = e[h] & (p[h] > 0.02)
                        X.append(lg(p[h][m])); Y.append(y[h][m])
                    X = np.concatenate(X)[:, None]; Y = np.concatenate(Y)
                    models[(g, h)] = LogisticRegression(C=10.0).fit(X, Y)
            for j, (key, fo, rg, p, y, e, k_t) in enumerate(rows):
                if fo != f:
                    continue
                q = p.copy()
                for h in range(H):
                    m = p[h] > 0.02
                    if m.any():
                        q[h][m] = models[(rg if use_reg else None, h)].predict_proba(lg(p[h][m])[:, None])[:, 1]
                out[j] = iou(q >= (0.31 if 1 <= k_t <= 5 else 0.37), G.go[key]["y"], G.go[key]["e"])
        return out

    base = np.array([iou(x[3] >= (0.31 if 1 <= x[6] <= 5 else 0.37), G.go[x[0]]["y"], G.go[x[0]]["e"]) for x in rows])
    on = []
    for k in G.ko:
        r = G.on[k]; pr = np.zeros(r["y"].shape, bool); pr[H - 1] = best_k_set(.5 * r["v1_0"] + .5 * r["v1_thr_0"]); on.append(iou(pr, r["y"], r["e"]))
    on = np.array(on)
    S0 = G.S(on, base)
    rg = np.array([x[2] for x in rows]).astype(bool)
    print(f"ongoing windows {len(rows)}, T-5 regime {rg.sum()}; i5 S_q {S0:.4f}; mean ongoing IoU regime {base[rg].mean():.3f} / other {base[~rg].mean():.3f}")
    for name, use in (("recalibration by horizon", False), ("recalibration by horizon x regime", True)):
        a = run(use); d = a - base
        print(f"{name}: S_q {G.S(on, a):.4f} ({G.S(on, a) - S0:+.4f}); windows better {int((d > 0).sum())} worse {int((d < 0).sum())}; "
              f"regime windows {d[rg].mean():+.4f}, others {d[~rg].mean():+.4f}")


if __name__ == "__main__":
    main()
