"""Bottleneck threshold onset (v1_thr) on cached out-of-fold probabilities; ongoing = g5.

Onset rules: v1 | v1_thr | mean (0.5 / 0.5 probabilities, top-k) | rank (mean of within-window ranks, k of v1)
| norm (mean of p / max p, k of v1); each also applied to D12_I5 panels only (D7 keeps v1).
dS0 = no-shift dS_q vs g5, shift = mean over +-45/+-60 min of (rule - g5) at the same shift. Per-family S_q as well.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE, FAMILY, H
from mix_eval import Grid
from model import best_k_set, iou

SH = (9, -9, 12, -12)
REF_GO = ({"v1": .2, "phys": .2, "aug": .6}, 0.37)


def choose(p1: np.ndarray, p2: np.ndarray, kind: str) -> np.ndarray:
    if kind == "v1":
        return best_k_set(p1)
    if kind == "thr":
        return best_k_set(p2)
    if kind == "mean":
        return best_k_set(0.5 * p1 + 0.5 * p2)
    k = int(best_k_set(p1).sum())
    if kind == "rank":
        r1 = np.argsort(np.argsort(-p1, kind="stable"), kind="stable")
        r2 = np.argsort(np.argsort(-p2, kind="stable"), kind="stable")
        sc = -(r1 + r2) / 2.0 + 1e-6 * p1
    else:   # norm
        sc = 0.5 * p1 / max(p1.max(), 1e-9) + 0.5 * p2 / max(p2.max(), 1e-9)
    s = np.zeros(len(p1), bool); s[np.argsort(-sc, kind="stable")[:k]] = True
    return s


def main() -> None:
    G = Grid()
    th = pickle.load(open(CACHE / "v1_thr_oof.pkl", "rb"))["on"]
    for k in G.on:
        G.on[k].update(th[k])
    go = {sh: G.ongoing(*REF_GO, sh=sh) for sh in (0,) + SH}

    def onset(kind, sh, i5_only=False):
        out = []
        for key in G.ko:
            r = G.on[key]
            kd = kind if (not i5_only or key[0].startswith("D12_I5")) else "v1"
            pr = np.zeros(r["y"].shape, bool); pr[H - 1] = choose(r[f"v1_{sh}"], r[f"v1_thr_{sh}"], kd)
            out.append(iou(pr, r["y"], r["e"]))
        return np.array(out)

    base = {sh: onset("v1", sh) for sh in (0,) + SH}
    ref = {sh: G.S(base[sh], go[sh]) for sh in (0,) + SH}
    fam_of = np.array([FAMILY[k[0]] for k in G.ko])
    rows, frows = [], []
    for kind in ("thr", "mean", "rank", "norm"):
        for i5 in (False, True):
            name = f"{kind}{' (D12_I5 only)' if i5 else ''}"
            a = {sh: onset(kind, sh, i5) for sh in (0,) + SH}
            d = a[0] - base[0]
            rows.append({"onset": name, "dS0": G.S(a[0], go[0]) - ref[0], "onset_win": d.mean(), "se": d.std() / np.sqrt(len(d)),
                         "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
                         "shift": np.mean([G.S(a[s], go[s]) - ref[s] for s in SH])})
            if not i5:
                for f in sorted(set(fam_of)):
                    m = fam_of == f
                    frows.append({"onset": name, "family": f, "n": int(m.sum()), "v1_onset_iou": base[0][m].mean(),
                                  "rule_onset_iou": a[0][m].mean(), "diff": d[m].mean(),
                                  "shift_diff": np.mean([(a[s][m] - base[s][m]).mean() for s in SH])})
    pd.set_option("display.width", 220)
    r = pd.DataFrame(rows); f = pd.DataFrame(frows)
    print(r.round(4).to_string(index=False)); print(f.round(4).to_string(index=False))
    r.to_csv(CACHE / "mix_eval3.csv", index=False); f.to_csv(CACHE / "mix_eval3_family.csv", index=False)


if __name__ == "__main__":
    main()
