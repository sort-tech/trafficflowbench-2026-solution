"""10-30 instruction, Step 2 and Step 3 grids on cached out-of-fold probabilities (reference g5: onset v1,
ongoing v1 .2 + phys .2 + aug .6, threshold 0.37).

Columns: dS0 = no-shift dS_q vs g5; shift = mean over +-45/+-60 min of (setting - g5) at the same shift
(onset and ongoing shifted together). TTA settings are scored without shift only (their shifted versions need
lookups beyond +-60 min).
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
REF = ({"v1": .2, "phys": .2, "aug": .6}, 0.37)


def main() -> None:
    G = Grid()
    c4 = pickle.load(open(CACHE / "cand_eval4.pkl", "rb"))["go"]
    for k in G.go:
        G.go[k].update(c4[k])
    jit = pickle.load(open(CACHE / "v1_jit_oof.pkl", "rb"))["on"]
    for k in G.on:
        G.on[k].update(jit[k])
    pp = {(r["panel"], r["t"]): r for r in pickle.load(open(CACHE / "pp_windows.pkl", "rb"))}
    kt = np.array([int(pp[k]["q_t5"].sum()) for k in G.kg])

    def go_scores(wts, thr, sh=0, tta=None, small_thr=None):
        out = []
        for i, k in enumerate(G.kg):
            r = G.go[k]
            if tta:
                parts = {}
                for v in wts:
                    if v == "phys":
                        parts[v] = r[f"phys_{sh}"]
                    else:
                        parts[v] = 0.25 * r[f"{v}_{-tta}"] + 0.5 * r[f"{v}_0"] + 0.25 * r[f"{v}_{tta}"]
            else:
                parts = {v: r[f"{v}_{sh}"] for v in wts}
            p = combine(parts, wts)
            t = small_thr if (small_thr is not None and 1 <= kt[i] <= 5) else thr
            out.append(iou(p >= t, r["y"], r["e"]))
        return np.array(out)

    def on_scores(kind, sh=0):
        out = []
        for k in G.ko:
            r = G.on[k]
            p1 = r[f"v1_{sh}"]
            if kind == "v1":
                s = best_k_set(p1)
            elif kind in ("v1_jit", "v1_phys_jit"):
                s = best_k_set(r[f"{kind}_{sh}"])
            elif kind.startswith("mean_"):
                s = best_k_set(0.5 * p1 + 0.5 * r[f"{kind[5:]}_{sh}"])
            elif kind.startswith("norm_"):           # rank by mean of p/max p, keep v1's k
                p2 = r[f"{kind[5:]}_{sh}"]
                sc = 0.5 * p1 / max(p1.max(), 1e-9) + 0.5 * p2 / max(p2.max(), 1e-9)
                kk = int(best_k_set(p1).sum())
                s = np.zeros(len(p1), bool); s[np.argsort(-sc, kind="stable")[:kk]] = True
            pr = np.zeros(r["y"].shape, bool); pr[H - 1] = s
            out.append(iou(pr, r["y"], r["e"]))
        return np.array(out)

    base_on = {sh: on_scores("v1", sh) for sh in (0,) + SH}
    ref_go = {sh: go_scores(*REF, sh=sh) for sh in (0,) + SH}
    ref = {sh: G.S(base_on[sh], ref_go[sh]) for sh in (0,) + SH}
    rows = []

    def add(name, on_fn, go_fn, shifts=True):
        a0 = G.S(on_fn(0), go_fn(0))
        d = np.concatenate([on_fn(0), go_fn(0)]) - np.concatenate([base_on[0], ref_go[0]])
        r = {"setting": name, "dS0": a0 - ref[0], "se": d.std() / np.sqrt(len(d)), "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
        if shifts:
            r["shift"] = np.mean([G.S(on_fn(s), go_fn(s)) - ref[s] for s in SH])
        rows.append(r)

    # Step 2-1 ratio grid
    mixes = [("v1.2 ph.2 aug.6 (g5)", {"v1": .2, "phys": .2, "aug": .6}), ("v1.10 ph.25 aug.65", {"v1": .1, "phys": .25, "aug": .65}),
             ("v1.10 ph.20 aug.70", {"v1": .1, "phys": .2, "aug": .7}), ("v1.05 ph.25 aug.70", {"v1": .05, "phys": .25, "aug": .7}),
             ("v1 0 ph.30 aug.70", {"phys": .3, "aug": .7}), ("v1.15 ph.20 aug.50 onphys.15", {"v1": .15, "phys": .2, "aug": .5, "onphys": .15}),
             ("v1.2 ph.2 onphys.6", {"v1": .2, "phys": .2, "onphys": .6})]
    for name, w in mixes:
        for t in (0.34, 0.35, 0.36, 0.37, 0.38):
            add(f"2-1 {name} thr{t}", lambda s: base_on[s], lambda s, w=w, t=t: go_scores(w, t, s))
    # Step 2-2 TTA (no shift)
    for d in (6, 9):
        for t in (0.35, 0.37, 0.39):
            add(f"2-2 g5 TTA +-{d * 5}min thr{t}", lambda s: base_on[s], lambda s, d=d, t=t: go_scores(REF[0], t, s, tta=d), shifts=False)
    # Step 2-4 small queues (1..5 links queued at T, empty T cells from T-5): threshold - 0.03
    for st in (0.34, 0.31):
        add(f"2-4 g5, K_T 1-5 thr{st}", lambda s: base_on[s], lambda s, st=st: go_scores(*REF, s, small_thr=st))
    # Step 3 onset (ongoing = g5)
    for kind in ("v1_jit", "v1_phys_jit", "mean_v1_jit", "mean_v1_phys_jit", "norm_v1_jit", "norm_v1_phys_jit"):
        add(f"3 onset {kind}", lambda s, k=kind: on_scores(k, s), lambda s: ref_go[s])
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print(f"K_T 1-5 ongoing windows: {int(((kt >= 1) & (kt <= 5)).sum())} of {len(kt)}")
    print(res.round(4).to_string(index=False))
    res.to_csv(CACHE / "mix_eval2.csv", index=False)


if __name__ == "__main__":
    main()
