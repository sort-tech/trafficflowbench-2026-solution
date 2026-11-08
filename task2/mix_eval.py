"""Mix grids on cached out-of-fold probabilities (cand_eval.py, cand_eval2.py, cand_eval3.py; no model calls).

Reference = lgb_v1_ongphys03 (onset v1, ongoing 0.3 v1 + 0.7 phys, threshold 0.37). For every setting:
dS (no shift) vs the reference, and the mean over +-45/+-60 min time-of-day shifts of dS vs the reference
at the same shift (positive = more robust than the reference). Both onset and ongoing are shifted together.

Usage: mix_eval.py onset | ongoing
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE, H, aggregate
from model import best_k_set, iou

SHIFTS = (0, 9, -9, 12, -12)


def load():
    c2 = pickle.load(open(CACHE / "cand_eval2.pkl", "rb"))
    c3 = pickle.load(open(CACHE / "cand_eval3.pkl", "rb"))
    go1 = {(r["panel"], r["t"]): r for r in pickle.load(open(CACHE / "cand_eval_ongoing.pkl", "rb"))}
    on = {}
    for key, r in c2["on"].items():
        on[key] = {"y": r["y"], "e": r["e"], "month": r["month"],
                   **{f"{v}_{s}": r[f"{v}_{s}"] for v in ("v1", "onphys", "phys") for s in SHIFTS},
                   **{f"aug_{s}": c3["on"][key][f"aug_{s}"] for s in SHIFTS}}
    go = {}
    for key, r in go1.items():
        go[key] = {"y": r["y"], "e": r["e"], "month": r["month"],
                   **{f"v1_{s}": r[f"v1_{s}"] for s in SHIFTS}, **{f"phys_{s}": r[f"ph_{s}"] for s in SHIFTS},
                   **{f"notod_{s}": c2["go"][key][f"notod_{s}"] for s in SHIFTS},
                   **{f"aug_{s}": c3["go"][key][f"aug_{s}"] for s in SHIFTS}}
    sp = CACHE / "seeds_v1.pkl"
    if sp.exists():                      # v1 averaged over 3 seeds (seeds_v1.py)
        sd = pickle.load(open(sp, "rb"))
        for key in on:
            on[key].update({f"v1b_{s}": sd["on"][key][f"v1b_{s}"] for s in SHIFTS})
        for key in go:
            go[key].update({f"v1b_{s}": sd["go"][key][f"v1b_{s}"] for s in SHIFTS})
    return on, go


def combine(parts: dict, wts: dict, mode: str = "arith", eps: float = 0.02) -> np.ndarray:
    tot = sum(wts.values())
    if mode == "arith":
        return sum(w * parts[v] for v, w in wts.items()) / tot
    if mode == "geo":
        g = np.exp(sum(w * np.log(parts[v].astype(np.float64) + eps) for v, w in wts.items()) / tot) - eps
        return np.clip(g, 0, 1)
    if mode == "logit":
        z = sum(w * np.log(np.clip(parts[v], 1e-4, 1 - 1e-4) / (1 - np.clip(parts[v], 1e-4, 1 - 1e-4)))
                for v, w in wts.items()) / tot
        return 1 / (1 + np.exp(-z))
    raise ValueError(mode)


class Grid:
    def __init__(self):
        self.on, self.go = load()
        self.ko, self.kg = list(self.on), list(self.go)
        self.meta = pd.DataFrame([{"panel": p, "t": t, "month": self.on[(p, t)]["month"], "condition": "queue_onset"} for p, t in self.ko]
                                 + [{"panel": p, "t": t, "month": self.go[(p, t)]["month"], "condition": "queue_ongoing"} for p, t in self.kg])
        self.cache = {}

    def onset(self, wts, gamma=1.0, sh=0):
        key = ("on", tuple(sorted(wts.items())), gamma, sh)
        if key not in self.cache:
            out = []
            for k in self.ko:
                r = self.on[k]
                p = combine({v: r[f"{v}_{sh}"] for v in wts}, wts) ** gamma
                pr = np.zeros(r["y"].shape, bool); pr[H - 1] = best_k_set(p)
                out.append(iou(pr, r["y"], r["e"]))
            self.cache[key] = np.array(out)
        return self.cache[key]

    def ongoing(self, wts, thr, mode="arith", eps=0.02, sh=0):
        key = ("go", tuple(sorted(wts.items())), thr, mode, eps, sh)
        if key not in self.cache:
            self.cache[key] = np.array([iou(combine({v: self.go[k][f"{v}_{sh}"] for v in wts}, wts, mode, eps) >= thr,
                                            self.go[k]["y"], self.go[k]["e"]) for k in self.kg])
        return self.cache[key]

    def S(self, a, b):
        return aggregate(self.meta.assign(iou=np.concatenate([a, b])), "iou")["S_queue"]


REF_GO = ({"v1": 0.3, "phys": 0.7}, 0.37)


def row(G, name, on_fn, go_fn):
    ref = {sh: G.S(G.onset({"v1": 1.0}, 1.0, sh), G.ongoing(*REF_GO, sh=sh)) for sh in SHIFTS}
    val = {sh: G.S(on_fn(sh), go_fn(sh)) for sh in SHIFTS}
    d0 = np.concatenate([on_fn(0), go_fn(0)]) - np.concatenate([G.onset({"v1": 1.0}), G.ongoing(*REF_GO)])
    r = {"setting": name, "dS0": val[0] - ref[0], "win": d0.mean(), "se": d0.std() / np.sqrt(len(d0)),
         "shift_mean_vs_ref": np.mean([val[s] - ref[s] for s in SHIFTS[1:]])}
    for s in SHIFTS[1:]:
        r[f"{s * 5:+d}"] = val[s] - ref[0]
    r["pass"] = (r["dS0"] >= -0.003) and (r["shift_mean_vs_ref"] > 0)
    return r


def onset_grid(G):
    rows = []
    mixes = [("v1", {"v1": 1.0}), ("v1.7 aug.3", {"v1": .7, "aug": .3}), ("v1.6 aug.4", {"v1": .6, "aug": .4}),
             ("v1.5 aug.5", {"v1": .5, "aug": .5}), ("v1.5 aug.3 onphys.2", {"v1": .5, "aug": .3, "onphys": .2}),
             ("v1.5 onphys.5", {"v1": .5, "onphys": .5}), ("aug", {"aug": 1.0})]
    for name, wts in mixes:
        for g in (1.0, 0.85, 0.75, 0.65):
            rows.append(row(G, f"{name} g{g}", lambda sh, w=wts, g=g: G.onset(w, g, sh), lambda sh: G.ongoing(*REF_GO, sh=sh)))
    return pd.DataFrame(rows)


def ongoing_grid(G):
    rows = []
    thrs = (0.27, 0.30, 0.33, 0.35, 0.37, 0.40, 0.43)
    settings = [("arith v1.3 phys.7", {"v1": .3, "phys": .7}, "arith", 0), ("arith v1.2 phys.8", {"v1": .2, "phys": .8}, "arith", 0),
                ("arith v1.2 phys.6 aug.2", {"v1": .2, "phys": .6, "aug": .2}, "arith", 0),
                ("arith v1.15 phys.7 aug.15", {"v1": .15, "phys": .7, "aug": .15}, "arith", 0)]
    for w in (0.3, 0.4, 0.5):
        settings.append((f"logit v1{w}", {"v1": w, "phys": 1 - w}, "logit", 0))
        for eps in (0.01, 0.02, 0.03):
            settings.append((f"geo eps{eps} v1{w}", {"v1": w, "phys": 1 - w}, "geo", eps))
    for name, wts, mode, eps in settings:
        best = max(thrs, key=lambda t: G.S(G.onset({"v1": 1.0}), G.ongoing(wts, t, mode, eps or 0.02)))
        r = row(G, f"{name} thr{best}", lambda sh: G.onset({"v1": 1.0}, 1.0, sh),
                lambda sh, w=wts, t=best, m=mode, e=eps: G.ongoing(w, t, m, e or 0.02, sh))
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    G = Grid()
    pd.set_option("display.width", 250)
    res = onset_grid(G) if sys.argv[1] == "onset" else ongoing_grid(G)
    print(res.round(4).to_string(index=False))
    res.to_csv(CACHE / f"mix_eval_{sys.argv[1]}.csv", index=False)
