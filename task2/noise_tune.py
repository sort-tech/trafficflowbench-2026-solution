"""Decision rules under the LB mixture assumption: a share of eligible cells carries labels unrelated
to the inputs (Bernoulli at the window's own positive rate). Out-of-fold full-model predictions (greedy CV)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE
from model import best_k_set, iou


def corrupt(y, e, share, rng):
    y = y.astype(bool).copy()
    idx = np.where(e.astype(bool))[0]
    if len(idx) == 0:
        return y
    pi = y[idx].mean()
    pick = idx[rng.random(len(idx)) < share]
    y[pick] = rng.random(len(pick)) < pi
    return y


def run(path, kind, rules, shares=(0.0, 0.2, 0.4), reps=3):
    d = pd.read_parquet(CACHE / path)
    groups = list(d.groupby(["panel", "t"], sort=False))
    out = {}
    for share in shares:
        res = {r: [] for r in rules}
        for rep in range(reps if share else 1):
            rng = np.random.default_rng(1000 + rep)
            for (panel, t), g in groups:
                g = g.sort_values(["h", "link"]) if "h" in g else g.sort_values("link")
                y = corrupt(g.y.to_numpy(), g.elig.to_numpy(), share, rng)
                e = g.elig.to_numpy()
                p = g.p.to_numpy()
                for name, f in rules.items():
                    res[name].append((panel, iou(f(p), y, e)))
        out[share] = {n: pd.DataFrame(v, columns=["panel", "v"]).groupby("panel").v.mean().mean() for n, v in res.items()}
    return pd.DataFrame(out)


if __name__ == "__main__":
    on_rules = {"k": best_k_set, "k_g0.8": lambda p: best_k_set(p ** 0.8), "k_g1.25": lambda p: best_k_set(p ** 1.25),
                "top1": lambda p: np.eye(len(p), dtype=bool)[np.argmax(p)], "thr0.5": lambda p: p >= 0.5, "thr0.3": lambda p: p >= 0.3}
    print("onset (cvpred_queue_onset, full)"); print(run("cvpred_queue_onset.parquet", "onset", on_rules).round(4))
    go_rules = {f"thr{t}": (lambda p, t=t: p >= t) for t in (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)}
    print("ongoing (cvpred_queue_ongoing_bag, full)"); print(run("cvpred_queue_ongoing_bag.parquet", "ongoing", go_rules).round(4))
