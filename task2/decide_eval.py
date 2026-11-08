"""Evaluate decision rules on saved CV predictions (no refit)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE
from model import best_k_set, iou


def evaluate(path: Path, rules: dict, col: str = "p") -> pd.Series:
    d = pd.read_parquet(path)
    res = {k: [] for k in rules}
    panels = []
    for (panel, t), g in d.groupby(["panel", "t"], sort=False):
        p = g[col].to_numpy()
        y = g.y.to_numpy()
        e = g.elig.to_numpy()
        for name, f in rules.items():
            res[name].append(iou(f(p, g), y, e))
        panels.append(panel)
    df = pd.DataFrame(res)
    df["panel"] = panels
    return df.groupby("panel").mean().mean().sort_values(ascending=False)


if __name__ == "__main__":
    which = sys.argv[1]
    rules = {f"thr{t:.2f}": (lambda p, g, t=t: p >= t) for t in (0.3, 0.35, 0.4, 0.45, 0.5)}
    for gam in (0.5, 0.7, 0.85, 1.0):
        rules[f"k_g{gam}"] = lambda p, g, gam=gam: best_k_set(p ** gam)
    col = sys.argv[2] if len(sys.argv) > 2 else "p"
    print(evaluate(CACHE / f"cvpred_{which}.parquet", rules, col).round(4).to_string())
