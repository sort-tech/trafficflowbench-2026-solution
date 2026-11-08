"""Tune ensemble weights and decision rules on saved out-of-fold probabilities (cache/ens_*.parquet).

Scores every combination at each time-of-day shift; shift-invariant members (onphys/online, phys/notod)
are the same at every shift. Reports the score at shift 0 and the mean/min over shifted scenarios.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE
from model import best_k_set, iou

SHIFTS = [0, 6, -6, 12, -12, 18, -18]


def panel_mean(rows: list[dict], col: str) -> float:
    df = pd.DataFrame(rows)
    return float(df.groupby("panel")[col].mean().mean())


def onset_scores(weights, gammas=(1.0,)):
    d = pd.read_parquet(CACHE / "ens_onset.parquet")
    out = []
    groups = list(d.groupby(["panel", "t"], sort=False))
    for w in weights:
        for other in ("onphys", "online"):
            for g_ in gammas:
                res = {}
                for s in SHIFTS:
                    rows = []
                    for (panel, t), g in groups:
                        p = w * g[f"p_aug_{s}"].to_numpy() + (1 - w) * g[f"p_{other}"].to_numpy()
                        rows.append({"panel": panel, "v": iou(best_k_set(p ** g_), g.y.to_numpy(), g.elig.to_numpy())})
                    res[s] = panel_mean(rows, "v")
                out.append({"cond": "onset", "w_aug": w, "other": other, "gamma": g_, **{f"s{s}": v for s, v in res.items()}})
    return pd.DataFrame(out)


def ongoing_scores(weights, thrs):
    d = pd.read_parquet(CACHE / "ens_ongoing.parquet")
    out = []
    groups = list(d.groupby(["panel", "t"], sort=False))
    for w in weights:
        for other in ("phys", "notod"):
            for thr in thrs:
                res = {}
                for s in SHIFTS:
                    rows = []
                    for (panel, t), g in groups:
                        p = w * g[f"p_aug_{s}"].to_numpy() + (1 - w) * g[f"p_{other}"].to_numpy()
                        rows.append({"panel": panel, "v": iou(p >= thr, g.y.to_numpy(), g.elig.to_numpy())})
                    res[s] = panel_mean(rows, "v")
                out.append({"cond": "ongoing", "w_aug": w, "other": other, "thr": thr, **{f"s{s}": v for s, v in res.items()}})
    return pd.DataFrame(out)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    sh = [f"s{s}" for s in SHIFTS if s != 0]
    df = df.copy()
    df["shift_mean"] = df[sh].mean(axis=1)
    df["shift_min"] = df[sh].min(axis=1)
    df["robust"] = (df["s0"] + 2 * df["shift_mean"]) / 3
    return df.sort_values("robust", ascending=False)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    a = summarize(onset_scores([0, 0.25, 0.5, 0.75, 1.0]))
    a.to_csv(CACHE / "ens_tune_onset.csv", index=False)
    print(a.round(4).head(12).to_string(index=False))
    b = summarize(ongoing_scores([0, 0.25, 0.5, 0.75, 1.0], [0.3, 0.35, 0.4, 0.45]))
    b.to_csv(CACHE / "ens_tune_ongoing.csv", index=False)
    print(b.round(4).head(12).to_string(index=False))
