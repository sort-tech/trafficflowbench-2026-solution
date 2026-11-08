"""Diagnostic: how persistence (T-5) and our CV predictions score under alternative truth definitions."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE, H, PANELS, Panel, aggregate
from model import best_k_set, iou

on = pd.read_parquet(CACHE / "cvpred_queue_onset.parquet")
go = pd.read_parquet(CACHE / "cvpred_queue_ongoing_bag.parquet")
DEFS = ["base", "thr1.1", "thr0.9", "allcells", "shift_up", "shift_down", "time+1", "time-1", "dilate_up1"]
rows = []
for panel in PANELS:
    P = Panel(panel)
    qc = np.load(CACHE / f"{panel}_qclean.npy")
    ratio = P.speed / P.vcut
    rfill = pd.DataFrame(ratio).ffill().bfill().to_numpy()
    for cond, d in (("queue_onset", on), ("queue_ongoing", go)):
        dp = d[d.panel == panel]
        for t, g in dp.groupby("t"):
            pred = np.zeros((H, P.n_l), bool)
            if cond == "queue_onset":
                g = g.sort_values("link")
                pred[H - 1, g.link.to_numpy()] = best_k_set(g.p.to_numpy())
            else:
                pred[g.h.to_numpy() - 1, g.link.to_numpy()] = g.p.to_numpy() >= 0.42
            pers = np.broadcast_to((P.q & P.elig)[t - 1], (H, P.n_l))
            sl = slice(t + 1, t + 1 + H)
            el = P.elig[sl]
            r = {"panel": panel, "condition": cond, "t": t}
            for dn in DEFS:
                e = el
                if dn == "base":
                    y = P.q[sl]
                elif dn.startswith("thr"):
                    y = np.nan_to_num(ratio[sl], nan=9) <= float(dn[3:])
                elif dn == "allcells":
                    y = rfill[sl] <= 1; e = np.ones_like(el)
                elif dn == "shift_up":      # truth at link l = observation at link l+1
                    y = np.roll(P.q[sl], -1, axis=1)
                elif dn == "shift_down":
                    y = np.roll(P.q[sl], 1, axis=1)
                elif dn == "time+1":
                    y = P.q[t + 2:t + 2 + H]
                elif dn == "time-1":
                    y = P.q[t:t + H]
                elif dn == "dilate_up1":
                    y = P.q[sl] | np.roll(P.q[sl], -1, axis=1)
                r[f"ours|{dn}"] = iou(pred, y, e)
                if cond == "queue_ongoing":
                    r[f"pers|{dn}"] = iou(pers, y, e)
            rows.append(r)
    print(panel, flush=True)
df = pd.DataFrame(rows)
out = []
for dn in DEFS:
    a = df.groupby(["condition", "panel"])[f"ours|{dn}"].mean().groupby("condition").mean()
    b = df[df.condition == "queue_ongoing"].groupby("panel")[f"pers|{dn}"].mean().mean()
    out.append({"truth": dn, "ours_onset": a["queue_onset"], "ours_ongoing": a["queue_ongoing"], "pers_ongoing": b})
print(pd.DataFrame(out).round(3).to_string(index=False))
print("LB-implied: ours_onset 0.754, ours_ongoing 0.766, pers_ongoing ~0.603")


def iou_pred_unfiltered(pred: np.ndarray, y: np.ndarray, elig: np.ndarray) -> float:
    """Hypothesis: truth restricted to eligible cells, predictions counted on every cell."""
    t = y.astype(bool) & elig.astype(bool)
    p = pred.astype(bool)
    u = (p | t).sum()
    return 1.0 if u == 0 else float((p & t).sum() / u)
