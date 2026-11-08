"""Diagnostic: run the release input path on the official train windows and score with observed labels.

Compares with the pseudo-window input path on the same windows (same models, same climatology).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from build import get_models, predict_mean, ONGOING_THR
from build_dataset import keep_mask
from common import CACHE, H, PANELS, Panel, aggregate
from features import Clim, Static, dow_of, horizon_rows, link_features, release_windows, train_window_arrays
from model import best_k_set, iou


def predict_cells(df, cond, clim, t_idx, models):
    if cond == "queue_onset":
        df = df.copy()
        df["h"] = np.int8(H)
        df["clim_h"] = df["clim_6"]; df["clim_dh"] = df["clim_6"] - df["clim_0"]
        df["tail_ext30"] = df["tail_rel"]; df["tail_ext15"] = df["tail_rel"]; df["dd_ext"] = df["dd_0"]
        dt = int(dow_of(t_idx) >= 5)
        df["p11_h"] = clim.p11[H - 1, dt, t_idx % 288]; df["p01_h"] = clim.p01[H - 1, dt, t_idx % 288]; df["tr_h"] = df["p01_h"]
        p = predict_mean(models["onset"], df)
        pred = np.zeros((H, len(df)), bool)
        pred[H - 1] = best_k_set(p)
        return pred
    d = horizon_rows(df, clim, t_idx)
    p = predict_mean(models["ongoing"], d)
    p[keep_mask(d) == 0] = 0
    pred = np.zeros((H, len(df)), bool)
    pred[d.h.to_numpy() - 1, d.link.to_numpy()] = p >= ONGOING_THR
    return pred


def main() -> None:
    models = get_models(False)
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    rows = []
    for panel in PANELS:
        st = Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        clim = Clim(P, qc, sorted(W.month.unique()), onset_t)
        for w, arr, t_idx in release_windows(panel, "train", st):
            y = P.q[t_idx + 1:t_idx + 1 + H]; e = P.elig[t_idx + 1:t_idx + 1 + H]
            pr_rel = predict_cells(link_features(arr, st, clim, t_idx), w.condition, clim, t_idx, models)
            rng = np.random.default_rng([PANELS.index(panel), int(t_idx)])
            arr2 = train_window_arrays(P, t_idx, rng, 0.0)
            pr_ps = predict_cells(link_features(arr2, st, clim, t_idx), w.condition, clim, t_idx, models)
            # compare inputs
            same_hist = np.allclose(np.nan_to_num(arr["speed"][:-1], nan=-1), np.nan_to_num(arr2["speed"][:-1], nan=-1))
            nan_T = np.isnan(arr["speed"][-1]).mean()
            pers = np.broadcast_to(np.nan_to_num(arr["speed"][-1], nan=1e9) <= st.vcut, (H, st.L))
            rows.append({"panel": panel, "condition": w.condition, "t": t_idx, "iou_release": iou(pr_rel, y, e),
                         "iou_pseudo": iou(pr_ps, y, e), "iou_persT": iou(pers, y, e), "same_hist": same_hist,
                         "nanT": round(nan_T, 3), "diff_cells": int((pr_rel != pr_ps).sum())})
        print(panel, flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.groupby(["condition"])[["iou_release", "iou_pseudo", "iou_persT", "diff_cells", "nanT"]].mean().round(3))
    print("same history:", df.same_hist.all())
    for c in ["iou_release", "iou_pseudo", "iou_persT"]:
        print(c, round(aggregate(df, c)["S_queue"], 4))
    df.to_csv(CACHE / "diag_release_train.csv", index=False)


if __name__ == "__main__":
    main()
