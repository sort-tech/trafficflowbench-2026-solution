"""Diagnostic: onset predictions on release windows (inputs only) with model confidence."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from build import get_models, predict_mean
from common import CACHE, H, PANELS, Panel
from features import Clim, Static, dow_of, link_features, release_windows
from model import best_k_set

models = get_models(False)
W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
rows = []
for panel in PANELS:
    st = Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
    onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
    clim = Clim(P, qc, sorted(W.month.unique()), onset_t)
    for split in ("validation", "private"):
        for w, arr, t_idx in release_windows(panel, split, st):
            if w.condition != "queue_onset":
                continue
            df = link_features(arr, st, clim, t_idx)
            df["h"] = np.int8(H); df["clim_h"] = df["clim_6"]; df["clim_dh"] = df["clim_6"] - df["clim_0"]
            df["tail_ext30"] = df["tail_rel"]; df["tail_ext15"] = df["tail_rel"]; df["dd_ext"] = df["dd_0"]
            dt = int(dow_of(t_idx) >= 5)
            df["p11_h"] = clim.p11[H - 1, dt, t_idx % 288]; df["p01_h"] = clim.p01[H - 1, dt, t_idx % 288]; df["tr_h"] = df["p01_h"]
            p = predict_mean(models["onset"], df)
            sel = best_k_set(p)
            lo = np.argsort(df.r0_dev.to_numpy())[:4]
            rows.append({"panel": panel, "split": split, "T": w.forecast_origin[5:16], "pmax": round(p.max(), 2),
                         "psum": round(p.sum(), 2), "pred": tuple(np.where(sel)[0]),
                         "onset_rate": round(float(clim.onset_rate[dt, t_idx % 288]) * 288, 2),
                         "low_rdev_links": tuple(lo), "low_rdev": tuple(np.round(df.r0_dev.to_numpy()[lo], 2))})
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 60)
print(df.to_string())
df.to_csv(CACHE / "diag_val_onset.csv", index=False)
