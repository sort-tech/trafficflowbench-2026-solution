"""Diagnostic: input-only comparison of demand level (flow vs train climatology) in release windows vs train."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE, PANELS, Panel
from features import Clim, Static, dow_of, release_windows

W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
rows = []
for panel in PANELS:
    st = Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
    onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
    clim = Clim(P, qc, sorted(W.month.unique()), onset_t)
    def stats(arr, t_idx):
        dt = int(dow_of(t_idx) >= 5); tod = t_idx % 288
        f = arr["flow"] / st.cap
        free = (arr["speed"] / st.vcut) > 1.2
        out = []
        for i in range(arr["flow"].shape[0]):
            ti = (tod - (arr["flow"].shape[0] - 1 - i)) % 288
            dev = f[i] - clim.fclim[dt, ti]
            m = free[i] & ~np.isnan(dev)
            out.append(np.nanmean(dev[m]) if m.any() else np.nan)
        return float(np.nanmean(out))
    for split in ("train", "validation", "private"):
        for w, arr, t_idx in release_windows(panel, split, st):
            rows.append({"panel": panel, "split": split, "condition": w.condition, "fdev": stats(arr, t_idx)})
    # train pseudo greedy windows (same statistic) for reference
    g = W[(W.panel == panel) & W.greedy].sample(min(200, int(W[(W.panel == panel) & W.greedy].shape[0])), random_state=0)
    for w in g.itertuples():
        sl = slice(w.t - 12, w.t + 1)
        arr = {"flow": P.flow[sl], "speed": P.speed[sl]}
        rows.append({"panel": panel, "split": "train_pseudo", "condition": w.condition, "fdev": stats(arr, w.t)})
    print(panel, flush=True)
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.groupby(["panel", "split"]).fdev.mean().unstack().round(3))
print(df.groupby(["split", "condition"]).fdev.mean().unstack().round(3))
