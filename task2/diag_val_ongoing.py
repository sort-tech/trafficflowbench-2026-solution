"""Diagnostic: ongoing windows (release inputs only) - history queue sizes vs predicted sizes."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from build import get_models, predict_mean, ONGOING_THR
from build_dataset import keep_mask
from common import CACHE, H, PANELS, Panel
from features import Clim, Static, horizon_rows, link_features, release_windows

split = sys.argv[1] if len(sys.argv) > 1 else "validation"
models = get_models(False)
W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
rows = []
for panel in PANELS:
    st = Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
    onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
    clim = Clim(P, qc, sorted(W.month.unique()), onset_t)
    for w, arr, t_idx in release_windows(panel, split, st):
        if w.condition != "queue_ongoing":
            continue
        ratio = arr["speed"] / st.vcut
        qe = (ratio <= 1) & (arr["pct"] >= 75)
        qv = ratio <= 1
        df = link_features(arr, st, clim, t_idx)
        d = horizon_rows(df, clim, t_idx)
        p = predict_mean(models["ongoing"], d)
        p[keep_mask(d) == 0] = 0
        pred = np.zeros((H, st.L), bool)
        pred[d.h.to_numpy() - 1, d.link.to_numpy()] = p >= ONGOING_THR
        filled_T = df.r_0.to_numpy() <= 1
        rows.append({"panel": panel, "T": w.forecast_origin[5:16], "hist_vis_q": list(qv[:-1].sum(1)), "T_q": int(qv[-1].sum()),
                     "T_filled_q": int(filled_T.sum()), "pred": list(pred.sum(1)),
                     "pred_minus_pers": int((pred & ~filled_T).sum()), "pers_minus_pred": int((~pred & filled_T).sum()),
                     "nanT": round(float(np.isnan(arr['speed'][-1]).mean()), 2)})
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 80)
print(df.to_string())
