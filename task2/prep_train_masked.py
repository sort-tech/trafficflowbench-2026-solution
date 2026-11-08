"""Cache the train masked view (Task 1 mask applied) as [time, link] speed/pct arrays (diagnostics)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from common import CACHE, PANELS, REL, STEP, T0
from features import Static

for panel in PANELS:
    st = Static(panel)
    pos = {k: i for i, k in enumerate(st.link_ids)}
    n_t = 273 * 288
    speed = np.full((n_t, st.L), np.nan, np.float32)
    pct = np.zeros((n_t, st.L), np.uint8)
    root = REL / "corridors" / panel / "train" / "mainline_states_masked"
    for reg in sorted(root.glob("mask_regime=*")):
        t = ds.dataset(str(reg), format="parquet").to_table(columns=["timestamp", "link_id", "speed_kmh", "pct_observed"]).to_pandas()
        ti = ((pd.to_datetime(t.timestamp, utc=True) - T0) // STEP).to_numpy()
        li = t.link_id.astype(str).map(pos).to_numpy()
        speed[ti, li] = t.speed_kmh.to_numpy(np.float32)
        pct[ti, li] = t.pct_observed.to_numpy().astype(np.uint8)
    np.savez_compressed(CACHE / f"{panel}_train_masked.npz", speed=speed, pct=pct)
    print(panel, flush=True)
