"""Cache validation/private masked mainline observations as dense [time, link] arrays per panel.

Used only for online (same-split, strictly pre-T) features: for a window at T, feature code
slices these arrays at indices < T. Output: cache/<PANEL>_<split>.npz with speed, pct (uint8).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from common import CACHE, PANELS, REL
from features import Static

SPLIT_START = {"validation": pd.Timestamp("2031-03-01T00:00:00Z"), "private": pd.Timestamp("2031-04-01T00:00:00Z")}
SPLIT_DAYS = {"validation": 31, "private": 30}


def build(panel: str, split: str) -> None:
    st = Static(panel)
    pos = {k: i for i, k in enumerate(st.link_ids)}
    t0 = SPLIT_START[split]
    n_t = SPLIT_DAYS[split] * 288
    speed = np.full((n_t, st.L), np.nan, np.float32)
    pct = np.zeros((n_t, st.L), np.uint8)
    root = REL / "corridors" / panel / split / "mainline_states_masked"
    t = ds.dataset(str(root), format="parquet", partitioning="hive").to_table(
        columns=["timestamp", "link_id", "speed_kmh", "pct_observed"]).to_pandas()
    ti = ((pd.to_datetime(t.timestamp, utc=True) - t0) // pd.Timedelta("5min")).to_numpy()
    li = t.link_id.astype(str).map(pos).to_numpy()
    ok = (ti >= 0) & (ti < n_t)
    speed[ti[ok], li[ok]] = t.speed_kmh.to_numpy(np.float32)[ok]
    pct[ti[ok], li[ok]] = t.pct_observed.to_numpy()[ok].astype(np.uint8)
    np.savez_compressed(CACHE / f"{panel}_{split}.npz", speed=speed, pct=pct)
    print(panel, split, n_t, st.L, f"rows={int(ok.sum()):,}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append")
    args = ap.parse_args()
    for p in args.panel or PANELS:
        for s in ("validation", "private"):
            build(p, s)


if __name__ == "__main__":
    main()
