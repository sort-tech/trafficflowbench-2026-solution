"""Cache train mainline observations as dense [time, link] arrays per panel.

Links are ordered by the released LWR topology order_index (upstream -> downstream).
Output: cache/<PANEL>_train.npz with speed, flow, pct (uint8), times (int64 ns), link_ids, vcut.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

KAGGLE = Path(__file__).resolve().parent.parent
REL = Path(os.environ.get("TFB_REL", KAGGLE / "data" / "kaggle_public"))
CACHE = Path(__file__).resolve().parent / "cache"
PANELS = ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]


def link_order(panel: str) -> pd.DataFrame:
    net = REL / "corridors" / panel / "network"
    topo = pd.read_csv(net / "lwr_mainline_topology.csv")
    links = pd.read_csv(net / "links.csv")
    df = links.merge(topo[["link_id", "order_index"]], on="link_id", how="left")
    df = df.sort_values("order_index").reset_index(drop=True)
    df["vcut"] = 0.6 * df.free_speed_kmh.astype(float)
    return df


def load_split(panel: str, split: str, layer: str) -> pd.DataFrame:
    root = REL / "corridors" / panel / split / layer
    cols = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed"]
    t = ds.dataset(str(root), format="parquet", partitioning="hive").to_table(columns=cols)
    return t.to_pandas()


def build(panel: str) -> None:
    order = link_order(panel)
    lid = order.link_id.astype(str).to_numpy()
    pos = {k: i for i, k in enumerate(lid)}
    t0 = pd.Timestamp("2030-06-01T00:00:00Z")
    times = pd.date_range(t0, pd.Timestamp("2031-02-28T23:55:00Z"), freq="5min")
    T, L = len(times), len(lid)
    speed = np.full((T, L), np.nan, np.float32)
    flow = np.full((T, L), np.nan, np.float32)
    occ = np.full((T, L), np.nan, np.float32)
    pct = np.zeros((T, L), np.uint8)
    n = 0
    cols = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed"]
    for month in sorted((REL / "corridors" / panel / "train" / "mainline_states").glob("year_month=*")):
        d = ds.dataset(str(month), format="parquet").to_table(columns=cols).to_pandas()
        ts = pd.to_datetime(d.timestamp, utc=True)
        ti = ((ts - t0) // pd.Timedelta("5min")).to_numpy()
        li = d.link_id.astype(str).map(pos).to_numpy()
        speed[ti, li] = d.speed_kmh.to_numpy(np.float32)
        flow[ti, li] = d.flow_vph.to_numpy(np.float32)
        occ[ti, li] = d.occupancy.to_numpy(np.float32)
        pct[ti, li] = d.pct_observed.to_numpy().astype(np.uint8)
        n += len(d)
        del d, ts
    np.savez_compressed(
        CACHE / f"{panel}_train.npz", speed=speed, flow=flow, occ=occ, pct=pct,
        times=times.asi8, link_ids=lid, vcut=order.vcut.to_numpy(np.float32),
        lanes=order.lanes.to_numpy(np.float32), length=order.length_km.to_numpy(np.float32),
        capacity=order.capacity_vph.to_numpy(np.float32), free=order.free_speed_kmh.to_numpy(np.float32),
    )
    print(panel, T, L, f"rows={n:,}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append")
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    for p in args.panel or PANELS:
        build(p)


if __name__ == "__main__":
    main()
