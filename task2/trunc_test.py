"""Award-review style check: rerun predictions with every row timestamped after the window's origin T removed.

For each validation/private window, the inputs are rebuilt from the raw release files with timestamp > T dropped:
  mainline_states_masked  -> the T row and the online-feature array come from rows <= T only (no cache/*.npz)
  window_history          -> rows of that window with timestamp <= T
  window_index            -> only that window's own row is used
(ramp_states is not read by these candidates.) Train-only inputs (climatology, thresholds, models) are unchanged.
The predictions of each window are compared cell by cell with the submitted candidate file.

Usage: trunc_test.py <name> <candidate csv> <build.py options...>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pyarrow.dataset as pds

import build
import features as FC
import features_v1 as F1
from common import NHIST, PANELS, REL, STEP, T0

START = {"validation": pd.Timestamp("2031-03-01T00:00:00Z"), "private": pd.Timestamp("2031-04-01T00:00:00Z")}
COLS = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed"]
_raw: dict = {}
STATS = {"windows": 0, "rows_after_T_dropped": 0}


def raw(panel: str, split: str) -> pd.DataFrame:
    if (panel, split) not in _raw:
        t = pds.dataset(str(REL / "corridors" / panel / split / "mainline_states_masked"), format="parquet",
                        partitioning="hive").to_table(columns=COLS).to_pandas()
        t["ts"] = pd.to_datetime(t.timestamp, utc=True)
        _raw.clear()
        _raw[(panel, split)] = t
    return _raw[(panel, split)]


def truncated(panel: str, split: str, T: pd.Timestamp) -> pd.DataFrame:
    t = raw(panel, split)
    keep = t.ts <= T
    STATS["rows_after_T_dropped"] += int((~keep).sum())
    return t[keep]


def truncated_with_ti(panel: str, split: str, T: pd.Timestamp) -> pd.DataFrame:
    """Rows <= T of a split in the split_clim table format (validation rows are all before any private T)."""
    t = pds.dataset(str(REL / "corridors" / panel / split / "mainline_states_masked"), format="parquet",
                    partitioning="hive").to_table(columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed"]).to_pandas()
    ts = pd.to_datetime(t.timestamp, utc=True)
    t = t[ts <= T].copy()
    t["ti"] = ((pd.to_datetime(t.timestamp, utc=True) - T0) // STEP).astype(np.int64)
    return t


def release_windows_trunc(panel: str, split: str, st):
    base = REL / "task2" / panel / split
    wi = pd.read_csv(base / "window_index.csv")
    hist_all = pd.read_parquet(base / "window_history.parquet")
    pos = {k: i for i, k in enumerate(st.link_ids)}
    for w in wi.itertuples(index=False):
        T = pd.Timestamp(w.forecast_origin)
        times = [(T - (NHIST - i) * STEP).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(NHIST + 1)]
        h = hist_all[(hist_all.window_id == w.window_id) & (pd.to_datetime(hist_all.timestamp, utc=True) <= T)]
        arr = {k: np.full((NHIST + 1, st.L), np.nan, np.float32) for k in ("speed", "flow", "occ")}
        arr["pct"] = np.zeros((NHIST + 1, st.L), np.uint8)
        ti = h.timestamp.map({s: i for i, s in enumerate(times)}).to_numpy()
        li = h.link_id.astype(str).map(pos).to_numpy()
        arr["speed"][ti, li] = h.speed_kmh.to_numpy(); arr["flow"][ti, li] = h.flow_vph.to_numpy()
        arr["occ"][ti, li] = h.occupancy.to_numpy(); arr["pct"][ti, li] = h.pct_observed.to_numpy()
        tr = truncated(panel, split, T)
        r = tr[tr.ts == T]
        li = r.link_id.astype(str).map(pos).to_numpy()
        arr["speed"][NHIST, li] = r.speed_kmh.to_numpy(); arr["flow"][NHIST, li] = r.flow_vph.to_numpy()
        arr["occ"][NHIST, li] = r.occupancy.to_numpy(); arr["pct"][NHIST, li] = r.pct_observed.to_numpy()
        STATS["windows"] += 1
        import split_clim
        split_clim.CUR_T = T          # split-climatology reads must not see rows after T
        split_clim._tables.clear()
        split_clim._tables[(panel, split)] = truncated_with_ti(panel, split, T)
        if split == "private":
            split_clim._tables[(panel, "validation")] = truncated_with_ti(panel, "validation", T)
        yield w, arr, int((T - T0) / STEP)


def online_source_trunc(panel: str, split: str, st):
    off = int((START[split] - T0) / STEP)
    n = 31 * 288
    pos = {k: i for i, k in enumerate(st.link_ids)}

    def f(t_idx: int) -> dict:
        T = T0 + t_idx * STEP
        tr = truncated(panel, split, T)
        sp = np.full((n, st.L), np.nan, np.float32)
        ti = ((tr.ts - START[split]) // STEP).to_numpy()
        sp[ti, tr.link_id.astype(str).map(pos).to_numpy()] = tr.speed_kmh.to_numpy(np.float32)
        assert np.isnan(sp[t_idx - off + 1:]).all(), "no row after T may be present"
        return FC.online_features(sp, st.vcut, t_idx - off, 0)
    return f


def main() -> None:
    name, cand = sys.argv[1], sys.argv[2]
    F1.release_windows = release_windows_trunc
    build.online_source = online_source_trunc
    def no_cache(*a, **k):
        raise AssertionError("split-level cache read in the truncated run")
    build.split_rate_source = no_cache
    sys.argv = ["build.py", "--split", "all", "--onset-variant", "v1", "--ongoing-variant", "v1"] + sys.argv[3:] + \
               ["--out", f"/tmp/trunc_{name}.csv",
                "--overwrite"]
    build.main()
    a = pd.read_csv(cand)
    b = pd.read_csv(sys.argv[sys.argv.index("--out") + 1])
    m = a.merge(b, on=["window_id", "timestamp", "link_id"])
    d = m.queue_pred_x != m.queue_pred_y
    print(f"{name}: windows {STATS['windows']}, rows after T dropped (summed over windows) {STATS['rows_after_T_dropped']:,}; "
          f"cells {len(m):,}, different {int(d.sum())}, windows with a difference {m[d].window_id.nunique()}")


if __name__ == "__main__":
    main()
