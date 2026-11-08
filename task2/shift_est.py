"""Estimate a split's time-of-day shift relative to the train climatology from pre-T data only.

For a window at T, the earlier days of the same split (up to ONLINE_DAYS) give an observed queue map
q_obs[day, tod, link]. The lag minimizing sum (q_obs - clim.q[daytype, tod + lag, link])^2 over
04:00-22:00 is the correction: looking the climatology up at tod + lag aligns it with the split.
Only whole days before T's day are read (asserted).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE, PANELS, REL, STEP, T0, Panel
from features import ONLINE_DAYS, Clim, Static, dow_of

LAGS = np.arange(-24, 25)          # +-120 min
TOD = np.arange(48, 264)           # 04:00-22:00


def estimate_shift(speed: np.ndarray, vcut: np.ndarray, t: int, split_start: int, t_abs: int,
                   clim_q: np.ndarray, max_days: int = ONLINE_DAYS) -> tuple[int, float, int]:
    """speed: split array; t: index of T in it; t_abs: index of T from T0 (for weekday); clim_q: [2, 288, L].
    Returns (lag, gain = relative SSE drop vs lag 0, days used)."""
    tod_t = t % 288
    day0 = t - tod_t
    ndays = int(min(max((day0 - split_start) // 288, 0), max_days))
    if ndays == 0:
        return 0, 0.0, 0
    sse = np.zeros(len(LAGS))
    for d in range(1, ndays + 1):
        ds = day0 - 288 * d
        assert ds + 288 <= day0 <= t, "only whole earlier days are used"
        sp = speed[ds + TOD]
        vis = ~np.isnan(sp)
        q = (np.nan_to_num(sp, nan=1e9) <= vcut).astype(np.float32)
        dt = int(dow_of(t_abs - 288 * d) >= 5)
        for i, lag in enumerate(LAGS):
            c = clim_q[dt, (TOD + lag) % 288]
            sse[i] += float((((q - c) ** 2) * vis).sum())
    i0 = int(np.where(LAGS == 0)[0][0])
    ib = int(np.argmin(sse))
    gain = (sse[i0] - sse[ib]) / max(sse[i0], 1e-9)
    return int(LAGS[ib]), float(gain), ndays


def release_lags() -> pd.DataFrame:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    rows = []
    for panel in PANELS:
        st = Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        clim = Clim(P, qc, sorted(W.month.unique()), onset_t)
        for split, start in (("validation", "2031-03-01"), ("private", "2031-04-01")):
            sp = np.load(CACHE / f"{panel}_{split}.npz")["speed"]
            off = int((pd.Timestamp(f"{start}T00:00:00Z") - T0) / STEP)
            wi = pd.read_csv(REL / "task2" / panel / split / "window_index.csv")
            for w in wi.itertuples():
                t_abs = int((pd.Timestamp(w.forecast_origin) - T0) / STEP)
                lag, gain, nd = estimate_shift(sp, st.vcut, t_abs - off, 0, t_abs, clim.q)
                rows.append({"panel": panel, "split": split, "window_id": w.window_id, "cond": w.condition,
                             "days": nd, "lag_min": lag * 5, "gain": round(gain, 3)})
        print(panel, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = release_lags()
    df.to_csv(CACHE / "release_lags.csv", index=False)
    pd.set_option("display.width", 200)
    last = df[df.days > 0].sort_values("days").groupby(["panel", "split"]).tail(1)
    print(last[["panel", "split", "days", "lag_min", "gain"]].to_string(index=False))
