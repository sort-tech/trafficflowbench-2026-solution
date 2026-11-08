"""Compare out-of-pattern queues (queued where the train climatology says < 2%) across splits.

Public observations only: train (out-of-fold climatology), validation/private masked views, and the
released window histories. Counts are per visible eligible cell, so masking does not bias them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE, NHIST, PANELS, REL, STEP, T0, Panel
from features import FOLDS, Clim, Static, dow_of, release_windows

THR = 0.02


def anomaly_map(speed, pct, vcut, clim_q, t_abs0):
    """Per row: (# visible eligible cells, # queued, # queued where climatology < THR)."""
    n = speed.shape[0]
    t_abs = t_abs0 + np.arange(n)
    dt = (dow_of(t_abs) >= 5).astype(int)
    c = clim_q[dt, t_abs % 288]
    vis = ~np.isnan(speed) & (pct >= 75)
    q = (np.nan_to_num(speed, nan=1e9) <= vcut) & vis
    return vis.sum(1), q.sum(1), (q & (c < THR)).sum(1)


def main() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in FOLDS.items() for m in ms}
    day_rows, win_rows = [], []
    mo = pd.read_csv(CACHE / "monthly_official_full.csv")
    for panel in PANELS:
        st = Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        ot = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        clims = {k: Clim(P, qc, [m for f, ms in FOLDS.items() if f != k for m in ms], ot) for k in FOLDS}
        clim_all = Clim(P, qc, sorted(W.month.unique()), ot)
        months = sorted(W.month.unique())
        for mon in months:
            s = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
            e = int((pd.Timestamp(f"{mon}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            v, q, a = anomaly_map(P.speed[s:e], P.pct[s:e], P.vcut, clims[m2f[mon]].q, s)
            day_rows.append({"panel": panel, "src": "train", "month": mon, "vis": v.sum(), "q": q.sum(), "anom": a.sum(),
                             "anom_rows": int((a >= 2).sum()), "rows": len(a)})
        for split, start in (("validation", "2031-03-01"), ("private", "2031-04-01")):
            z = np.load(CACHE / f"{panel}_{split}.npz")
            s = int((pd.Timestamp(f"{start}T00:00:00Z") - T0) / STEP)
            v, q, a = anomaly_map(z["speed"], z["pct"], P.vcut, clim_all.q, s)
            day_rows.append({"panel": panel, "src": split, "month": split, "vis": v.sum(), "q": q.sum(), "anom": a.sum(),
                             "anom_rows": int((a >= 2).sum()), "rows": len(a)})
        # window histories: official-style train windows vs release windows
        for w in mo[mo.panel == panel].itertuples():
            k = m2f[w.month]
            v, q, a = anomaly_map(P.speed[w.t - NHIST:w.t + 1], P.pct[w.t - NHIST:w.t + 1], P.vcut, clims[k].q, w.t - NHIST)
            win_rows.append({"panel": panel, "src": "train", "cond": w.condition, "anom": int(a.sum()), "q": int(q.sum())})
        for split in ("validation", "private"):
            for w, arr, t_idx in release_windows(panel, split, st):
                v, q, a = anomaly_map(arr["speed"], arr["pct"], P.vcut, clim_all.q, t_idx - NHIST)
                win_rows.append({"panel": panel, "src": split, "cond": w.condition, "anom": int(a.sum()), "q": int(q.sum())})
        print(panel, flush=True)
    d = pd.DataFrame(day_rows)
    d["anom_per_1k_vis"] = d.anom / d.vis * 1000
    d["anom_share_of_q"] = d.anom / d.q.clip(lower=1)
    d["anom_row_frac"] = d.anom_rows / d.rows
    pd.set_option("display.width", 220)
    t = d[d.src == "train"].groupby("panel")[["anom_per_1k_vis", "anom_share_of_q", "anom_row_frac"]].agg(["mean", "max"])
    r = d[d.src != "train"].set_index(["panel", "src"])[["anom_per_1k_vis", "anom_share_of_q", "anom_row_frac"]].unstack()
    print("train months (mean, max over 9 months):"); print(t.round(3))
    print("validation / private:"); print(r.round(3))
    w = pd.DataFrame(win_rows)
    w["has_anom"] = w.anom >= 3
    w["anom_share"] = w.anom / w.q.clip(lower=1)
    print(w.groupby(["cond", "src"])[["has_anom", "anom_share"]].mean().round(3))
    d.to_csv(CACHE / "incident_days.csv", index=False); w.to_csv(CACHE / "incident_windows.csv", index=False)


if __name__ == "__main__":
    main()
