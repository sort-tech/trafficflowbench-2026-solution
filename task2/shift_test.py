"""Robustness to a seed-level timing shift: time-of-day features looked up at tod+shift.

State features (speeds, flows, queue geometry) come from the true window; only the
time-of-day dependent inputs (tod, climatology, priors) are shifted. Out-of-fold models.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from build import predict_mean, ONGOING_THR
from build_dataset import keep_mask
from common import CACHE, H, PANELS, STEP, T0, Panel, aggregate
from features import FOLDS, Clim, Static, dow_of, ongoing_rows, onset_rows, link_features, online_features, train_window_arrays
from model import best_k_set, iou
from monthly_official import fold_models

VAR = sys.argv[1] if len(sys.argv) > 1 else "full"
SHIFTS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 6, -6, 12, -12, 18, -18]
THR = float(sys.argv[3]) if len(sys.argv) > 3 else ONGOING_THR


def main() -> None:
    import monthly_official
    monthly_official.VAR = VAR
    models = fold_models()
    mo = pd.read_csv(CACHE / "monthly_official_full.csv")  # same window set for every variant
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in FOLDS.items() for m in ms}
    rows = []
    for panel in PANELS:
        st = Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        clims = {k: Clim(P, qc, [m for f, ms in FOLDS.items() if f != k for m in ms], onset_t) for k in FOLDS}
        for w in mo[mo.panel == panel].itertuples():
            k = m2f[w.month]; clim = clims[k]
            rng = np.random.default_rng([PANELS.index(panel), int(w.t)])
            arr = train_window_arrays(P, w.t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
            sl = slice(w.t + 1, w.t + 1 + H); y, e = P.q[sl], P.elig[sl]
            month_start = int((pd.Timestamp(f"{w.month}-01T00:00:00Z") - T0) / STEP)
            online = online_features(P.speed, P.vcut, int(w.t), month_start)
            r = {"panel": panel, "condition": w.condition, "t": w.t}
            for s in SHIFTS:
                ts = w.t + s
                df = link_features(arr, st, clim, ts, online)
                pred = np.zeros((H, st.L), bool)
                if w.condition == "queue_onset":
                    df = onset_rows(df, clim, ts)
                    pred[H - 1] = best_k_set(predict_mean(models[("onset", k)], df))
                else:
                    d = ongoing_rows(df, clim, ts)
                    p = predict_mean(models[("ongoing", k)], d)
                    p[keep_mask(d) == 0] = 0
                    pred[d.h.to_numpy() - 1, d.link.to_numpy()] = p >= THR
                r[f"s{s}"] = iou(pred, y, e)
            rows.append(r)
        print(panel, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / f"shift_test_{VAR}.csv", index=False)
    for s in SHIFTS:
        a = aggregate(df, f"s{s}")
        print(f"{VAR} shift {s * 5:+4d} min: S={a['S_queue']:.3f} onset={a['onset']:.3f} ongoing={a['ongoing']:.3f}")


if __name__ == "__main__":
    main()
