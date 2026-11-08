"""Out-of-fold cell probabilities of several variants on the official-style monthly windows.

For every window of cache/monthly_official_full.csv (5 onset + 5 ongoing per panel and train month),
fold models fit on the other 3-month folds predict:
  onset   : aug (at each time-of-day shift), onphys, online
  ongoing : aug (at each shift), phys, notod
Shift-invariant variants are predicted once. Output: cache/ens_onset.parquet, cache/ens_ongoing.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import monthly_official
from build import predict_mean
from build_dataset import keep_mask
from common import CACHE, H, PANELS, STEP, T0, Panel
from features import FOLDS, Clim, Static, link_features, ongoing_rows, online_features, onset_rows, train_window_arrays

SHIFTS = [0, 6, -6, 12, -12, 18, -18]
ON_VARS = ["onphys", "online"]
GO_VARS = ["phys", "notod"]


def load_models() -> dict:
    out = {}
    for var in set(ON_VARS + GO_VARS + ["aug"]):
        monthly_official.VAR = var
        out.update({(var,) + k: v for k, v in monthly_official.fold_models().items()})
    return out


def main() -> None:
    models = load_models()
    mo = pd.read_csv(CACHE / "monthly_official_full.csv")
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in FOLDS.items() for m in ms}
    on_rows, go_rows = [], []
    for panel in PANELS:
        st = Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        clims = {k: Clim(P, qc, [m for f, ms in FOLDS.items() if f != k for m in ms], onset_t) for k in FOLDS}
        for w in mo[mo.panel == panel].itertuples():
            k = m2f[w.month]; clim = clims[k]
            rng = np.random.default_rng([PANELS.index(panel), int(w.t)])
            arr = train_window_arrays(P, w.t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
            month_start = int((pd.Timestamp(f"{w.month}-01T00:00:00Z") - T0) / STEP)
            online = online_features(P.speed, P.vcut, int(w.t), month_start)
            base = link_features(arr, st, clim, w.t, online)
            if w.condition == "queue_onset":
                d = onset_rows(base, clim, w.t)
                rec = pd.DataFrame({"panel": panel, "month": w.month, "t": w.t, "link": np.arange(st.L),
                                    "y": P.q[w.t + H], "elig": P.elig[w.t + H]})
                for var in ON_VARS:
                    rec[f"p_{var}"] = predict_mean(models[(var, "onset", k)], d)
                for s in SHIFTS:
                    ds = onset_rows(link_features(arr, st, clim, w.t + s, online), clim, w.t + s)
                    rec[f"p_aug_{s}"] = predict_mean(models[("aug", "onset", k)], ds)
                on_rows.append(rec)
            else:
                d = ongoing_rows(base, clim, w.t)
                keep = keep_mask(d) == 1
                h = d.h.to_numpy().astype(int); li = d.link.to_numpy().astype(int)
                rec = pd.DataFrame({"panel": panel, "month": w.month, "t": w.t, "link": li, "h": h,
                                    "y": P.q[w.t + h, li], "elig": P.elig[w.t + h, li]})
                for var in GO_VARS:
                    rec[f"p_{var}"] = np.where(keep, predict_mean(models[(var, "ongoing", k)], d), 0.0)
                for s in SHIFTS:
                    ds = ongoing_rows(link_features(arr, st, clim, w.t + s, online), clim, w.t + s)
                    rec[f"p_aug_{s}"] = np.where(keep_mask(ds) == 1, predict_mean(models[("aug", "ongoing", k)], ds), 0.0)
                go_rows.append(rec)
        print(panel, flush=True)
    pd.concat(on_rows, ignore_index=True).to_parquet(CACHE / "ens_onset.parquet", index=False)
    pd.concat(go_rows, ignore_index=True).to_parquet(CACHE / "ens_ongoing.parquet", index=False)


if __name__ == "__main__":
    main()
