"""Month-by-month official-style evaluation on train.

For each train month and panel, the selector is run with the official cap (5 onset + 5 ongoing,
chronological from the month start). Each window is predicted with fold models fit on the other
3-month folds (out-of-fold), and S_queue is computed per month exactly like the leaderboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb

from build import predict_mean, ONGOING_THR
from build_dataset import keep_mask
from common import CACHE, H, PANELS, STEP, T0, Panel, aggregate, select_windows
from features import FOLDS, Clim, Static, dow_of, ongoing_rows, onset_rows, link_features, online_features, train_window_arrays
from model import best_k_set, feature_cols, fit, iou, VARIANTS

MD = Path(__file__).resolve().parent / "models" / "cv"
VAR = sys.argv[1] if len(sys.argv) > 1 else "full"
THR = float(sys.argv[2]) if len(sys.argv) > 2 and __name__ == "__main__" else ONGOING_THR


def fold_models() -> dict:
    MD.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, cond in (("onset", 1), ("ongoing", 0)):
        for k in FOLDS:
            path = MD / f"{name}_{VAR}_{k}.txt"
            if not path.exists():
                parts = []
                for p in PANELS:
                    filters = [("cond", "=", cond), ("vis", "=", 1)] + ([("keep", "=", 1)] if cond == 0 else [])
                    d = pd.read_parquet(CACHE / f"ds_{p}.parquet", filters=filters)
                    parts.append(d[d.fold != k])
                df = pd.concat(parts, ignore_index=True)
                feats = [f for f in feature_cols(df) if f not in VARIANTS[VAR]]
                m = fit(df, feats, 300)
                m.save_model(str(path))
                del df
                print("fit", name, k, flush=True)
            out[(name, k)] = [lgb.Booster(model_file=str(path))]
    return out


def main() -> None:
    models = fold_models()
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in FOLDS.items() for m in ms}
    months = [m for ms in FOLDS.values() for m in ms]
    rows = []
    for panel in PANELS:
        st = Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        clims = {k: Clim(P, qc, [m for f, ms in FOLDS.items() if f != k for m in ms], onset_t) for k in FOLDS}
        for mon in months:
            start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
            end = int((pd.Timestamp(f"{mon}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            sel = select_windows(P, qc, max_per_condition=5, start=max(start, 12), end=end)
            k = m2f[mon]
            clim = clims[k]
            for w in sel.itertuples():
                rng = np.random.default_rng([PANELS.index(panel), int(w.t)])
                arr = train_window_arrays(P, w.t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                month_start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
                df = link_features(arr, st, clim, w.t, online_features(P.speed, P.vcut, int(w.t), month_start))
                pred = np.zeros((H, st.L), bool)
                if w.condition == "queue_onset":
                    df = onset_rows(df, clim, w.t)
                    pred[H - 1] = best_k_set(predict_mean(models[("onset", k)], df))
                else:
                    d = ongoing_rows(df, clim, w.t)
                    p = predict_mean(models[("ongoing", k)], d)
                    p[keep_mask(d) == 0] = 0
                    pred[d.h.to_numpy() - 1, d.link.to_numpy()] = p >= THR
                sl = slice(w.t + 1, w.t + 1 + H)
                y, e = P.q[sl], P.elig[sl]
                pers = np.broadcast_to((P.q & P.elig)[w.t - 1], (H, st.L))
                rows.append({"panel": panel, "month": mon, "condition": w.condition, "t": w.t,
                             "ours": iou(pred, y, e), "pers": iou(pers, y, e),
                             "npos": int((y & e).sum())})
        print(panel, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / f"monthly_official_{VAR}_{THR}.csv", index=False)
    res = []
    for mon, g in df.groupby("month"):
        a = aggregate(g, "ours"); b = aggregate(g, "pers")
        res.append({"month": mon, "S_ours": a["S_queue"], "onset": a["onset"], "ongoing": a["ongoing"],
                    "S_pers": b["S_queue"], "pers_ongoing": b["ongoing"], "n": len(g)})
    r = pd.DataFrame(res)
    print(r.round(3).to_string(index=False))
    print("mean", r[["S_ours", "onset", "ongoing", "S_pers", "pers_ongoing"]].mean().round(3).to_dict(),
          "min", r.S_ours.min().round(3), "max", r.S_ours.max().round(3))


if __name__ == "__main__":
    main()
