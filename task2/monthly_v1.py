"""Month-by-month official-style evaluation for the lgb_v1 recipe (features_v1.py), with or without the causal
T-row restorer. Same windows, masking and fold scheme as monthly_official.py: fold models (v1 features,
seed SEED, 300 rounds) are fit without the window's 3-month fold, and with --causal-t the T row is filled by the
restorer trained without that fold.

Usage: monthly_v1.py [--causal-t] [--thr 0.4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb

import features_v1 as F1
from build import load_v1_dataset, predict_mean
from build_dataset_v1 import keep_mask, onset_rows, out_dir
from common import CACHE, H, PANELS, STEP, T0, Panel, aggregate, select_windows
from model import best_k_set, fit_dataset, iou

MD = Path(__file__).resolve().parent / "models" / "cv"


def fold_models(causal_t: bool) -> dict:
    tag = "v1_causal_t" if causal_t else "v1"
    out = {}
    for name, cond in (("onset", 1), ("ongoing", 0)):
        for k in F1.FOLDS:
            path = MD / f"{name}_{tag}_{k}.txt"
            if not path.exists():
                ds = load_v1_dataset(out_dir(causal_t), cond, drop_fold=k)
                m = fit_dataset(ds, 300)
                m.save_model(str(path))
                del ds
                print("fit", name, tag, k, flush=True)
            out[(name, k)] = [lgb.Booster(model_file=str(path))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--causal-t", action="store_true")
    ap.add_argument("--thr", type=float, default=0.4)
    args = ap.parse_args()
    tag = "v1_causal_t" if args.causal_t else "v1"
    models = fold_models(args.causal_t)
    rest = {}
    if args.causal_t:
        from restore_t import Restorer
        rest = {k: Restorer(k) for k in F1.FOLDS}
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    rows = []
    for panel in PANELS:
        st = F1.Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        clims = {k: F1.Clim(P, qc, [m for f, ms in F1.FOLDS.items() if f != k for m in ms], onset_t) for k in F1.FOLDS}
        for mon in months:
            start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
            end = int((pd.Timestamp(f"{mon}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            sel = select_windows(P, qc, max_per_condition=5, start=max(start, 12), end=end)
            k = m2f[mon]
            clim = clims[k]
            for w in sel.itertuples():
                rng = np.random.default_rng([PANELS.index(panel), int(w.t)])
                arr = F1.train_window_arrays(P, w.t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                df = F1.link_features(arr, st, clim, w.t, rest[k].fill(arr, st) if rest else None)
                pred = np.zeros((H, st.L), bool)
                if w.condition == "queue_onset":
                    df = onset_rows(df)
                    pred[H - 1] = best_k_set(predict_mean(models[("onset", k)], df))
                else:
                    d = F1.horizon_rows(df, clim, w.t)
                    p = predict_mean(models[("ongoing", k)], d)
                    p[keep_mask(d) == 0] = 0
                    pred[d.h.to_numpy() - 1, d.link.to_numpy()] = p >= args.thr
                sl = slice(w.t + 1, w.t + 1 + H)
                y, e = P.q[sl], P.elig[sl]
                pers = np.broadcast_to((P.q & P.elig)[w.t - 1], (H, st.L))
                rows.append({"panel": panel, "month": mon, "condition": w.condition, "t": w.t,
                             "ours": iou(pred, y, e), "pers": iou(pers, y, e), "npos": int((y & e).sum())})
        print(panel, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / f"monthly_official_{tag}_{args.thr}.csv", index=False)
    res = []
    for mon, g in df.groupby("month"):
        a = aggregate(g, "ours")
        res.append({"month": mon, "S_ours": a["S_queue"], "onset": a["onset"], "ongoing": a["ongoing"], "n": len(g)})
    r = pd.DataFrame(res)
    print(r.round(3).to_string(index=False))
    print(tag, "mean", r[["S_ours", "onset", "ongoing"]].mean().round(4).to_dict(),
          "min", r.S_ours.min().round(3), "max", r.S_ours.max().round(3))


if __name__ == "__main__":
    main()
