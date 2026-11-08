"""Onset models on the frozen lgb_v1 features plus bottleneck threshold features (thr_feats.py).

Table cache/v1_thr/ds_<PANEL>.parquet: all onset pseudo windows (as build_dataset_v1.py), v1 onset rows + THR_COLS.
Thresholds for a fold's windows are fitted on the days of the other two folds; the full model's thresholds
(release) on all train days. Models: models/v1_thr/onset_s{0,1,2}.txt (seed SEED + 1000 i), fold models
models/cv/onset_v1_thr_<k>.txt (seed SEED). OOF: cand_eval onset windows, time-of-day lookups shifted by
0, +-45, +-60 min (threshold features are state features and stay) -> cache/v1_thr_oof.pkl.

Usage: v1_thr.py build | fit | oof
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb

import features_v1 as F1
from build import MODELS, predict_mean
from build_dataset_v1 import onset_rows
from common import CACHE, H, PANELS, STEP, T0, Panel, select_windows
from model import NON_FEATURES, SEED, fit_dataset, make_dataset
from thr_feats import THR_COLS, day_mask_for_months, fit_thresholds, window_features

OUT = CACHE / "v1_thr"
SPLIT_CLIM = float(os.environ["TFB_SPLIT_CLIM"]) if os.environ.get("TFB_SPLIT_CLIM") else None   # B18 retrained variant
SHIFTS = (0, 9, -9, 12, -12)


def fold_thresholds(P) -> dict:
    out = {k: fit_thresholds(P, day_mask_for_months(P, [m for f, ms in F1.FOLDS.items() if f != k for m in ms])) for k in F1.FOLDS}
    out["all"] = fit_thresholds(P, np.ones(P.n_t // 288, bool))
    return out


def rows(arr, st, clim, t: int, s: int, thr) -> pd.DataFrame:
    df = onset_rows(F1.link_features(arr, st, clim, t + s))
    tf = window_features(arr, st.lanes, thr)
    for c in THR_COLS:
        df[c] = tf[c].to_numpy()
    return df


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    W0 = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    thr_log = {}
    for panel in PANELS:
        P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy"); st = F1.Static(panel)
        W = W0[W0.panel == panel]
        onset_t = W[W.condition == "queue_onset"].t.to_numpy()
        Won = W[W.condition == "queue_onset"].sort_values("t")
        thr = fold_thresholds(P)
        thr_log[panel] = thr
        SC = None
        if SPLIT_CLIM is not None:
            from split_clim import TrainSplitClims
            SC = TrainSplitClims(P, qc, onset_t, "v1", SPLIT_CLIM)
        out = []
        for fold, months in F1.FOLDS.items():
            clim_fold = F1.Clim(P, qc, [m for f, ms in F1.FOLDS.items() if f != fold for m in ms], onset_t)
            for w in Won[Won.month.map(m2f) == fold].itertuples(index=False):
                t = int(w.t)
                clim = clim_fold if SC is None else SC.blend(clim_fold, t)
                rng = np.random.default_rng([PANELS.index(panel), t])
                arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                df = rows(arr, st, clim, t, 0, thr[fold])
                li = np.arange(st.L)
                df["y"] = P.q[t + H, li].astype(np.int8)
                df["vis"] = P.visible[t + H, li].astype(np.int8)
                df["t"] = np.int32(t); df["fold"] = fold
                out.append(df)
        res = pd.concat(out, ignore_index=True)
        for c in res.columns:
            if res[c].dtype == np.float64:
                res[c] = res[c].astype(np.float32)
        res.to_parquet(OUT / f"ds_{panel}.parquet", index=False)
        print(panel, "rows", len(res), "groups(all)", [(a, b, h, round(c)) for a, b, h, c in thr["all"]], flush=True)
    pickle.dump(thr_log, open(OUT / "thresholds.pkl", "wb"))


def fit() -> None:
    (MODELS / "v1_thr").mkdir(exist_ok=True)
    for fold in (None, "A", "B", "C"):
        paths = [MODELS / "v1_thr" / f"onset_s{i}.txt" for i in range(3)] if fold is None else [MODELS / "cv" / f"onset_v1_thr_{fold}.txt"]
        if all(p.exists() for p in paths):
            continue
        filters = [("vis", "=", 1)] + ([("fold", "!=", fold)] if fold else [])
        df = pd.concat([pd.read_parquet(OUT / f"ds_{p}.parquet", filters=filters) for p in PANELS], ignore_index=True)
        fs = [c for c in df.columns if c not in NON_FEATURES]
        ds = make_dataset(df, fs)
        del df
        for i, p in enumerate(paths):
            fit_dataset(ds, 300, {"seed": SEED + 1000 * i}).save_model(str(p))
        print("fit", fold, flush=True)


def oof() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    M = {k: [lgb.Booster(model_file=str(MODELS / "cv" / f"onset_v1_thr_{k}.txt"))] for k in F1.FOLDS}
    thr_all = pickle.load(open(OUT / "thresholds.pkl", "rb"))
    on = {}
    for panel in PANELS:
        st = F1.Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        c1 = {k: F1.Clim(P, qc, [m for f, ms in F1.FOLDS.items() if f != k for m in ms], onset_t) for k in F1.FOLDS}
        for mon in months:
            start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
            end = int((pd.Timestamp(f"{mon}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            sel = select_windows(P, qc, max_per_condition=5, start=max(start, 12), end=end)
            k = m2f[mon]
            for w in sel.itertuples():
                if w.condition != "queue_onset":
                    continue
                t = int(w.t)
                rng = np.random.default_rng([PANELS.index(panel), t])
                arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                on[(panel, t)] = {f"v1_thr_{sh}": predict_mean(M[k], rows(arr, st, c1[k], t, sh, thr_all[panel][k])) for sh in SHIFTS}
        print(panel, flush=True)
    pickle.dump({"on": on}, open(CACHE / "v1_thr_oof.pkl", "wb"))


if __name__ == "__main__":
    {"build": build, "fit": fit, "oof": oof}[sys.argv[1]]()
