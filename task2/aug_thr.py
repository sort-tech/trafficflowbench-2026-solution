"""(d) bottleneck threshold features in the ongoing aug model (aug_thr).

  aug_thr.py feats   thr columns for every ongoing window of cache/ds_<PANEL>.parquet (same window arrays and
                     masking as build_dataset.py; thresholds fitted on the other folds' days) -> cache/aug_thr/thr_<PANEL>.parquet
  aug_thr.py fit [folds]   aug features + THR_COLS: fold models models/cv/ongoing_aug_thr_<k>.txt (seed SEED) and full models
                     models/aug_thr/ongoing_s{0,1,2}.txt (seed SEED + 1000 i; thresholds from all train days at release)
  aug_thr.py oof     out-of-fold probabilities on the cand_eval ongoing windows, shifts 0, +-45, +-60 min -> cache/aug_thr_oof.pkl
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq

import features as FC
import features_v1 as F1
from build import MODELS, predict_mean
from build_dataset import keep_mask as keep_c
from common import CACHE, H, PANELS, STEP, T0, Panel, select_windows
from model import NON_FEATURES, SEED, VARIANTS, fit_dataset, make_dataset
from thr_feats import THR_COLS, day_mask_for_months, fit_thresholds, window_features

OUT = CACHE / "aug_thr"
SHIFTS = (0, 9, -9, 12, -12)


def fold_thr(P) -> dict:
    out = {k: fit_thresholds(P, day_mask_for_months(P, [m for f, ms in F1.FOLDS.items() if f != k for m in ms])) for k in F1.FOLDS}
    out["all"] = fit_thresholds(P, np.ones(P.n_t // 288, bool))
    return out


def feats() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    thr_log = {}
    for panel in PANELS:
        P = Panel(panel); st = FC.Static(panel)
        thr = fold_thr(P); thr_log[panel] = thr
        w = pd.read_parquet(CACHE / f"ds_{panel}.parquet", columns=["t", "cond", "fold"], filters=[("cond", "=", 0)]).drop_duplicates("t")
        parts = []
        for r in w.itertuples(index=False):
            t = int(r.t)
            rng = np.random.default_rng([PANELS.index(panel), t])
            arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
            tf = window_features(arr, st.lanes, thr[r.fold])
            tf["t"] = np.int32(t); tf["link"] = np.arange(st.L, dtype=np.int16)
            parts.append(tf)
        pd.concat(parts, ignore_index=True).to_parquet(OUT / f"thr_{panel}.parquet", index=False)
        print(panel, len(w), flush=True)
    pickle.dump(thr_log, open(OUT / "thresholds.pkl", "wb"))


def fit(folds_only: bool = False) -> None:
    (MODELS / "aug_thr").mkdir(exist_ok=True)
    names = pq.read_schema(CACHE / f"ds_{PANELS[0]}.parquet").names
    base = [c for c in names if c not in NON_FEATURES and c not in VARIANTS["aug"]]
    fs = base + THR_COLS
    for fold in ("A", "B", "C") if folds_only else (None, "A", "B", "C"):
        paths = [MODELS / "aug_thr" / f"ongoing_s{i}.txt" for i in range(3)] if fold is None else [MODELS / "cv" / f"ongoing_aug_thr_{fold}.txt"]
        if all(p.exists() for p in paths):
            continue
        filters = [("cond", "=", 0), ("vis", "=", 1), ("keep", "=", 1)] + ([("fold", "!=", fold)] if fold else [])
        parts = []
        for p in PANELS:
            d = pd.read_parquet(CACHE / f"ds_{p}.parquet", columns=base + ["y", "t", "link"], filters=filters)
            tf = pd.read_parquet(OUT / f"thr_{p}.parquet")
            parts.append(d.merge(tf, on=["t", "link"], how="left").drop(columns=["t", "link"]))
            del d
        df = pd.concat(parts, ignore_index=True); del parts
        ds = make_dataset(df, fs)
        del df
        for i, path in enumerate(paths):
            fit_dataset(ds, 300, {"seed": SEED + 1000 * i}).save_model(str(path))
        del ds
        print("fit", fold, flush=True)


def oof() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    M = {k: [lgb.Booster(model_file=str(MODELS / "cv" / f"ongoing_aug_thr_{k}.txt"))] for k in F1.FOLDS}
    thr_all = pickle.load(open(OUT / "thresholds.pkl", "rb"))
    go = {}
    for panel in PANELS:
        stc = FC.Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        cc = {k: FC.Clim(P, qc, [m for f, ms in F1.FOLDS.items() if f != k for m in ms], onset_t) for k in F1.FOLDS}
        for mon in months:
            start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
            end = int((pd.Timestamp(f"{mon}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            sel = select_windows(P, qc, max_per_condition=5, start=max(start, 12), end=end)
            k = m2f[mon]
            for w in sel.itertuples():
                if w.condition != "queue_ongoing":
                    continue
                t = int(w.t)
                rng = np.random.default_rng([PANELS.index(panel), t])
                arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                online = FC.online_features(P.speed, P.vcut, t, start)
                tf = window_features(arr, stc.lanes, thr_all[panel][k])
                rec = {}
                for sh in SHIFTS:
                    dc = FC.ongoing_rows(FC.link_features(arr, stc, cc[k], t + sh, online), cc[k], t + sh)
                    lk = dc.link.to_numpy().astype(int)
                    for c in THR_COLS:
                        dc[c] = tf[c].to_numpy()[lk]
                    p = predict_mean(M[k], dc); p[keep_c(dc) == 0] = 0
                    a = np.zeros((H, stc.L), np.float32); a[dc.h.to_numpy() - 1, lk] = p
                    rec[f"augthr_{sh}"] = a
                go[(panel, t)] = rec
        print(panel, flush=True)
    pickle.dump({"go": go}, open(CACHE / "aug_thr_oof.pkl", "wb"))


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(len(sys.argv) > 2 and sys.argv[2] == "folds")
    else:
        {"feats": feats, "oof": oof}[sys.argv[1]]()
