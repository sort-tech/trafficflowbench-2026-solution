"""Step 2-3: ongoing aug models trained on 3 independent draws of the time-shift augmentation (shift seeds 1-3,
tables cache/aug_m3/S<s>/ds_<PANEL>.parquet from build_dataset.py --shift-seed s) x 3 tree seeds = 9 models.

  aug_m3.py fit <s> [full]   full models models/aug_m3/ongoing_S<s>_s<i>.txt (seed SEED + 1000 i, i = 0..2) and fold models
                      models/cv/ongoing_augm3S<s>_<k>.txt (seed SEED); "full" skips the fold models (reproduction)
  aug_m3.py oof       out-of-fold probability (mean of the 3 shift-seed fold models) on the cand_eval ongoing windows,
                      shifts 0, +-45, +-60 min -> cache/aug_m3_oof.pkl
Same features and parameters as the aug variant (num_threads 3, deterministic, force_row_wise).
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

SHIFTS = (0, 9, -9, 12, -12)


def fit(s: int, full_only: bool = False) -> None:
    d = CACHE / "aug_m3" / f"S{s}"
    out = MODELS / "aug_m3"; out.mkdir(parents=True, exist_ok=True)
    names = pq.read_schema(d / f"ds_{PANELS[0]}.parquet").names
    feats = [c for c in names if c not in NON_FEATURES and c not in VARIANTS["aug"]]
    assert feats == lgb.Booster(model_file=str(MODELS / "aug" / "ongoing_s0.txt")).feature_name()
    for fold in (None,) if full_only else (None, "A", "B", "C"):
        paths = [out / f"ongoing_S{s}_s{i}.txt" for i in range(3)] if fold is None else [MODELS / "cv" / f"ongoing_augm3S{s}_{fold}.txt"]
        if all(p.exists() for p in paths):
            continue
        filters = [("cond", "=", 0), ("vis", "=", 1), ("keep", "=", 1)] + ([("fold", "!=", fold)] if fold else [])
        df = pd.concat([pd.read_parquet(d / f"ds_{p}.parquet", columns=feats + ["y"], filters=filters) for p in PANELS],
                       ignore_index=True)
        ds = make_dataset(df, feats)
        del df
        for i, p in enumerate(paths):
            fit_dataset(ds, 300, {"seed": SEED + 1000 * i}).save_model(str(p))
        del ds
        print("fit", s, fold, flush=True)


def oof() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    M = {k: [lgb.Booster(model_file=str(MODELS / "cv" / f"ongoing_augm3S{s}_{k}.txt")) for s in (1, 2, 3)] for k in F1.FOLDS}
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
                rec = {}
                for sh in SHIFTS:
                    dc = FC.ongoing_rows(FC.link_features(arr, stc, cc[k], t + sh, online), cc[k], t + sh)
                    p = predict_mean(M[k], dc); p[keep_c(dc) == 0] = 0
                    a = np.zeros((H, stc.L), np.float32); a[dc.h.to_numpy() - 1, dc.link.to_numpy()] = p
                    rec[f"augm3_{sh}"] = a
                go[(panel, t)] = rec
        print(panel, flush=True)
    pickle.dump({"go": go}, open(CACHE / "aug_m3_oof.pkl", "wb"))


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(int(sys.argv[2]), len(sys.argv) > 3 and sys.argv[3] == "full")
    else:
        oof()
