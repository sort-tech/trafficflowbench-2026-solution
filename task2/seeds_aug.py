"""Extra seeds for the out-of-fold aug ongoing models (the release aug models already average seeds
SEED, SEED+1000, SEED+2000): fold models models/cv/ongoing_augs<i>_<k>.txt for i = 1, 2, then 3-seed mean
probabilities on the cand_eval ongoing windows (shifts 0, +-45, +-60 min) -> cache/seeds_aug.pkl.
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
from build import predict_mean
from build_dataset import keep_mask as keep_c
from common import CACHE, H, PANELS, STEP, T0, Panel, select_windows
from model import NON_FEATURES, SEED, VARIANTS, fit_dataset, make_dataset

MD = Path(__file__).resolve().parent / "models" / "cv"
SHIFTS = (0, 9, -9, 12, -12)


def fit_folds() -> None:
    names = pq.read_schema(CACHE / f"ds_{PANELS[0]}.parquet").names
    feats = [c for c in names if c not in NON_FEATURES and c not in VARIANTS["aug"]]
    ref = lgb.Booster(model_file=str(MD / "ongoing_aug_A.txt")).feature_name()
    assert feats == ref, "feature list differs from the existing aug fold models"
    for k in F1.FOLDS:
        paths = {i: MD / f"ongoing_augs{i}_{k}.txt" for i in (1, 2)}
        if all(p.exists() for p in paths.values()):
            continue
        filters = [("cond", "=", 0), ("vis", "=", 1), ("keep", "=", 1), ("fold", "!=", k)]
        df = pd.concat([pd.read_parquet(CACHE / f"ds_{p}.parquet", columns=feats + ["y"], filters=filters) for p in PANELS],
                       ignore_index=True)
        ds = make_dataset(df, feats)
        del df
        for i, p in paths.items():
            fit_dataset(ds, 300, {"seed": SEED + 1000 * i}).save_model(str(p))
        del ds
        print("fit", k, flush=True)


def oof() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    M = {k: [lgb.Booster(model_file=str(MD / f"ongoing_aug_{k}.txt"))]
         + [lgb.Booster(model_file=str(MD / f"ongoing_augs{i}_{k}.txt")) for i in (1, 2)] for k in F1.FOLDS}
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
                    rec[f"augb_{sh}"] = a
                go[(panel, t)] = rec
        print(panel, flush=True)
    pickle.dump({"go": go}, open(CACHE / "seeds_aug.pkl", "wb"))


if __name__ == "__main__":
    fit_folds()
    oof()
