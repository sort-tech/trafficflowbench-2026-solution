"""(5) Bottleneck threshold features on the time-of-day-free onset tables: aug_thr and onphys_thr onset models.

Rows: onset windows of cache/ds_<PANEL>.parquet (current features, aug_* time-shift columns) + THR_COLS from the same
window arrays, thresholds fitted on the other folds' days (cache/onset_thr2/thr_<PANEL>.parquet). Feature sets:
aug variant (no tod / climatology) and onphys variant (no tod / climatology / aug_*), each + THR_COLS.
Models: models/{onset_augthr,onset_onphysthr}/onset_s{0,1,2}.txt (seed SEED + 1000 i), fold models
models/cv/onset_{augthr,onphysthr}_<k>.txt; OOF on the cand_eval onset windows (shifts 0, +-45, +-60 min; online and
threshold features at the true T) -> cache/onset_thr2_oof.pkl.

Usage: onset_thr2.py feats | fit | oof
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
from aug_thr import fold_thr
from build import MODELS, predict_mean
from common import CACHE, PANELS, STEP, T0, Panel, select_windows
from model import NON_FEATURES, SEED, VARIANTS, fit_dataset, make_dataset
from thr_feats import THR_COLS, window_features

OUT = CACHE / "onset_thr2"
SHIFTS = (0, 9, -9, 12, -12)
NAMES = {"augthr": "aug", "onphysthr": "onphys"}


def feats() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = {}
    for panel in PANELS:
        P = Panel(panel); st = FC.Static(panel)
        thr = fold_thr(P); log[panel] = thr
        w = pd.read_parquet(CACHE / f"ds_{panel}.parquet", columns=["t", "fold"], filters=[("cond", "=", 1)]).drop_duplicates("t")
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
    pickle.dump(log, open(OUT / "thresholds.pkl", "wb"))


def fit() -> None:
    names = pq.read_schema(CACHE / f"ds_{PANELS[0]}.parquet").names
    for name, var in NAMES.items():
        base = [c for c in names if c not in NON_FEATURES and c not in VARIANTS[var]]
        fs = base + THR_COLS
        (MODELS / f"onset_{name}").mkdir(exist_ok=True)
        for fold in (None, "A", "B", "C"):
            paths = [MODELS / f"onset_{name}" / f"onset_s{i}.txt" for i in range(3)] if fold is None \
                else [MODELS / "cv" / f"onset_{name}_{fold}.txt"]
            if all(p.exists() for p in paths):
                continue
            filters = [("cond", "=", 1), ("vis", "=", 1)] + ([("fold", "!=", fold)] if fold else [])
            parts = []
            for p in PANELS:
                d = pd.read_parquet(CACHE / f"ds_{p}.parquet", columns=base + ["y", "t", "link"], filters=filters)
                parts.append(d.merge(pd.read_parquet(OUT / f"thr_{p}.parquet"), on=["t", "link"], how="left").drop(columns=["t", "link"]))
            df = pd.concat(parts, ignore_index=True); del parts
            ds = make_dataset(df, fs); del df
            for i, path in enumerate(paths):
                fit_dataset(ds, 300, {"seed": SEED + 1000 * i}).save_model(str(path))
            print("fit", name, fold, flush=True)


def oof() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    M = {k: {n: [lgb.Booster(model_file=str(MODELS / "cv" / f"onset_{n}_{k}.txt"))] for n in NAMES} for k in F1.FOLDS}
    thr_all = pickle.load(open(OUT / "thresholds.pkl", "rb"))
    on = {}
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
                if w.condition != "queue_onset":
                    continue
                t = int(w.t)
                rng = np.random.default_rng([PANELS.index(panel), t])
                arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                online = FC.online_features(P.speed, P.vcut, t, start)
                tf = window_features(arr, stc.lanes, thr_all[panel][k])
                rec = {}
                for sh in SHIFTS:
                    d = FC.onset_rows(FC.link_features(arr, stc, cc[k], t + sh, online), cc[k], t + sh)
                    for c in THR_COLS:
                        d[c] = tf[c].to_numpy()
                    for n in NAMES:
                        rec[f"{n}_{sh}"] = predict_mean(M[k][n], d)
                on[(panel, t)] = rec
        print(panel, flush=True)
    pickle.dump({"on": on}, open(CACHE / "onset_thr2_oof.pkl", "wb"))


if __name__ == "__main__":
    {"feats": feats, "fit": fit, "oof": oof}[sys.argv[1]]()
