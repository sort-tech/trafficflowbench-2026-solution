"""Extra lgb_v1 seeds (20262025, 20263025) without touching models/v1: full models in models/v1_seeds/ and fold
models in models/cv/<cond>_v1s<i>_<k>.txt, then out-of-fold v1 probabilities averaged over the 3 seeds on the
cand_eval windows (shifts 0, +-45, +-60 min) -> cache/seeds_v1.pkl.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb

import features_v1 as F1
from build import MODELS, load_v1_dataset, predict_mean
from build_dataset_v1 import keep_mask, onset_rows, out_dir
from common import CACHE, H, PANELS, STEP, T0, Panel, select_windows
from model import fit_dataset

SEEDS = {1: 20262025, 2: 20263025}
MD = MODELS / "cv"
SD = MODELS / "v1_seeds"
SHIFTS = (0, 9, -9, 12, -12)


def fit_all() -> None:
    SD.mkdir(exist_ok=True)
    for name, cond in (("onset", 1), ("ongoing", 0)):
        for fold in [None, "A", "B", "C"]:
            paths = {i: (SD / f"{name}_s{i}.txt") if fold is None else (MD / f"{name}_v1s{i}_{fold}.txt") for i in SEEDS}
            if all(p.exists() for p in paths.values()):
                continue
            ds = load_v1_dataset(out_dir(False), cond, drop_fold=fold)
            for i, s in SEEDS.items():
                m = fit_dataset(ds, 300, {"seed": s})
                m.save_model(str(paths[i]))
            del ds
            print("fit", name, fold, flush=True)


def oof() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    M = {k: {n: [lgb.Booster(model_file=str(MD / f"{n}_v1_{k}.txt"))]
             + [lgb.Booster(model_file=str(MD / f"{n}_v1s{i}_{k}.txt")) for i in SEEDS] for n in ("onset", "ongoing")}
         for k in F1.FOLDS}
    on, go = {}, {}
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
                t = int(w.t)
                rng = np.random.default_rng([PANELS.index(panel), t])
                arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                rec = {}
                for sh in SHIFTS:
                    df = F1.link_features(arr, st, c1[k], t + sh)
                    if w.condition == "queue_onset":
                        rec[f"v1b_{sh}"] = predict_mean(M[k]["onset"], onset_rows(df))
                    else:
                        d = F1.horizon_rows(df, c1[k], t + sh)
                        p = predict_mean(M[k]["ongoing"], d); p[keep_mask(d) == 0] = 0
                        a = np.zeros((H, st.L), np.float32); a[d.h.to_numpy() - 1, d.link.to_numpy()] = p
                        rec[f"v1b_{sh}"] = a
                (on if w.condition == "queue_onset" else go)[(panel, t)] = rec
        print(panel, flush=True)
    pickle.dump({"on": on, "go": go}, open(CACHE / "seeds_v1.pkl", "wb"))


if __name__ == "__main__":
    fit_all()
    oof()
