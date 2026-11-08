"""More out-of-fold ongoing probabilities on the cand_eval windows: onphys (fold models ongoing_onphys_<k>) at
shifts 0, +-30, +-45, +-60 min, and v1 / aug at +-30 min (for test-time averaging over time-of-day lookups).
Online features stay at the true T. Writes cache/cand_eval4.pkl.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb

import features as FC
import features_v1 as F1
from build import predict_mean
from build_dataset import keep_mask as keep_c
from build_dataset_v1 import keep_mask as keep_v1
from common import CACHE, H, PANELS, STEP, T0, Panel, select_windows

MD = Path(__file__).resolve().parent / "models" / "cv"


def to_arr(d, p, L):
    a = np.zeros((H, L), np.float32); a[d.h.to_numpy() - 1, d.link.to_numpy()] = p
    return a


def main() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    M = {k: {n: [lgb.Booster(model_file=str(MD / f"ongoing_{n}_{k}.txt"))] for n in ("onphys", "aug", "v1")} for k in F1.FOLDS}
    go = {}
    for panel in PANELS:
        st1 = F1.Static(panel); stc = FC.Static(panel)
        P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        others = {k: [m for f, ms in F1.FOLDS.items() if f != k for m in ms] for k in F1.FOLDS}
        c1 = {k: F1.Clim(P, qc, others[k], onset_t) for k in F1.FOLDS}
        cc = {k: FC.Clim(P, qc, others[k], onset_t) for k in F1.FOLDS}
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
                for sh in (0, 6, -6, 9, -9, 12, -12):
                    dc = FC.ongoing_rows(FC.link_features(arr, stc, cc[k], t + sh, online), cc[k], t + sh)
                    kc = keep_c(dc) == 0
                    p = predict_mean(M[k]["onphys"], dc); p[kc] = 0
                    rec[f"onphys_{sh}"] = to_arr(dc, p, stc.L)
                    if sh in (6, -6):
                        p = predict_mean(M[k]["aug"], dc); p[kc] = 0
                        rec[f"aug_{sh}"] = to_arr(dc, p, stc.L)
                        d1 = F1.horizon_rows(F1.link_features(arr, st1, c1[k], t + sh), c1[k], t + sh)
                        p = predict_mean(M[k]["v1"], d1); p[keep_v1(d1) == 0] = 0
                        rec[f"v1_{sh}"] = to_arr(d1, p, st1.L)
                go[(panel, t)] = rec
        print(panel, flush=True)
    pickle.dump({"go": go}, open(CACHE / "cand_eval4.pkl", "wb"))


if __name__ == "__main__":
    main()
