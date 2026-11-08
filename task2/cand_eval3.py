"""Out-of-fold aug probabilities (onset and ongoing, fold models models/cv/<cond>_aug_<k>.txt) on the same 719
windows as cand_eval.py / cand_eval2.py, with time-of-day lookups shifted by 0, +-45, +-60 min (online features
at the true T; the aug_* columns equal the shifted lookups, as at inference). Writes cache/cand_eval3.pkl.
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
from common import CACHE, H, PANELS, STEP, T0, Panel, select_windows

MD = Path(__file__).resolve().parent / "models" / "cv"
SHIFTS = (0, 9, -9, 12, -12)


def main() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    M = {k: {n: [lgb.Booster(model_file=str(MD / f"{n}_aug_{k}.txt"))] for n in ("onset", "ongoing")} for k in F1.FOLDS}
    on, go = {}, {}
    for panel in PANELS:
        stc = FC.Static(panel)
        P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        cc = {k: FC.Clim(P, qc, [m for f, ms in F1.FOLDS.items() if f != k for m in ms], onset_t) for k in F1.FOLDS}
        for mon in months:
            start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
            end = int((pd.Timestamp(f"{mon}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            sel = select_windows(P, qc, max_per_condition=5, start=max(start, 12), end=end)
            k = m2f[mon]
            for w in sel.itertuples():
                t = int(w.t)
                rng = np.random.default_rng([PANELS.index(panel), t])
                arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                online = FC.online_features(P.speed, P.vcut, t, start)
                rec = {}
                for sh in SHIFTS:
                    df = FC.link_features(arr, stc, cc[k], t + sh, online)
                    if w.condition == "queue_onset":
                        rec[f"aug_{sh}"] = predict_mean(M[k]["onset"], FC.onset_rows(df, cc[k], t + sh))
                    else:
                        dc = FC.ongoing_rows(df, cc[k], t + sh)
                        p = predict_mean(M[k]["ongoing"], dc); p[keep_c(dc) == 0] = 0
                        a = np.zeros((H, stc.L), np.float32); a[dc.h.to_numpy() - 1, dc.link.to_numpy()] = p
                        rec[f"aug_{sh}"] = a
                (on if w.condition == "queue_onset" else go)[(panel, t)] = rec
        print(panel, flush=True)
    pickle.dump({"on": on, "go": go}, open(CACHE / "cand_eval3.pkl", "wb"))


if __name__ == "__main__":
    main()
