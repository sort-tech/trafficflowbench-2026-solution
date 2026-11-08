"""B7 (c) control: onset model on the v1_thr table without any time-of-day / climatology feature
(tod, clim_*, onset_prior, onset_rate, clim_h, clim_dh dropped; dow, state and threshold features kept).

  v1_thr_notod.py fit    fold models models/cv/onset_v1_thr_notod_<k>.txt, full models models/v1_thr_notod/onset_s{0,1,2}.txt
  v1_thr_notod.py eval   out-of-fold i5 onset rule with v1_thr replaced, or the whole onset replaced, by this model
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
from build import MODELS, predict_mean
from build_dataset_v1 import onset_rows
from common import CACHE, H, PANELS, STEP, T0, Panel, select_windows
from model import NON_FEATURES, SEED, TOD_FEATURES, best_k_set, fit_dataset, iou, make_dataset
from thr_feats import THR_COLS, window_features

OUT = CACHE / "v1_thr"


def feats(cols):
    return [c for c in cols if c not in NON_FEATURES and c not in TOD_FEATURES]


def fit() -> None:
    (MODELS / "v1_thr_notod").mkdir(exist_ok=True)
    for fold in (None, "A", "B", "C"):
        paths = [MODELS / "v1_thr_notod" / f"onset_s{i}.txt" for i in range(3)] if fold is None else [MODELS / "cv" / f"onset_v1_thr_notod_{fold}.txt"]
        if all(p.exists() for p in paths):
            continue
        df = pd.concat([pd.read_parquet(OUT / f"ds_{p}.parquet", filters=[("vis", "=", 1)] + ([("fold", "!=", fold)] if fold else []))
                        for p in PANELS], ignore_index=True)
        ds = make_dataset(df, feats(df.columns)); del df
        for i, p in enumerate(paths):
            fit_dataset(ds, 300, {"seed": SEED + 1000 * i}).save_model(str(p))
        print("fit", fold, flush=True)


def evaluate() -> None:
    from mix_eval5 import load
    G, kt = load()
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    thr_fold = pickle.load(open(OUT / "thresholds.pkl", "rb"))
    M = {k: [lgb.Booster(model_file=str(MODELS / "cv" / f"onset_v1_thr_notod_{k}.txt"))] for k in F1.FOLDS}
    pn = {}
    for panel in PANELS:
        st = F1.Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        clims = {k: F1.Clim(P, qc, [m for f, ms in F1.FOLDS.items() if f != k for m in ms], onset_t) for k in F1.FOLDS}
        for (p_, t), r in G.on.items():
            if p_ != panel:
                continue
            k = m2f[r["month"]]
            rng = np.random.default_rng([PANELS.index(panel), t])
            arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
            d = onset_rows(F1.link_features(arr, st, clims[k], t))
            tf = window_features(arr, st.lanes, thr_fold[panel][k])
            for c in THR_COLS:
                d[c] = tf[c].to_numpy()
            pn[(p_, t)] = predict_mean(M[k], d)
        print(panel, flush=True)
    pickle.dump(pn, open(CACHE / "v1_thr_notod_oof.pkl", "wb"))
    W5 = {"v1": .15, "phys": .15, "aug": .35, "onphys": .35}
    from mix_eval import combine
    go = np.array([iou(combine({v: G.go[k][f"{v}_0"] for v in W5}, W5) >= (0.31 if 1 <= kt[i] <= 5 else 0.37), G.go[k]["y"], G.go[k]["e"])
                   for i, k in enumerate(G.kg)])

    def onset(rule):
        out = []
        for k in G.ko:
            r = G.on[k]; pr = np.zeros(r["y"].shape, bool); pr[H - 1] = best_k_set(rule(r, pn[k])); out.append(iou(pr, r["y"], r["e"]))
        return np.array(out)
    base = onset(lambda r, q: .5 * r["v1_0"] + .5 * r["v1_thr_0"])
    S0 = G.S(base, go)
    for name, rule in (("v1 .5 + v1_thr_notod .5", lambda r, q: .5 * r["v1_0"] + .5 * q),
                       ("v1 .5 + v1_thr .25 + v1_thr_notod .25", lambda r, q: .5 * r["v1_0"] + .25 * r["v1_thr_0"] + .25 * q),
                       ("v1_thr_notod alone", lambda r, q: q)):
        a = onset(rule); d = a - base
        print(f"{name}: dS_q {G.S(a, go) - S0:+.4f}, onset IoU {d.mean():+.4f} +- {d.std() / np.sqrt(len(d)):.4f} (better {int((d > 0).sum())} worse {int((d < 0).sum())})")


if __name__ == "__main__":
    {"fit": fit, "eval": evaluate}[sys.argv[1]]()
