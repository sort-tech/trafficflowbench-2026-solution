"""Step 3: onset models on the frozen lgb_v1 features with time-of-day jitter (no online features, no v2 changes).

Table (cache/v1_jit/ds_<PANEL>.parquet, onset windows only, same windows/masking/folds as build_dataset_v1.py):
  jit 0  the window as is
  jit 1, 2  the same state features, time-of-day lookups (tod, clim_*, onset_prior, onset_rate) taken at
            T + s, s ~ U{-12..12} steps (+-60 min) within the same day; label unchanged
  plus, for the phys variant only: onset_static (weekday/weekend onset propensity per link from the other
  folds' onset windows) and cap_margin_dmin (min 1 - flow/capacity over the 6 links from l downstream, at T).
Models: v1_jit (v1 features) and v1_phys_jit (+ 2 features); 300 rounds; full models with seeds SEED,
SEED+1000, SEED+2000 in models/v1_jit/, models/v1_phys_jit/; fold models (seed SEED) in
models/cv/onset_<name>_<k>.txt.

Usage: v1_jit.py build | fit | oof
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
from model import NON_FEATURES, SEED, fit_dataset, make_dataset

OUT = CACHE / "v1_jit"
EXTRA = ["onset_static", "cap_margin_dmin"]
SHIFTS = (0, 9, -9, 12, -12)


def onset_static(P: Panel, qc: np.ndarray, onset_t: np.ndarray, months: list[str]) -> np.ndarray:
    """[2, L] weekday/weekend onset propensity per link from onset windows in `months` (as features.Clim)."""
    mon = pd.DatetimeIndex(T0 + onset_t * STEP).strftime("%Y-%m").to_numpy()
    o = onset_t[np.isin(mon, months)]
    cnt = np.zeros((2, P.n_l)); nd = np.zeros(2)
    for t in o:
        d = int(F1.dow_of(t) >= 5)
        cnt[d] += np.where(P.visible[t + H], P.q[t + H], np.where(P.visible[t + H + 1], P.q[t + H + 1], qc[t + H]))
        nd[d] += 1
    return ((cnt + 0.5 * cnt.sum(0) / max(nd.sum(), 1)) / (nd[:, None] + 0.5)).astype(np.float32)


def add_extra(df: pd.DataFrame, ost: np.ndarray, t_idx: int) -> pd.DataFrame:
    L = len(df)
    df["onset_static"] = ost[int(F1.dow_of(t_idx) >= 5)]
    m = 1.0 - df["f_0"].to_numpy()
    df["cap_margin_dmin"] = np.array([m[l:min(l + 6, L)].min() for l in range(L)], np.float32)
    return df


def window_rows(arr, st, clim, t: int, s: int, ost) -> pd.DataFrame:
    """v1 onset rows with time-of-day lookups at t + s (state features from the window at t)."""
    df = onset_rows(F1.link_features(arr, st, clim, t + s))
    return add_extra(df, ost, t)


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    W0 = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    month2fold = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    for panel in PANELS:
        P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy"); st = F1.Static(panel)
        W = W0[W0.panel == panel]
        onset_t = W[W.condition == "queue_onset"].t.to_numpy()
        Won = W[W.condition == "queue_onset"].sort_values("t")
        out = []
        for fold, months in F1.FOLDS.items():
            other = [m for f, ms in F1.FOLDS.items() if f != fold for m in ms]
            clim = F1.Clim(P, qc, other, onset_t)
            ost = onset_static(P, qc, onset_t, other)
            for w in Won[Won.month.map(month2fold) == fold].itertuples(index=False):
                t = int(w.t)
                rng = np.random.default_rng([PANELS.index(panel), t])
                arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                jr = np.random.default_rng([PANELS.index(panel), t, 13])
                tod = t % 288
                for j in range(3):
                    s = 0 if j == 0 else int(jr.integers(max(-12, -tod), min(12, 287 - tod) + 1))
                    df = window_rows(arr, st, clim, t, s, ost)
                    li = np.arange(st.L)
                    df["y"] = P.q[t + H, li].astype(np.int8)
                    df["vis"] = P.visible[t + H, li].astype(np.int8)
                    df["t"] = np.int32(t); df["jit"] = np.int8(j); df["fold"] = fold
                    out.append(df)
            print(panel, fold, flush=True)
        res = pd.concat(out, ignore_index=True)
        for c in res.columns:
            if res[c].dtype == np.float64:
                res[c] = res[c].astype(np.float32)
        res.to_parquet(OUT / f"ds_{panel}.parquet", index=False)
        print(panel, "rows", len(res), flush=True)


def feats(name: str, cols: list[str]) -> list[str]:
    f = [c for c in cols if c not in NON_FEATURES | {"jit"} | set(EXTRA)]
    return f + EXTRA if name == "v1_phys_jit" else f


def fit() -> None:
    for name in ("v1_jit", "v1_phys_jit"):
        (MODELS / name).mkdir(exist_ok=True)
        for fold in (None, "A", "B", "C"):
            paths = [MODELS / name / f"onset_s{i}.txt" for i in range(3)] if fold is None else [MODELS / "cv" / f"onset_{name}_{fold}.txt"]
            if all(p.exists() for p in paths):
                continue
            filters = [("vis", "=", 1)] + ([("fold", "!=", fold)] if fold else [])
            df = pd.concat([pd.read_parquet(OUT / f"ds_{p}.parquet", filters=filters) for p in PANELS], ignore_index=True)
            fs = feats(name, list(df.columns))
            ds = make_dataset(df, fs)
            del df
            for i, p in enumerate(paths):
                fit_dataset(ds, 300, {"seed": SEED + 1000 * i}).save_model(str(p))
            print("fit", name, fold, flush=True)


def oof() -> None:
    """Out-of-fold onset probabilities of both models on the cand_eval onset windows (shifts) -> cache/v1_jit_oof.pkl."""
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    M = {k: {n: [lgb.Booster(model_file=str(MODELS / "cv" / f"onset_{n}_{k}.txt"))] for n in ("v1_jit", "v1_phys_jit")} for k in F1.FOLDS}
    on = {}
    for panel in PANELS:
        st = F1.Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        others = {k: [m for f, ms in F1.FOLDS.items() if f != k for m in ms] for k in F1.FOLDS}
        c1 = {k: F1.Clim(P, qc, others[k], onset_t) for k in F1.FOLDS}
        osts = {k: onset_static(P, qc, onset_t, others[k]) for k in F1.FOLDS}
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
                rec = {}
                for sh in SHIFTS:
                    df = window_rows(arr, st, c1[k], t, sh, osts[k])
                    for n in ("v1_jit", "v1_phys_jit"):
                        rec[f"{n}_{sh}"] = predict_mean(M[k][n], df)
                on[(panel, t)] = rec
        print(panel, flush=True)
    pickle.dump({"on": on}, open(CACHE / "v1_jit_oof.pkl", "wb"))


if __name__ == "__main__":
    {"build": build, "fit": fit, "oof": oof}[sys.argv[1]]()
