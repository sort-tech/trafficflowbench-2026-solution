"""(7) In-split online threshold adaptation for v1_thr, out-of-fold check only.

For each train day and threshold group (fold thresholds), the per-lane flow at the reference link just before the
day's first onset in the group (3-slot mean, events as in thr_feats.fit_thresholds). A window on day D uses only the
complete earlier days of its own month: c_adap = n / (n + 3) * median(those values) + 3 / (n + 3) * c_train.
Table cache/v1_thr_on/ds_<PANEL>.parquet (v1 onset rows + THR_COLS with c_adap), fold models
models/cv/onset_v1_thron_<k>.txt, OOF cache/v1_thron_oof.pkl (cand_eval onset windows, shifts 0, +-45, +-60 min).

Usage: thr_online.py build | fit | oof
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.ndimage import label

import features_v1 as F1
from build import MODELS, predict_mean
from build_dataset_v1 import onset_rows
from common import CACHE, H, PANELS, STEP, T0, Panel, select_windows
from model import NON_FEATURES, SEED, fit_dataset, make_dataset
from thr_feats import THR_COLS, _interp_day, window_features

OUT = CACHE / "v1_thr_on"
SHIFTS = (0, 9, -9, 12, -12)
K0 = 3


def daily_obs(P, thr) -> np.ndarray:
    """[n_days, n_groups] per-lane flow at the group's reference link just before its first onset that day (NaN if none)."""
    return daily_obs_arrays(P.speed, P.flow, P.lanes, P.free, thr)


def daily_obs_arrays(speed, flow, lanes, free, thr, n_days: int | None = None) -> np.ndarray:
    vf = free.astype(np.float32); lanes = np.maximum(lanes, 1).astype(np.float32)
    nd = speed.shape[0] // 288 if n_days is None else n_days
    out = np.full((speed.shape[0] // 288, len(thr)), np.nan)
    for d in range(nd):
        sl = slice(d * 288, (d + 1) * 288)
        if not np.isfinite(speed[sl]).any():
            continue
        v = _interp_day(speed[sl])
        lab, n = label(v <= 0.6 * vf, structure=np.ones((3, 3)))
        starts = []
        for e in range(1, n + 1):
            ts, ls = np.nonzero(lab == e)
            if len(ts) >= 6:
                t0 = ts.min(); starts.append((t0, ls[ts == t0].max()))
        if not starts:
            continue
        q = _interp_day(flow[sl] / lanes)
        for g, (a, b, h, c) in enumerate(thr):
            s = [t0 for t0, hd in starts if a <= hd <= b and t0 > 2]
            if s:
                t0 = min(s)
                out[d, g] = q[t0 - 3:t0, h].mean()
    return out


def adapted(thr, obs: np.ndarray, day: int, month_start_day: int):
    prev = obs[month_start_day:day]                     # complete earlier days of the same month only
    out = []
    for g, (a, b, h, c) in enumerate(thr):
        v = prev[:, g][np.isfinite(prev[:, g])]
        n = len(v)
        out.append((a, b, h, (n * float(np.median(v)) + K0 * c) / (n + K0) if n else c))
    return out


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    thr_fold = pickle.load(open(CACHE / "v1_thr" / "thresholds.pkl", "rb"))
    W0 = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    obs_log = {}
    for panel in PANELS:
        P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy"); st = F1.Static(panel)
        W = W0[W0.panel == panel]
        onset_t = W[W.condition == "queue_onset"].t.to_numpy()
        Won = W[W.condition == "queue_onset"].sort_values("t")
        obs = {k: daily_obs(P, thr_fold[panel][k]) for k in F1.FOLDS}
        obs_log[panel] = obs
        out = []
        for fold in F1.FOLDS:
            clim = F1.Clim(P, qc, [m for f, ms in F1.FOLDS.items() if f != fold for m in ms], onset_t)
            for w in Won[Won.month.map(m2f) == fold].itertuples(index=False):
                t = int(w.t)
                rng = np.random.default_rng([PANELS.index(panel), t])
                arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                ms = int((pd.Timestamp(f"{w.month}-01T00:00:00Z") - T0) / STEP) // 288
                thr = adapted(thr_fold[panel][fold], obs[fold], t // 288, ms)
                df = onset_rows(F1.link_features(arr, st, clim, t))
                tf = window_features(arr, st.lanes, thr)
                for c in THR_COLS:
                    df[c] = tf[c].to_numpy()
                li = np.arange(st.L)
                df["y"] = P.q[t + H, li].astype(np.int8); df["vis"] = P.visible[t + H, li].astype(np.int8)
                df["t"] = np.int32(t); df["fold"] = fold
                out.append(df)
        res = pd.concat(out, ignore_index=True)
        for c in res.columns:
            if res[c].dtype == np.float64:
                res[c] = res[c].astype(np.float32)
        res.to_parquet(OUT / f"ds_{panel}.parquet", index=False)
        print(panel, len(res), flush=True)
    pickle.dump(obs_log, open(OUT / "daily_obs.pkl", "wb"))


def fit(full: bool = False) -> None:
    """Fold models (seed SEED); with full, also models/v1_thron/onset_s{0,1,2}.txt (all folds, seeds SEED + 1000 i)."""
    (MODELS / "v1_thron").mkdir(exist_ok=True)
    for fold in (("A", "B", "C") + ((None,) if full else ())):
        paths = [MODELS / "v1_thron" / f"onset_s{i}.txt" for i in range(3)] if fold is None else [MODELS / "cv" / f"onset_v1_thron_{fold}.txt"]
        if all(p.exists() for p in paths):
            continue
        filters = [("vis", "=", 1)] + ([("fold", "!=", fold)] if fold else [])
        df = pd.concat([pd.read_parquet(OUT / f"ds_{p}.parquet", filters=filters) for p in PANELS], ignore_index=True)
        ds = make_dataset(df, [c for c in df.columns if c not in NON_FEATURES]); del df
        for i, p in enumerate(paths):
            fit_dataset(ds, 300, {"seed": SEED + 1000 * i}).save_model(str(p))
        print("fit", fold, flush=True)


def release_obs(panel: str, split: str, st, thr) -> np.ndarray:
    """Daily pre-onset flows on a release split from its masked observations (mainline_states_masked).
    Callers use only the complete days before the window's day."""
    import pyarrow.dataset as pds
    from common import REL
    z = np.load(CACHE / f"{panel}_{split}.npz")
    speed = z["speed"]
    root = REL / "corridors" / panel / split / "mainline_states_masked"
    t = pds.dataset(str(root), format="parquet", partitioning="hive").to_table(columns=["timestamp", "link_id", "flow_vph"]).to_pandas()
    start = {"validation": pd.Timestamp("2031-03-01T00:00:00Z"), "private": pd.Timestamp("2031-04-01T00:00:00Z")}[split]
    ti = ((pd.to_datetime(t.timestamp, utc=True) - start) // STEP).to_numpy()
    pos = {k: i for i, k in enumerate(st.link_ids)}
    flow = np.full(speed.shape, np.nan, np.float32)
    ok = (ti >= 0) & (ti < speed.shape[0])
    flow[ti[ok], t.link_id.astype(str).map(pos).to_numpy()[ok]] = t.flow_vph.to_numpy(np.float32)[ok]
    free = (st.vcut / 0.6).astype(np.float32)
    return daily_obs_arrays(speed, flow, st.lanes, free, thr)


def oof() -> None:
    thr_fold = pickle.load(open(CACHE / "v1_thr" / "thresholds.pkl", "rb"))
    obs_log = pickle.load(open(OUT / "daily_obs.pkl", "rb"))
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    M = {k: [lgb.Booster(model_file=str(MODELS / "cv" / f"onset_v1_thron_{k}.txt"))] for k in F1.FOLDS}
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
                thr = adapted(thr_fold[panel][k], obs_log[panel][k], t // 288, start // 288)
                tf = window_features(arr, st.lanes, thr)
                rec = {}
                for sh in SHIFTS:
                    df = onset_rows(F1.link_features(arr, st, c1[k], t + sh))
                    for c in THR_COLS:
                        df[c] = tf[c].to_numpy()
                    rec[f"v1_thron_{sh}"] = predict_mean(M[k], df)
                on[(panel, t)] = rec
        print(panel, flush=True)
    pickle.dump({"on": on}, open(CACHE / "v1_thron_oof.pkl", "wb"))


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(len(sys.argv) > 2 and sys.argv[2] == "full")
    else:
        {"build": build, "oof": oof}[sys.argv[1]]()
