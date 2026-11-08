"""Causal restorer for the empty cells of the released row at T.

The old fill copies T-5 into an empty T cell (ffill) and only interpolates in space when a whole row is
empty. Here a small LightGBM regressor per quantity (speed / v_cut, flow / capacity, occupancy) predicts the
empty T cell from inputs at or before T only:
  - own link at T-5, T-10, T-15, T-30, T-60
  - links +-1..3 at T (observed or empty) with their distance, and their T-5 value
  - nearest observed link upstream / downstream at T and its distance
Training: random train origins (half pseudo-window origins, half 04:00-22:00 uniform); T cells with
pct_observed >= 75 are hidden at 20/30/50% like train_window_arrays, and the hidden observation is the target.
Fold models (trained without one 3-month fold) are used for that fold's pseudo windows; the "all" model for
validation/private.

Usage: restore_t.py fit     (models/restore_t/<target>_<fold>.txt)
       restore_t.py check   (held-out error and queue agreement vs T-5 copy + spatial fill)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import CACHE, NHIST, PANELS, STEP, T0, Panel

MD = HERE / "models" / "restore_t"
FOLDS = {"A": ["2030-06", "2030-07", "2030-08"], "B": ["2030-09", "2030-10", "2030-11"],
         "C": ["2030-12", "2031-01", "2031-02"]}
TARGETS = ("r", "f", "o")
LAGS = (1, 2, 3, 6, 12)            # T-5 .. T-60 (rows NHIST - k)
NB = (-3, -2, -1, 1, 2, 3)
MASK_RATES = [0.2, 0.3, 0.5]
N_PER_PANEL = 4000
PARAMS = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "num_threads": 3, "seed": 20261027, "deterministic": True, "force_row_wise": True, "verbose": -1}
ROUNDS = 300
DEFAULT = {"r": 1.67, "f": 0.5, "o": 0.05}


def scaled(arr: dict, vcut: np.ndarray, cap: np.ndarray) -> dict:
    return {"r": arr["speed"] / vcut, "f": arr["flow"] / cap, "o": arr["occ"].astype(np.float32)}


def nearest_obs(v: np.ndarray, pos: np.ndarray):
    """For each link: nearest observed value upstream (lower index) and downstream at the same row."""
    L = len(v)
    idx = np.arange(L)
    obs = ~np.isnan(v)
    up = np.where(obs, idx, -1)
    up = np.maximum.accumulate(np.concatenate([[-1], up[:-1]]))           # strictly upstream
    dn = np.where(obs, idx, L)
    dn = np.minimum.accumulate(np.concatenate([dn[1:], [L]])[::-1])[::-1]  # strictly downstream
    vu = np.where(up >= 0, v[np.clip(up, 0, L - 1)], np.nan)
    vd = np.where(dn < L, v[np.clip(dn, 0, L - 1)], np.nan)
    du = np.where(up >= 0, pos - pos[np.clip(up, 0, L - 1)], np.nan)
    dd = np.where(dn < L, pos[np.clip(dn, 0, L - 1)] - pos, np.nan)
    return vu, du, vd, dd


def features(x: dict, pos: np.ndarray, vcut: np.ndarray, lanes: np.ndarray, pid: int) -> pd.DataFrame:
    """x: scaled r/f/o arrays [13, L] (row NHIST = T, may hold NaN). One row per link."""
    L = len(pos)
    F = {"pid": np.full(L, pid, np.int16), "vcut": vcut, "lanes": lanes}
    for q in TARGETS:
        a = x[q]
        for k in LAGS:
            F[f"{q}_m{k}"] = a[NHIST - k]
        for j in NB:
            ok = (np.arange(L) + j >= 0) & (np.arange(L) + j < L)
            idx = np.clip(np.arange(L) + j, 0, L - 1)
            F[f"{q}T_o{j}"] = np.where(ok, a[NHIST, idx], np.nan)
            F[f"{q}1_o{j}"] = np.where(ok, a[NHIST - 1, idx], np.nan)
        vu, du, vd, dd = nearest_obs(a[NHIST], pos)
        F[f"{q}T_up"], F[f"{q}T_dn"] = vu, vd
        if q == "r":
            F["d_up"], F["d_dn"] = du, dd
    for j in NB:
        ok = (np.arange(L) + j >= 0) & (np.arange(L) + j < L)
        idx = np.clip(np.arange(L) + j, 0, L - 1)
        F[f"dist_o{j}"] = np.where(ok, pos[idx] - pos, np.nan)
    F["miss_frac"] = np.full(L, float(np.isnan(x["r"][NHIST]).mean()), np.float32)
    df = pd.DataFrame(F)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


class Restorer:
    """Fills the empty T-row cells of a window array with the fold's (or the all-train) models."""

    def __init__(self, fold: str = "all"):
        self.m = {q: lgb.Booster(model_file=str(MD / f"{q}_{fold}.txt")) for q in TARGETS}

    def fill(self, arr: dict, st) -> dict:
        """Returns speed/flow/occ [L] estimates (only used where the T cell is empty)."""
        x = scaled(arr, st.vcut, st.cap)
        df = features(x, st.pos_km, st.vcut, st.lanes, st.pid)
        out = {}
        for q in TARGETS:
            p = self.m[q].predict(df[self.m[q].feature_name()], num_threads=1)
            out[q] = p.astype(np.float32)
        return {"speed": out["r"] * st.vcut, "flow": out["f"] * st.cap, "occ": out["o"]}


# ---------------------------------------------------------------- training data

def sample_times(P: Panel, panel: str, rng: np.random.Generator) -> np.ndarray:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet", columns=["panel", "t"])
    wt = W[W.panel == panel].t.to_numpy()
    a = rng.choice(wt, min(N_PER_PANEL // 2, len(wt)), replace=False)
    days = rng.integers(0, P.n_t // 288, N_PER_PANEL - len(a))
    b = days * 288 + rng.integers(48, 264, len(days))
    t = np.concatenate([a, b])
    return np.sort(t[(t >= NHIST) & (t < P.n_t)])


def window_samples(P: Panel, st, t: int, rng: np.random.Generator):
    """Hide part of the T row like train_window_arrays; return features + targets of hidden cells."""
    sl = slice(t - NHIST, t + 1)
    arr = {"speed": P.speed[sl].copy(), "flow": P.flow[sl].copy(), "occ": P.occ[sl].copy()}
    rate = MASK_RATES[rng.integers(3)]
    m = (P.pct[t] >= 75) & (rng.random(P.n_l) < rate) & ~np.isnan(P.speed[t])
    if not m.any():
        return None
    truth = scaled({k: v[-1:] for k, v in arr.items()}, st.vcut, st.cap)
    for k in arr:
        arr[k][-1, m] = np.nan
    x = scaled(arr, st.vcut, st.cap)
    df = features(x, st.pos_km, st.vcut, st.lanes, st.pid)
    df["y_r"], df["y_f"], df["y_o"] = truth["r"][0], truth["f"][0], truth["o"][0]
    # baseline: forward fill in time (T-5, then earlier), spatial interpolation if still empty
    base = {}
    for q in TARGETS:
        a = x[q]
        v = a[NHIST].copy()
        for k in range(1, NHIST + 1):
            e = np.isnan(v)
            v[e] = a[NHIST - k, e]
        if np.isnan(v).any():
            e = np.isnan(v)
            if e.all():
                v[:] = DEFAULT[q]
            else:
                xi = np.arange(len(v))
                v[e] = np.interp(xi[e], xi[~e], v[~e])
        base[q] = v
    df["b_r"], df["b_f"], df["b_o"] = base["r"], base["f"], base["o"]
    df["t"] = np.int32(t)
    df["panel"] = P.panel
    df["link"] = np.arange(P.n_l, dtype=np.int16)
    return df[m]


def build_samples() -> pd.DataFrame:
    from features import Static
    path = CACHE / "restore_t_samples.parquet"
    if path.exists():
        return pd.read_parquet(path)
    parts = []
    for panel in PANELS:
        P = Panel(panel)
        st = Static(panel)
        rng = np.random.default_rng([PANELS.index(panel), 27])
        for t in sample_times(P, panel, rng):
            d = window_samples(P, st, int(t), np.random.default_rng([PANELS.index(panel), int(t), 27]))
            if d is not None:
                parts.append(d)
        print(panel, sum(len(p) for p in parts), flush=True)
        del P
    df = pd.concat(parts, ignore_index=True)
    mon = pd.DatetimeIndex(T0 + df.t.to_numpy() * STEP).strftime("%Y-%m")
    df["fold"] = pd.Series(mon).map({m: f for f, ms in FOLDS.items() for m in ms}).to_numpy()
    df.to_parquet(path, index=False)
    return df


FEATS_EXCLUDE = {"y_r", "y_f", "y_o", "b_r", "b_f", "b_o", "t", "panel", "link", "fold"}


def fit_all() -> None:
    MD.mkdir(parents=True, exist_ok=True)
    df = build_samples()
    feats = [c for c in df.columns if c not in FEATS_EXCLUDE]
    for fold in list(FOLDS) + ["all"]:
        tr = df if fold == "all" else df[df.fold != fold]
        for q in TARGETS:
            path = MD / f"{q}_{fold}.txt"
            if path.exists():
                continue
            ok = tr[f"y_{q}"].notna().to_numpy()
            ds = lgb.Dataset(tr.loc[ok, feats], label=tr.loc[ok, f"y_{q}"].to_numpy(), categorical_feature=["pid"],
                             params=dict(PARAMS))
            m = lgb.train(PARAMS, ds, num_boost_round=ROUNDS)
            m.save_model(str(path))
            print("fit", q, fold, int(ok.sum()), flush=True)


def check() -> None:
    df = build_samples()
    feats = [c for c in df.columns if c not in FEATS_EXCLUDE]
    res = []
    for fold in FOLDS:
        te = df[df.fold == fold].copy()
        for q in TARGETS:
            m = lgb.Booster(model_file=str(MD / f"{q}_{fold}.txt"))
            te[f"p_{q}"] = m.predict(te[feats], num_threads=3)
        res.append(te)
    te = pd.concat(res, ignore_index=True)
    rows = []
    for name, g in [("all", te), ("window origins", te[te.t.isin(pd.read_parquet(CACHE / "pseudo_windows.parquet").t)])]:
        for q in TARGETS:
            ok = g[f"y_{q}"].notna()
            y = g.loc[ok, f"y_{q}"]
            rows.append({"set": name, "target": q, "n": int(ok.sum()),
                         "mae_copy": float((g.loc[ok, f"b_{q}"] - y).abs().mean()),
                         "mae_model": float((g.loc[ok, f"p_{q}"] - y).abs().mean())})
        yq = g.y_r <= 1.0
        for src in ("b", "p"):
            pq_ = g[f"{src}_r"] <= 1.0
            inter = (yq & pq_).sum(); union = (yq | pq_).sum()
            rows.append({"set": name, "target": f"queue[{src}]", "n": int(yq.sum()),
                         "acc": float((yq == pq_).mean()), "iou": float(inter / max(union, 1)),
                         "fp": int((pq_ & ~yq).sum()), "fn": int((yq & ~pq_).sum())})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    out.to_csv(CACHE / "restore_t_check.csv", index=False)


if __name__ == "__main__":
    {"fit": fit_all, "check": check}[sys.argv[1]]()
