"""LightGBM cell models for Task 2 and the expected-IoU decision rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

# time-of-day dependent features (climatology, priors); dropped by the "notod" variant
TOD_FEATURES = {"tod", "clim_0", "clim_any0", "clim_3", "clim_6", "clim_m6", "onset_prior", "onset_rate", "clim_h",
                "clim_dh", "p11_h", "p01_h", "tr_h", "f0_dev", "fdev_mean"}
ONLINE_FEATURES = {"on_ndays", "oclim_0", "oclim_6", "oclim_12", "oclim_any_0", "oclim_any_6", "oclim_any_12",
                   "ofirst_dist", "ofirst_frac", "today_q", "today_since", "today_nq",
                   "oclim6_o-3", "oclim6_o-1", "oclim6_o1", "oclim6_o3",
                   "today_q_o-3", "today_q_o-1", "today_q_o1", "today_q_o3"}
PHYS_FEATURES = {"rh_w", "rh_tail_rel", "rh_head_rel", "w_cong", "cap_margin", "cap_margin_dmin", "rh_ext",
                 "onset_static", "onset_static_all"}
AUG_FEATURES = {"aug_" + c for c in ["tod", "clim_0", "clim_any0", "clim_3", "clim_6", "clim_m6", "onset_prior",
                                     "onset_rate", "f0_dev", "fdev_mean", "clim_h", "clim_dh", "p11_h", "p01_h", "tr_h"]}
NEW = PHYS_FEATURES | AUG_FEATURES
# full: train-seed climatology (lgb_v2); notod: neither; online: same-split pre-T statistics instead of climatology;
# phys: notod + physics/static propensity; onphys: online + physics; aug: online + physics + time-shift-augmented climatology
VARIANTS = {"full": ONLINE_FEATURES | NEW, "notod": TOD_FEATURES | ONLINE_FEATURES | NEW,
            "online": TOD_FEATURES | NEW, "phys": TOD_FEATURES | ONLINE_FEATURES | AUG_FEATURES,
            "onphys": TOD_FEATURES | AUG_FEATURES, "aug": TOD_FEATURES}
NON_FEATURES = {"y", "vis", "elig", "yc", "keep", "t", "cond", "greedy", "fold", "link", "panel", "window_id"}
SEED = 20261025

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 3,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def make_dataset(df: pd.DataFrame, feats: list[str]) -> lgb.Dataset:
    ds = lgb.Dataset(df[feats], label=df["y"].to_numpy(), categorical_feature=["pid"],
                     params=dict(PARAMS), free_raw_data=True)
    return ds.construct()


def fit_dataset(ds: lgb.Dataset, rounds: int, params: dict | None = None) -> lgb.Booster:
    return lgb.train(dict(PARAMS, **(params or {})), ds, num_boost_round=rounds)


def fit(df: pd.DataFrame, feats: list[str], rounds: int, params: dict | None = None) -> lgb.Booster:
    p = dict(PARAMS, **(params or {}))
    ds = lgb.Dataset(df[feats], label=df["y"].to_numpy(), categorical_feature=["pid"], free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=rounds)


def best_k_set(p: np.ndarray, max_k: int | None = None, min_gain: float = 0.0) -> np.ndarray:
    """Choose the top-k cells maximizing sum_{i<=k} p_i / (k + sum_{i>k} p_i).

    Ratio-of-expectations approximation to the expected IoU. Returns a bool mask.
    """
    n = len(p)
    out = np.zeros(n, bool)
    if n == 0:
        return out
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    cum = np.cumsum(ps)
    total = cum[-1]
    k = np.arange(1, n + 1)
    score = cum / (k + (total - cum))
    if max_k is not None:
        score = score[:max_k]
    kb = int(np.argmax(score)) + 1
    if score[kb - 1] <= min_gain:
        return out
    out[order[:kb]] = True
    return out


def iou(pred: np.ndarray, y: np.ndarray, elig: np.ndarray) -> float:
    e = elig.astype(bool)
    if not e.any():
        return 0.0
    pr = pred.astype(bool) & e
    t = y.astype(bool) & e
    u = (pr | t).sum()
    return 1.0 if u == 0 else float((pr & t).sum() / u)
