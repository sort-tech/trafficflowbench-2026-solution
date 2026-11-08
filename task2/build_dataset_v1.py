"""Feature tables for the lgb_v1 recipe (frozen features_v1.py), optionally with the causal T-row restorer.

Same windows, folds, masking and seeds as the original lgb_v1 run (2026-10-25). Writes
cache/v1/ds_<PANEL>.parquet, or cache/v1_causal_t/ds_<PANEL>.parquet with --causal-t (each fold's windows are
filled by the restorer trained without that fold).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE, H, PANELS, Panel
from features_v1 import FOLDS, Clim, Static, horizon_rows, link_features, train_window_arrays

MASK_RATES = [0.2, 0.3, 0.5]


def out_dir(causal_t: bool) -> Path:
    return CACHE / ("v1_causal_t" if causal_t else "v1")


def keep_mask(df: pd.DataFrame) -> np.ndarray:
    """Input-only row filter: links far from any queue with negligible climatology are predicted 0."""
    near = np.minimum(df.dd_0.to_numpy(), df.du_0.to_numpy()) < 8.0
    clim = np.maximum(df.clim_0.to_numpy(), df.clim_6.to_numpy()) > 0.003
    onset = df.onset_prior.to_numpy() > 0.003
    return near | clim | onset


def onset_rows(df: pd.DataFrame) -> pd.DataFrame:
    df["h"] = np.int8(H)
    df["clim_h"] = df["clim_6"]
    df["clim_dh"] = df["clim_6"] - df["clim_0"]
    df["tail_ext30"] = df["tail_rel"]
    df["tail_ext15"] = df["tail_rel"]
    df["dd_ext"] = df["dd_0"]
    return df


def build(panel: str, max_ongoing_train: int, causal_t: bool, split_clim: float | None = None) -> None:
    P = Panel(panel)
    qc = np.load(CACHE / f"{panel}_qclean.npy")
    st = Static(panel)
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    W = W[W.panel == panel].copy()
    onset_t = W[W.condition == "queue_onset"].t.to_numpy()
    # thin non-greedy ongoing windows for training size
    go = W[(W.condition == "queue_ongoing") & ~W.greedy]
    if len(go) > max_ongoing_train:
        go = go.sample(max_ongoing_train, random_state=PANELS.index(panel))
    W = pd.concat([W[W.greedy | (W.condition == "queue_onset")], go]).drop_duplicates(["t", "condition"]).sort_values("t")
    month2fold = {m: f for f, ms in FOLDS.items() for m in ms}
    SC = None
    if split_clim is not None:
        from split_clim import TrainSplitClims
        SC = TrainSplitClims(P, qc, onset_t, "v1", split_clim)
    out = []
    for fold, months in FOLDS.items():
        other = [m for f, ms in FOLDS.items() if f != fold for m in ms]
        clim_fold = Clim(P, qc, other, onset_t)
        rest = None
        if causal_t:
            from restore_t import Restorer
            rest = Restorer(fold)
        Wf = W[W.month.map(month2fold) == fold]
        for w in Wf.itertuples(index=False):
            clim = clim_fold if SC is None else SC.blend(clim_fold, int(w.t))
            rng = np.random.default_rng([PANELS.index(panel), int(w.t)])
            arr = train_window_arrays(P, w.t, rng, MASK_RATES[rng.integers(3)])
            df = link_features(arr, st, clim, w.t, rest.fill(arr, st) if rest else None)
            if w.condition == "queue_onset":
                df = onset_rows(df)
                hs = np.full(st.L, H)
                li = np.arange(st.L)
            else:
                df = horizon_rows(df, clim, w.t)
                hs = df.h.to_numpy().astype(int)
                li = df.link.to_numpy().astype(int)
            tt = w.t + hs
            df["y"] = P.q[tt, li].astype(np.int8)
            df["vis"] = P.visible[tt, li].astype(np.int8)
            df["elig"] = P.elig[tt, li].astype(np.int8)
            df["yc"] = qc[tt, li].astype(np.int8)
            df["keep"] = keep_mask(df).astype(np.int8)
            df["t"] = np.int32(w.t)
            df["cond"] = np.int8(w.condition == "queue_onset")
            df["greedy"] = np.int8(w.greedy)
            df["fold"] = fold
            out.append(df)
        print(panel, fold, len(Wf), flush=True)
    res = pd.concat(out, ignore_index=True)
    for c in res.columns:
        if res[c].dtype == np.float64:
            res[c] = res[c].astype(np.float32)
    d = out_dir(causal_t)
    d.mkdir(parents=True, exist_ok=True)
    res.to_parquet(d / f"ds_{panel}.parquet", index=False)
    print(panel, "rows", len(res), "kept", int(res.keep.sum()), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append")
    ap.add_argument("--max-ongoing-train", type=int, default=1200)
    ap.add_argument("--causal-t", action="store_true")
    ap.add_argument("--split-clim", type=float, help="B18 retrained variant (see build_dataset.py)")
    args = ap.parse_args()
    for p in args.panel or PANELS:
        build(p, args.max_ongoing_train, args.causal_t, args.split_clim)


if __name__ == "__main__":
    main()
