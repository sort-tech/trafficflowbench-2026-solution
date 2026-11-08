"""Feature tables for train pseudo windows (fold-aware climatology).

Writes cache/ds_<PANEL>.parquet: one row per (window, link, h) for ongoing windows (h=1..6)
and per (window, link) with h=6 for onset windows, plus labels and eligibility for scoring.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE, H, PANELS, STEP, T0, Panel
from features import FOLDS, Clim, Static, aug_h_cols, aug_link_cols, dow_of, horizon_rows, link_features, online_features, train_window_arrays

MASK_RATES = [0.2, 0.3, 0.5]


def keep_mask(df: pd.DataFrame) -> np.ndarray:
    """Input-only row filter: links far from any queue with negligible climatology are predicted 0."""
    near = np.minimum(df.dd_0.to_numpy(), df.du_0.to_numpy()) < 8.0
    clim = np.maximum(df.clim_0.to_numpy(), df.clim_6.to_numpy()) > 0.003
    onset = df.onset_prior.to_numpy() > 0.003
    return near | clim | onset


def build(panel: str, max_ongoing_train: int, shift_seed: int = 0, out_dir: Path | None = None,
          split_clim: float | None = None) -> None:
    """shift_seed > 0 draws another set of time-shift augmentation offsets (aug_m3); 0 keeps the original table."""
    P = Panel(panel)
    qc = np.load(CACHE / f"{panel}_qclean.npy")
    st = Static(panel)
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    W = W[W.panel == panel].copy()
    onset_t = W[W.condition == "queue_onset"].t.to_numpy()
    rng0 = np.random.default_rng(PANELS.index(panel))
    # thin non-greedy ongoing windows for training size
    go = W[(W.condition == "queue_ongoing") & ~W.greedy]
    if len(go) > max_ongoing_train:
        go = go.sample(max_ongoing_train, random_state=PANELS.index(panel))
    W = pd.concat([W[W.greedy | (W.condition == "queue_onset")], go]).drop_duplicates(["t", "condition"]).sort_values("t")
    month2fold = {m: f for f, ms in FOLDS.items() for m in ms}
    SC = None
    if split_clim is not None:
        from split_clim import TrainSplitClims
        SC = TrainSplitClims(P, qc, onset_t, "fc", split_clim)
    out = []
    for fold, months in FOLDS.items():
        other = [m for f, ms in FOLDS.items() if f != fold for m in ms]
        clim_fold = Clim(P, qc, other, onset_t)
        Wf = W[W.month.map(month2fold) == fold]
        for w in Wf.itertuples(index=False):
            clim = clim_fold if SC is None else SC.blend(clim_fold, int(w.t))
            rng = np.random.default_rng([PANELS.index(panel), int(w.t)])
            arr = train_window_arrays(P, w.t, rng, MASK_RATES[rng.integers(3)])
            month_start = int((pd.Timestamp(f"{w.month}-01T00:00:00Z") - T0) / STEP)
            # official windows fall early in the month (onset days 1-5, ongoing days 1-12), so training
            # windows see a matching number of earlier days
            rng2 = np.random.default_rng([PANELS.index(panel), int(w.t), 7])
            max_days = int(rng2.integers(0, 5)) if w.condition == "queue_onset" else int(rng2.integers(0, 11))
            online = online_features(P.speed, P.vcut, int(w.t), month_start, max_days)
            df = link_features(arr, st, clim, w.t, online)
            # time-shift augmentation: time-of-day inputs looked up at a random origin within +-90 min
            rng3 = np.random.default_rng([PANELS.index(panel), int(w.t), 11] + ([shift_seed] if shift_seed else []))
            t_aug = int(w.t) + int(rng3.integers(-18, 19))
            df = pd.concat([df, aug_link_cols(link_features(arr, st, clim, t_aug))], axis=1)
            if w.condition == "queue_onset":
                df["h"] = np.int8(H)
                df["clim_h"] = df["clim_6"]
                df["clim_dh"] = df["clim_6"] - df["clim_0"]
                df["tail_ext30"] = df["tail_rel"]
                df["tail_ext15"] = df["tail_rel"]
                df["dd_ext"] = df["dd_0"]
                df["p11_h"] = clim.p11[H - 1, int(dow_of(w.t) >= 5), w.t % 288]
                df["p01_h"] = clim.p01[H - 1, int(dow_of(w.t) >= 5), w.t % 288]
                df["tr_h"] = df["p01_h"]
                df["rh_ext"] = np.where(df["rh_tail_rel"] > 90, np.nan, df["rh_tail_rel"] + df["rh_w"] * H / 12.0)
                hs = np.full(st.L, H)
                li = np.arange(st.L)
            else:
                df = horizon_rows(df, clim, w.t)
                hs = df.h.to_numpy().astype(int)
                li = df.link.to_numpy().astype(int)
            df = pd.concat([df, aug_h_cols(df.assign(link=df["link"]), clim, t_aug)], axis=1)
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
    d = out_dir or CACHE
    d.mkdir(parents=True, exist_ok=True)
    res.to_parquet(d / f"ds_{panel}.parquet", index=False)
    print(panel, "rows", len(res), "kept", int(res.keep.sum()), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append")
    ap.add_argument("--max-ongoing-train", type=int, default=2500)
    ap.add_argument("--shift-seed", type=int, default=0, help="other time-shift augmentation draws (aug_m3)")
    ap.add_argument("--out-dir", type=Path, help="write ds_<PANEL>.parquet here instead of cache/")
    ap.add_argument("--split-clim", type=float, help="climatology of each window = lam * fold + (1 - lam) * previous month "
                    "+ own month before its day (B18 retrained variant)")
    args = ap.parse_args()
    for p in args.panel or PANELS:
        build(p, args.max_ongoing_train, args.shift_seed, args.out_dir, args.split_clim)


if __name__ == "__main__":
    main()
