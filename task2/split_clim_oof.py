"""B7 (b) out-of-fold analogue of the private recipe: onset climatology = lam * train-fold climatology + (1 - lam) *
climatology from "the previous month + the window's own month before its day" (train observations), for the i5
onset rule (fold models v1 / v1_thr). Official-style windows of months 2..9; month 1 keeps lam = 1.

light_clim() reproduces features_v1.Clim (q with +-15 min smoothing, onset location prior with a +-60 min kernel,
onset_rate) on an arbitrary set of rows and onset windows, so the same code serves release windows.
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
from common import CACHE, FAMILY, H, NHIST, PANELS, STEP, T0, Panel, aggregate, select_windows
from model import best_k_set, iou
from split_clim import Blend
from thr_feats import THR_COLS, window_features

LAMS = (1.0, 0.75, 0.5)


class LightClim:
    pass


def light_clim(qc: np.ndarray, rows: np.ndarray, onset_t: np.ndarray, L: int) -> LightClim:
    """qc: [n, L] clean queue on the absolute index; rows: absolute row indices used; onset_t: onset origins used."""
    tod = rows % 288
    dt = (F1.dow_of(rows) >= 5).astype(int)
    num = np.zeros((2, 288, L)); den = np.zeros((2, 288, 1))
    for d in (0, 1):
        m = dt == d
        np.add.at(num[d], tod[m], qc[rows[m]])
        np.add.at(den[d], tod[m], 1)
    k = np.arange(-F1.CLIM_HALF, F1.CLIM_HALF + 1)
    ns = sum(np.roll(num, s, axis=1) for s in k); ds = sum(np.roll(den, s, axis=1) for s in k)
    c = LightClim()
    c.q = (ns / np.maximum(ds, 1)).astype(np.float32)
    c.any = c.q.max(axis=2)
    on = np.zeros((2, 288, L)); oc = np.zeros((2, 288, 1))
    for t in onset_t:
        lab = qc[t + H]; d = int(F1.dow_of(t) >= 5)
        for s in range(-12, 13):
            w = np.exp(-0.5 * (s / 6.0) ** 2)
            on[d, (t + s) % 288] += w * lab; oc[d, (t + s) % 288] += w
    n = max(len(onset_t), 1)
    c.onset = ((on + 0.05 * on.sum(axis=1, keepdims=True) / n) / (oc + 0.05 * oc.sum(axis=1, keepdims=True) / n + 1e-6)).astype(np.float32)
    c.onset_rate = (oc[:, :, 0] / n).astype(np.float32)
    c.n_onset = len(onset_t)
    return c


def main() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    thr_fold = pickle.load(open(CACHE / "v1_thr" / "thresholds.pkl", "rb"))
    M = {k: {n: [lgb.Booster(model_file=str(MODELS / "cv" / f"onset_{n}_{k}.txt"))] for n in ("v1", "v1_thr")} for k in F1.FOLDS}
    rows = []
    for panel in PANELS:
        st = F1.Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        clims = {k: F1.Clim(P, qc, [m for f, ms in F1.FOLDS.items() if f != k for m in ms], onset_t) for k in F1.FOLDS}
        for i, mon in enumerate(months):
            start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
            end = int((pd.Timestamp(f"{mon}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            prev_start = int((pd.Timestamp(f"{months[i - 1]}-01T00:00:00Z") - T0) / STEP) if i else None
            k = m2f[mon]
            for w in select_windows(P, qc, max_per_condition=5, start=max(start, 12), end=end).itertuples():
                if w.condition != "queue_onset":
                    continue
                t = int(w.t); day0 = t - t % 288
                rng = np.random.default_rng([PANELS.index(panel), t])
                arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                tf = window_features(arr, st.lanes, thr_fold[panel][k])
                r = {"panel": panel, "month": mon, "condition": "queue_onset", "t": t}
                sc = None
                if prev_start is not None:
                    use = np.arange(prev_start, day0)                      # previous month + own month before T's day
                    ot = onset_t[(onset_t >= prev_start) & (onset_t + H < day0)]
                    sc = light_clim(qc, use, ot, st.L)
                    r["n_onset"] = sc.n_onset
                for lam in LAMS:
                    clim = clims[k] if (lam == 1.0 or sc is None) else Blend(clims[k], sc, lam)
                    d = onset_rows(F1.link_features(arr, st, clim, t))
                    d2 = d.copy()
                    for c in THR_COLS:
                        d2[c] = tf[c].to_numpy()
                    p = 0.5 * predict_mean(M[k]["v1"], d) + 0.5 * predict_mean(M[k]["v1_thr"], d2)
                    pr = np.zeros((H, st.L), bool); pr[H - 1] = best_k_set(p)
                    sl = slice(t + 1, t + 1 + H)
                    r[f"lam{lam}"] = iou(pr, P.q[sl], P.elig[sl])
                rows.append(r)
        print(panel, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "split_clim_oof.csv", index=False)
    df["family"] = df.panel.map(FAMILY)
    cols = [f"lam{l}" for l in LAMS]
    print("onset windows", len(df), "mean IoU", df[cols].mean().round(4).to_dict())
    print(df.groupby("family")[cols].mean().round(4))
    for c in cols[1:]:
        d = df[c] - df["lam1.0"]
        mo = df.assign(d=d).groupby("month").d.mean()
        print(f"{c}: onset IoU {d.mean():+.4f} +- {d.std() / np.sqrt(len(d)):.4f} (better {int((d > 0).sum())} worse {int((d < 0).sum())}); "
              f"months up {int((mo > 0).sum())} down {int((mo < 0).sum())}; S_q effect ~ {d.mean() / 2:+.4f}")


if __name__ == "__main__":
    main()
