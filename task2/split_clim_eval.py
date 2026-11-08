"""B7 (b) evidence on validation itself: does a climatology from validation observations help onset?

March is split in halves. The climatology for the v1 onset features is lam * train + (1 - lam) * (one half's
observations); onset pseudo windows of the other half (organizer-style selector on the masked observations) are
predicted with the i5 onset rule (v1 .5 + v1_thr .5, release models) and scored on the observed eligible T+30 cells
(hidden cells are left out of both prediction and truth). Both directions (1-15 -> 16-31 and 16-31 -> 1-15).
Window inputs: masked rows T-60..T of that window; T-row cells hidden at the release rate are not re-hidden.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb

import features_v1 as F1
from build import MODELS, predict_mean, sorted_boosters
from build_dataset_v1 import onset_rows
from common import CACHE, FAMILY, H, NHIST, PANELS, STEP, T0, Panel, aggregate, select_windows
from model import best_k_set, iou
from split_clim import START, Blend, SplitPanel, split_clim
from thr_feats import THR_COLS, fit_thresholds, window_features

LAMS = (1.0, 0.75, 0.5, 0.25, 0.0)


def main() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m_v1 = [lgb.Booster(model_file=str(MODELS / "v1" / "onset.txt"))]
    m_thr = sorted_boosters(MODELS / "v1_thr", "onset_s*.txt")
    rows = []
    for panel in PANELS:
        st = F1.Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        months = sorted(W.month.unique())
        c_train = F1.Clim(P, qc, months, onset_t)
        thr = fit_thresholds(P, np.ones(P.n_t // 288, bool))
        del P, qc
        full = SplitPanel(panel, st, {"validation": 31})
        fq = full.clean_queue()
        a0 = int((START["validation"] - T0) / STEP)
        for fit_days, eval_days in ((range(0, 15), range(15, 31)), (range(15, 31), range(0, 15))):
            # climatology from the fit half only
            sp = SplitPanel(panel, st, {"validation": 31})
            keep = np.zeros(sp.n_t, bool)
            for d in fit_days:
                keep[a0 + 288 * d:a0 + 288 * (d + 1)] = True
            for arr_ in (sp.speed, sp.flow):
                arr_[~keep] = np.nan
            sp.visible = ~np.isnan(sp.speed); sp.q = np.nan_to_num(sp.speed, nan=1e9) <= sp.vcut
            sp.elig = sp.elig & keep[:, None]
            sp.ranges = [(a0 + 288 * fit_days[0], a0 + 288 * (fit_days[-1] + 1))]
            q2 = sp.clean_queue()
            ot = select_windows(sp, q2, start=sp.ranges[0][0] + 12, end=sp.ranges[0][1])
            ot = ot[ot.condition == "queue_onset"].t.to_numpy() if len(ot) else np.array([], int)
            if len(ot) == 0:
                continue
            c_split = F1.Clim(sp, q2, ["2031-03"], ot)
            ev = select_windows(full, fq, start=a0 + 288 * eval_days[0] + 12, end=a0 + 288 * (eval_days[-1] + 1))
            ev = ev[ev.condition == "queue_onset"]
            for w in ev.itertuples():
                t = int(w.t)
                sl = slice(t - NHIST, t + 1)
                arr = {"speed": full.speed[sl].copy(), "flow": full.flow[sl].copy(), "occ": full.occ[sl].copy(),
                       "pct": full.pct[sl].copy()}
                y = full.q[t + H] & full.elig[t + H] & full.visible[t + H]
                e = full.elig[t + H] & full.visible[t + H]
                if not y.any():
                    continue
                tf = window_features(arr, st.lanes, thr)
                r = {"panel": panel, "t": t, "fit": f"{fit_days[0] + 1}-{fit_days[-1] + 1}", "n_split_onsets": len(ot)}
                for lam in LAMS:
                    clim = c_train if lam == 1.0 else Blend(c_train, c_split, lam)
                    d = onset_rows(F1.link_features(arr, st, clim, t))
                    d2 = d.copy()
                    for c in THR_COLS:
                        d2[c] = tf[c].to_numpy()
                    p = 0.5 * predict_mean(m_v1, d) + 0.5 * predict_mean(m_thr, d2)
                    pr = np.zeros((H, st.L), bool); pr[H - 1] = best_k_set(p)
                    Y = np.zeros((H, st.L), bool); Y[H - 1] = y
                    E = np.zeros((H, st.L), bool); E[H - 1] = e
                    r[f"lam{lam}"] = iou(pr, Y, E)
                rows.append(r)
        print(panel, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "split_clim_eval.csv", index=False)
    df["family"] = df.panel.map(FAMILY)
    cols = [f"lam{l}" for l in LAMS]
    print("onset pseudo windows", len(df))
    print(df[cols].mean().round(4).to_dict())
    print(df.groupby("family")[cols].mean().round(3))
    print(df.groupby("fit")[cols].mean().round(3))
    for c in cols[1:]:
        d = df[c] - df["lam1.0"]
        print(f"{c} vs train: {d.mean():+.4f} +- {d.std() / np.sqrt(len(d)):.4f}  better {int((d > 0).sum())} worse {int((d < 0).sum())}")


if __name__ == "__main__":
    main()
