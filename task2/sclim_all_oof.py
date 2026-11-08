"""B18 out-of-fold analogue of --split-clim-all: for each official-style train window (months 2..9), both climatologies
are blended lam * fold climatology + (1 - lam) * climatology of "the previous month + the window's own month before
its day" (train observations), and every i5 component is re-predicted with its fold model (onset v1 / v1_thr,
ongoing v1 / phys / aug / onphys). i5 decisions; S_q vs the unblended out-of-fold i5.
"""
from __future__ import annotations

import pickle
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb

import features as FC
import features_v1 as F1
from build import MODELS, predict_mean
from build_dataset import keep_mask as keep_c
from build_dataset_v1 import keep_mask as keep_v1, onset_rows as onset_v1
from common import CACHE, FAMILY, H, PANELS, STEP, T0, Panel, aggregate, select_windows
from model import best_k_set, iou
from split_clim import BlendAll
from thr_feats import THR_COLS, window_features

LAMS = (0.75, 0.5, 0.25)
if os.environ.get("TFB_LAMS"):
    LAMS = tuple(float(x) for x in os.environ["TFB_LAMS"].split(","))
W5 = {"v1": .15, "phys": .15, "aug": .35, "onphys": .35}


class View:
    """Panel arrays cut at row n (rows from n on do not exist)."""

    def __init__(self, P, n):
        self.n_t, self.n_l = n, P.n_l
        for a in ("speed", "flow", "visible", "q", "elig"):
            setattr(self, a, getattr(P, a)[:n])
        self.vcut, self.capacity = P.vcut, P.capacity


def main() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    thr_fold = pickle.load(open(CACHE / "v1_thr" / "thresholds.pkl", "rb"))
    M = {k: {n: [lgb.Booster(model_file=str(MODELS / "cv" / f"{n}_{k}.txt"))]
             for n in ("onset_v1", "onset_v1_thr", "ongoing_v1", "ongoing_phys", "ongoing_aug", "ongoing_onphys")} for k in F1.FOLDS}
    recs = []
    for panel in PANELS:
        st1 = F1.Static(panel); stc = FC.Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        others = {k: [m for f, ms in F1.FOLDS.items() if f != k for m in ms] for k in F1.FOLDS}
        c1 = {k: F1.Clim(P, qc, others[k], onset_t) for k in F1.FOLDS}
        cc = {k: FC.Clim(P, qc, others[k], onset_t) for k in F1.FOLDS}
        cache = {}
        for i, mon in enumerate(months[1:], start=1):
            start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
            end = int((pd.Timestamp(f"{mon}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            k = m2f[mon]
            for w in select_windows(P, qc, max_per_condition=5, start=max(start, 12), end=end).itertuples():
                t = int(w.t); day0 = t - t % 288
                if day0 not in cache:
                    V = View(P, day0)
                    ot = onset_t[(onset_t >= start - 288 * 31) & (onset_t + H + 1 < day0)]
                    ot = ot[np.isin(pd.DatetimeIndex(T0 + ot * STEP).strftime("%Y-%m"), [months[i - 1], mon])]
                    mm = [months[i - 1], mon] if day0 > start else [months[i - 1]]
                    cache[day0] = (F1.Clim(V, qc[:day0], mm, ot), FC.Clim(V, qc[:day0], mm, ot), len(ot))
                s1, sc, n_on = cache[day0]
                rng = np.random.default_rng([PANELS.index(panel), t])
                arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                online = FC.online_features(P.speed, P.vcut, t, start)
                sl = slice(t + 1, t + 1 + H)
                r = {"panel": panel, "month": mon, "condition": w.condition, "t": t, "y": P.q[sl], "e": P.elig[sl]}
                for lam in LAMS:
                    b1 = BlendAll(c1[k], s1, lam, n_on); bc = BlendAll(cc[k], sc, lam, n_on)
                    d1 = F1.link_features(arr, st1, b1, t)
                    if w.condition == "queue_onset":
                        o1 = onset_v1(d1.copy()); o2 = o1.copy()
                        tf = window_features(arr, st1.lanes, thr_fold[panel][k])
                        for c in THR_COLS:
                            o2[c] = tf[c].to_numpy()
                        r[lam] = 0.5 * predict_mean(M[k]["onset_v1"], o1) + 0.5 * predict_mean(M[k]["onset_v1_thr"], o2)
                    else:
                        g1 = F1.horizon_rows(d1, b1, t)
                        p = predict_mean(M[k]["ongoing_v1"], g1); p[keep_v1(g1) == 0] = 0
                        parts = {"v1": np.zeros((H, st1.L))}; parts["v1"][g1.h.to_numpy() - 1, g1.link.to_numpy()] = p
                        gc = FC.ongoing_rows(FC.link_features(arr, stc, bc, t, online), bc, t)
                        kc = keep_c(gc) == 0
                        for v in ("phys", "aug", "onphys"):
                            q = predict_mean(M[k][f"ongoing_{v}"], gc); q[kc] = 0
                            a = np.zeros((H, stc.L)); a[gc.h.to_numpy() - 1, gc.link.to_numpy()] = q; parts[v] = a
                        r[lam] = sum(W5[v] * parts[v] for v in W5) / sum(W5.values())
                recs.append(r)
        print(panel, flush=True)
    pickle.dump(recs, open(CACHE / "sclim_all_oof.pkl", "wb"))
    report(recs)


def report(recs) -> None:
    from mix_eval5 import load
    from mix_eval import combine
    G, kt = load()
    pp = {(r["panel"], r["t"]): r for r in pickle.load(open(CACHE / "pp_windows.pkl", "rb"))}
    meta = pd.DataFrame([{k: r[k] for k in ("panel", "month", "condition", "t")} for r in recs])
    def score(r, p):
        if r["condition"] == "queue_onset":
            pr = np.zeros(r["y"].shape, bool); pr[H - 1] = best_k_set(p)
        else:
            k_t = int(pp[(r["panel"], r["t"])]["q_t5"].sum())
            pr = p >= (0.31 if 1 <= k_t <= 5 else 0.37)
        return iou(pr, r["y"], r["e"])
    base = []
    for r in recs:
        key = (r["panel"], r["t"])
        if r["condition"] == "queue_onset":
            g = G.on[key]; base.append(score(r, .5 * g["v1_0"] + .5 * g["v1_thr_0"]))
        else:
            g = G.go[key]; base.append(score(r, combine({v: g[f"{v}_0"] for v in W5}, W5)))
    base = np.array(base)
    S0 = aggregate(meta.assign(iou=base), "iou")["S_queue"]
    rows = []
    for lam in LAMS:
        a = np.array([score(r, r[lam]) for r in recs]); d = a - base
        mo = meta.assign(d=d).groupby("month").d.mean()
        on = meta.condition.to_numpy() == "queue_onset"
        rows.append({"lam": lam, "dS_q": aggregate(meta.assign(iou=a), "iou")["S_queue"] - S0, "win": d.mean(), "se": d.std() / np.sqrt(len(d)),
                     "onset": d[on].mean(), "ongoing": d[~on].mean(), "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
                     "months_up": int((mo > 0).sum()), "months_down": int((mo < 0).sum())})
    pd.set_option("display.width", 200)
    print(f"windows {len(recs)} (months 2-9); i5 S_q on these {S0:.4f}")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report(pickle.load(open(CACHE / "sclim_all_oof.pkl", "rb")))
    else:
        main()
