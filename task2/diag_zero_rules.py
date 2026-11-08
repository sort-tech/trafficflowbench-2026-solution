"""Alternative decisions for low-confidence windows, scored on train official-style windows (lgb_v1, out-of-fold).

Onset:   model top-k | top-k + slowest link at T (+-1) | slowest link at T (+-1) | top-1 | flattened top-k (p**0.5)
Ongoing: model p >= 0.4 | persistence of the T row (T-5 where empty) | union | p >= 0.3 | p >= 0.5
Writes cache/diag_zero_rules.csv (one row per window, IoU of each alternative, model expected IoU).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb

import features_v1 as F1
from build import predict_mean
from build_dataset_v1 import keep_mask, onset_rows
from common import CACHE, H, PANELS, STEP, T0, Panel, select_windows
from model import best_k_set, iou

MD = Path(__file__).resolve().parent / "models" / "cv"


def main() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    models = {k: {n: [lgb.Booster(model_file=str(MD / f"{n}_v1_{k}.txt"))] for n in ("onset", "ongoing")} for k in F1.FOLDS}
    rows = []
    for panel in PANELS:
        st = F1.Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        clims = {k: F1.Clim(P, qc, [m for f, ms in F1.FOLDS.items() if f != k for m in ms], onset_t) for k in F1.FOLDS}
        L = st.L
        for mon in months:
            start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
            end = int((pd.Timestamp(f"{mon}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            sel = select_windows(P, qc, max_per_condition=5, start=max(start, 12), end=end)
            k = m2f[mon]
            for w in sel.itertuples():
                rng = np.random.default_rng([PANELS.index(panel), int(w.t)])
                arr = F1.train_window_arrays(P, w.t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                df = F1.link_features(arr, st, clims[k], w.t)
                sl = slice(w.t + 1, w.t + 1 + H)
                y, e = P.q[sl], P.elig[sl]
                ratio = arr["speed"] / st.vcut
                alt = {}
                if w.condition == "queue_onset":
                    p = predict_mean(models[k]["onset"], onset_rows(df))
                    s = best_k_set(p)
                    o = np.sort(p)[::-1]; kk = int(s.sum())
                    ex = o[:kk].sum() / (kk + o[kk:].sum())
                    rT = np.nan_to_num(ratio[-1], nan=9.0)
                    j = int(np.argmin(rT))
                    slow = np.zeros(L, bool); slow[max(j - 1, 0):j + 2] = True
                    top1 = np.zeros(L, bool); top1[int(np.argmax(p))] = True
                    for name, m in (("model", s), ("model_slow", s | slow), ("slow", slow), ("top1", top1),
                                    ("flat", best_k_set(np.sqrt(p)))):
                        pr = np.zeros((H, L), bool); pr[H - 1] = m
                        alt[name] = iou(pr, y, e)
                else:
                    d = F1.horizon_rows(df, clims[k], w.t)
                    p = predict_mean(models[k]["ongoing"], d)
                    p[keep_mask(d) == 0] = 0
                    hh, ll = d.h.to_numpy() - 1, d.link.to_numpy()
                    s = p >= 0.4
                    ex = p[s].sum() / (s.sum() + p[~s].sum()) if s.any() else 0.0
                    sp = arr["speed"][-1].copy()
                    miss = np.isnan(sp); sp[miss] = arr["speed"][-2][miss]
                    pers = np.broadcast_to(np.nan_to_num(sp, nan=1e9) <= st.vcut, (H, L))
                    for name, thr in (("model", 0.4), ("thr30", 0.3), ("thr50", 0.5)):
                        pr = np.zeros((H, L), bool); pr[hh, ll] = p >= thr
                        alt[name] = iou(pr, y, e)
                        if name == "model":
                            alt["union"] = iou(pr | pers, y, e)
                    alt["persT"] = iou(pers, y, e)
                rows.append({"panel": panel, "month": mon, "condition": w.condition, "t": int(w.t), "exp_iou": float(ex), **alt})
        print(panel, flush=True)
    pd.DataFrame(rows).to_csv(CACHE / "diag_zero_rules.csv", index=False)


if __name__ == "__main__":
    main()
