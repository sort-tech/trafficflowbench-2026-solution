"""Zero-score windows: input-only indicators on train pseudo windows (lgb_v1 recipe, out-of-fold) and on
validation/private release windows.

Every indicator uses only the window input (T-60..T) and earlier rows of the same split (whole earlier days
plus today before T-60, like online_features). Outputs:
  cache/diag_zero_train.csv    official-style monthly windows with IoU of lgb_v1 (OOF) and indicators
  cache/diag_zero_release.csv  release windows with the same indicators and a readable summary
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
from common import CACHE, H, NHIST, PANELS, STEP, T0, Panel, select_windows
from model import best_k_set, iou

MD = Path(__file__).resolve().parent / "models"
THR = 0.4
ANOM = 0.02
WIDE = 18          # +-90 min


def wide_clim(q: np.ndarray) -> np.ndarray:
    """Max of the climatology over +-90 min (a queue only slightly off its usual time is not 'out of pattern')."""
    return np.max([np.roll(q, s, axis=1) for s in range(-WIDE, WIDE + 1)], axis=0)


def spans(mask: np.ndarray) -> str:
    """Link index runs of a bool vector, e.g. '12-15,40'."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return "-"
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != p + 1:
            out.append(f"{s}-{p}" if p > s else f"{s}")
            s = i
        p = i
    out.append(f"{s}-{p}" if p > s else f"{s}")
    return ",".join(out)


def earlier_days_q(speed: np.ndarray, vcut: np.ndarray, t: int, split_start: int, max_days: int = 10) -> tuple[np.ndarray, int]:
    """Queue frequency per (offset -6..+12 steps around T's time of day, link) on whole earlier days of the split."""
    tod = t % 288
    day0 = t - tod
    nd = int(min(max((day0 - split_start) // 288, 0), max_days))
    acc = np.zeros((19, speed.shape[1])); cnt = np.zeros((19, speed.shape[1]))
    for d in range(1, nd + 1):
        a = t - 288 * d - 6
        assert a + 19 <= t, "rows before T only"
        sp = speed[a:a + 19]
        vis = ~np.isnan(sp)
        acc += (np.nan_to_num(sp, nan=1e9) <= vcut) & vis
        cnt += vis
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan), nd


def indicators(arr, st, clim, cw, t_idx, cond, models, early) -> tuple[dict, np.ndarray]:
    """Returns indicator dict and the predicted [H, L] bool array."""
    L = st.L
    tod = t_idx % 288
    dt = int(F1.dow_of(t_idx) >= 5)
    df = F1.link_features(arr, st, clim, t_idx)
    pred = np.zeros((H, L), bool)
    ratio = arr["speed"] / st.vcut
    elig = arr["pct"] >= 75
    qh = (np.nan_to_num(ratio, nan=9) <= 1.0) & elig                      # eligible queued cells, T-60..T
    tods = (t_idx - NHIST + np.arange(NHIST + 1)) % 288
    c_hist = clim.q[dt, tods]                                           # [13, L]
    cw_hist = cw[dt, tods]
    I = {"tod_h": round(tod * 5 / 60, 2), "dow": int(F1.dow_of(t_idx)), "miss_T": float(np.isnan(arr["speed"][-1]).mean()),
         "hist_q": int(qh.sum()), "hist_anom": int((qh & (c_hist < ANOM)).sum()), "hist_anom90": int((qh & (cw_hist < ANOM)).sum()),
         "rT_min": float(np.nanmin(ratio[-1])) if np.isfinite(ratio[-1]).any() else np.nan,
         "n_near_T": int((np.nan_to_num(ratio[-1], nan=9) < 1.15).sum())}
    if cond == "queue_onset":
        d = onset_rows(df)
        p = predict_mean(models["onset"], d)
        sel = best_k_set(p)
        pred[H - 1] = sel
        o = np.sort(p)[::-1]
        I.update({"p_max": float(o[0]), "p_sum": float(p.sum()), "k": int(sel.sum()),
                  "exp_iou": float(o[:sel.sum()].sum() / (sel.sum() + o[sel.sum():].sum())),
                  "clim_at_pred": float(clim.q[dt, (tod + H) % 288][sel].mean()),
                  "onset_rate": float(clim.onset_rate[dt, (tod + H) % 288]),
                  "clim_top": spans(clim.q[dt, (tod + H) % 288] >= max(0.5 * clim.q[dt, (tod + H) % 288].max(), 0.05))})
        # precursor: is the slowest link at T inside or next to the predicted set?
        r = np.nan_to_num(ratio[-1], nan=9)
        j = int(np.argmin(r))
        pi = np.flatnonzero(sel)
        I["slow_dist"] = int(np.min(np.abs(pi - j))) if len(pi) else 99
        hit = early[0][6 + H] if early[1] else None
    else:
        d = F1.horizon_rows(df, clim, t_idx)
        p = predict_mean(models["ongoing"], d)
        p[keep_mask(d) == 0] = 0
        sel = p >= THR
        pred[d.h.to_numpy() - 1, d.link.to_numpy()] = sel
        I.update({"p_max": float(p.max()), "p_mean_sel": float(p[sel].mean()) if sel.any() else 0.0, "k": int(sel.sum()),
                  "exp_iou": float(p[sel].sum() / (sel.sum() + p[~sel].sum())) if sel.any() else 0.0,
                  "clim_at_pred": float(clim.q[dt, (tod + 6) % 288][pred[-1]].mean()) if pred[-1].any() else np.nan,
                  "qT_n": int(qh[-1].sum()), "qT_anom90": int((qh[-1] & (cw_hist[-1] < ANOM)).sum()),
                  "clim_top": spans(clim.q[dt, tod] >= max(0.5 * clim.q[dt, tod].max(), 0.05))})
        hit = early[0][6 + H] if early[1] else None
    I["hist_qT5"] = spans(qh[-2])
    I["qT"] = spans(qh[-1])
    I["pred_h6"] = spans(pred[-1])
    I["early_days"] = early[1]
    if hit is not None:
        fin = np.nan_to_num(hit, nan=0.0)
        I["early_at_pred"] = float(fin[pred[-1]].mean()) if pred[-1].any() else np.nan
        I["early_any"] = float(fin.max())
        I["early_top"] = spans(fin >= max(0.5 * fin.max(), 0.2)) if fin.max() > 0 else "-"
    return I, pred


def main() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    fold_models = {k: {n: [lgb.Booster(model_file=str(MD / "cv" / f"{n}_v1_{k}.txt"))] for n in ("onset", "ongoing")}
                   for k in F1.FOLDS}
    full_models = {n: [lgb.Booster(model_file=str(MD / "v1" / f"{n}.txt"))] for n in ("onset", "ongoing")}
    tr_rows, rel_rows = [], []
    for panel in PANELS:
        st = F1.Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        clims = {k: F1.Clim(P, qc, [m for f, ms in F1.FOLDS.items() if f != k for m in ms], onset_t) for k in F1.FOLDS}
        cws = {k: wide_clim(c.q) for k, c in clims.items()}
        for mon in months:
            start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
            end = int((pd.Timestamp(f"{mon}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            sel = select_windows(P, qc, max_per_condition=5, start=max(start, 12), end=end)
            k = m2f[mon]
            for w in sel.itertuples():
                rng = np.random.default_rng([PANELS.index(panel), int(w.t)])
                arr = F1.train_window_arrays(P, w.t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                early = earlier_days_q(P.speed, P.vcut, int(w.t), start)
                I, pred = indicators(arr, st, clims[k], cws[k], int(w.t), w.condition, fold_models[k], early)
                sl = slice(w.t + 1, w.t + 1 + H)
                y, e = P.q[sl], P.elig[sl]
                I.update({"panel": panel, "month": mon, "condition": w.condition, "t": int(w.t), "iou": iou(pred, y, e),
                          "truth_h6": spans((y & e)[-1])})
                tr_rows.append(I)
        clim_all = F1.Clim(P, qc, months, onset_t)
        cw_all = wide_clim(clim_all.q)
        del P, qc
        for split, start in (("validation", "2031-03-01"), ("private", "2031-04-01")):
            sp = np.load(CACHE / f"{panel}_{split}.npz")["speed"]
            off = int((pd.Timestamp(f"{start}T00:00:00Z") - T0) / STEP)
            for w, arr, t_idx in F1.release_windows(panel, split, st):
                early = earlier_days_q(sp, st.vcut, t_idx - off, 0)
                I, _ = indicators(arr, st, clim_all, cw_all, t_idx, w.condition, full_models, early)
                I.update({"panel": panel, "split": split, "window_id": w.window_id, "condition": w.condition,
                          "T": w.forecast_origin, "day": int(w.forecast_origin[8:10])})
                rel_rows.append(I)
        print(panel, flush=True)
    pd.DataFrame(tr_rows).to_csv(CACHE / "diag_zero_train.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(CACHE / "diag_zero_release.csv", index=False)


if __name__ == "__main__":
    main()
