"""Out-of-fold checks for two LB probes built on lgb_v1 (no refit of v1).

A todsnap: onset windows with v1 expected IoU < 0.35 only. Time-of-day lookups (tod, climatology, onset prior,
   onset_rate) move to tod' within +-120 min of the same day (same weekday/weekend climatology) where
   onset_rate(tod' + 30 min) is largest (ties: nearest); state features stay; top-k is chosen again.
B ongphys: ongoing probability w * v1 + (1 - w) * phys (fold models models/cv/ongoing_phys_<k>.txt, same
   3-month folds, single seed), threshold re-tuned; also time-of-day lookups shifted by +-45/+-60 min.
Writes cache/cand_eval_onset.csv and cache/cand_eval_ongoing.pkl, prints the summaries.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb

import features as FC
import features_v1 as F1
from build import predict_mean
from build_dataset_v1 import keep_mask, onset_rows
from common import CACHE, H, PANELS, STEP, T0, Panel, aggregate, select_windows
from model import best_k_set, iou
from postprocess import exp_iou, snap_shift

MD = Path(__file__).resolve().parent / "models" / "cv"
SNAP_CONF = 0.35
SNAP_RANGE = 24          # +-120 min
SHIFTS = (0, 9, -9, 12, -12)


def snapped_onset(models, arr, st, clim, t_idx: int):
    """Returns (p, pred, exp, shift) for an onset window with the snap rule applied when exp < SNAP_CONF."""
    p = predict_mean(models, onset_rows(F1.link_features(arr, st, clim, t_idx)))
    e0 = exp_iou(p)
    if e0 >= SNAP_CONF:
        return p, best_k_set(p), e0, 0
    s = snap_shift(clim, t_idx)
    if s == 0:
        return p, best_k_set(p), e0, 0
    p2 = predict_mean(models, onset_rows(F1.link_features(arr, st, clim, t_idx + s)))
    return p2, best_k_set(p2), e0, s


def main() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    v1 = {k: {n: [lgb.Booster(model_file=str(MD / f"{n}_v1_{k}.txt"))] for n in ("onset", "ongoing")} for k in F1.FOLDS}
    ph = {k: [lgb.Booster(model_file=str(MD / f"ongoing_phys_{k}.txt"))] for k in F1.FOLDS}
    on_rows, go_recs = [], []
    for panel in PANELS:
        st1 = F1.Static(panel); stc = FC.Static(panel)
        P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        others = {k: [m for f, ms in F1.FOLDS.items() if f != k for m in ms] for k in F1.FOLDS}
        c1 = {k: F1.Clim(P, qc, others[k], onset_t) for k in F1.FOLDS}
        cc = {k: FC.Clim(P, qc, others[k], onset_t) for k in F1.FOLDS}
        for mon in months:
            start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
            end = int((pd.Timestamp(f"{mon}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            sel = select_windows(P, qc, max_per_condition=5, start=max(start, 12), end=end)
            k = m2f[mon]
            for w in sel.itertuples():
                t = int(w.t)
                rng = np.random.default_rng([PANELS.index(panel), t])
                arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                sl = slice(t + 1, t + 1 + H)
                y, e = P.q[sl], P.elig[sl]
                if w.condition == "queue_onset":
                    p0 = predict_mean(v1[k]["onset"], onset_rows(F1.link_features(arr, st1, c1[k], t)))
                    _, pred, e0, s = snapped_onset(v1[k]["onset"], arr, st1, c1[k], t)
                    a = np.zeros((H, st1.L), bool); a[H - 1] = best_k_set(p0)
                    b = np.zeros((H, st1.L), bool); b[H - 1] = pred
                    on_rows.append({"panel": panel, "month": mon, "condition": w.condition, "t": t, "exp": e0, "shift": s,
                                    "v1": iou(a, y, e), "snap": iou(b, y, e)})
                else:
                    rec = {"panel": panel, "month": mon, "condition": w.condition, "t": t, "y": y, "e": e}
                    for sh in SHIFTS:
                        d1 = F1.horizon_rows(F1.link_features(arr, st1, c1[k], t + sh), c1[k], t + sh)
                        p1 = predict_mean(v1[k]["ongoing"], d1); p1[keep_mask(d1) == 0] = 0
                        dc = FC.ongoing_rows(FC.link_features(arr, stc, cc[k], t + sh), cc[k], t + sh)
                        p2 = predict_mean(ph[k], dc)
                        from build_dataset import keep_mask as keep_c
                        p2[keep_c(dc) == 0] = 0
                        a1 = np.zeros((H, st1.L), np.float32); a1[d1.h.to_numpy() - 1, d1.link.to_numpy()] = p1
                        a2 = np.zeros((H, st1.L), np.float32); a2[dc.h.to_numpy() - 1, dc.link.to_numpy()] = p2
                        rec[f"v1_{sh}"], rec[f"ph_{sh}"] = a1, a2
                    go_recs.append(rec)
        print(panel, flush=True)
    on = pd.DataFrame(on_rows)
    on.to_csv(CACHE / "cand_eval_onset.csv", index=False)
    pickle.dump(go_recs, open(CACHE / "cand_eval_ongoing.pkl", "wb"))
    report(on, go_recs)


def report(on: pd.DataFrame, go: list[dict]) -> None:
    meta_go = pd.DataFrame([{k: r[k] for k in ("panel", "month", "condition", "t")} for r in go])
    base_go = np.array([iou(r["v1_0"] >= 0.4, r["y"], r["e"]) for r in go])

    def S(on_col, go_sc):
        df = pd.concat([on[["panel", "month", "condition", "t"]].assign(iou=on[on_col].to_numpy()),
                        meta_go.assign(iou=go_sc)], ignore_index=True)
        return df

    base = S("v1", base_go)
    S0 = aggregate(base, "iou")["S_queue"]
    snap = S("snap", base_go)
    ch = on[on["shift"] != 0]
    d = on.snap - on.v1
    mo_up = sum(aggregate(snap[snap.month == m], "iou")["S_queue"] > aggregate(base[base.month == m], "iou")["S_queue"] + 1e-9
                for m in sorted(on.month.unique()))
    mo_dn = sum(aggregate(snap[snap.month == m], "iou")["S_queue"] < aggregate(base[base.month == m], "iou")["S_queue"] - 1e-9
                for m in sorted(on.month.unique()))
    print(f"v1 S_q {S0:.4f}")
    print(f"A todsnap: dS {aggregate(snap, 'iou')['S_queue'] - S0:+.4f}; low-conf onset windows {int((on.exp < SNAP_CONF).sum())}, "
          f"moved {len(ch)} (shift min {ch['shift'].min() * 5 if len(ch) else 0}..{ch['shift'].max() * 5 if len(ch) else 0}), "
          f"better {int((d > 0).sum())} worse {int((d < 0).sum())}, months up/down {mo_up}/{mo_dn}")
    print(ch[["panel", "month", "t", "exp", "shift", "v1", "snap"]].round(3).to_string(index=False))
    rows = []
    for w in (1.0, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.0):
        for thr in (0.33, 0.35, 0.37, 0.40, 0.42, 0.45):
            r = {"w": w, "thr": thr}
            for sh in SHIFTS:
                sc = np.array([iou(w * x[f"v1_{sh}"] + (1 - w) * x[f"ph_{sh}"] >= thr, x["y"], x["e"]) for x in go])
                r[f"S_{sh * 5:+d}"] = aggregate(S("v1", sc), "iou")["S_queue"] - S0
                if sh == 0:
                    dd = sc - base_go
                    r.update({"win_mean": dd.mean(), "win_se": dd.std() / np.sqrt(len(dd)),
                              "better": int((dd > 0).sum()), "worse": int((dd < 0).sum())})
            rows.append(r)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("B ongphys (dS vs v1 at shift 0; S_+-xx = dS vs unshifted v1 with time-of-day lookups shifted):")
    print(res.round(4).to_string(index=False))
    res.to_csv(CACHE / "cand_eval_ongphys.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report(pd.read_csv(CACHE / "cand_eval_onset.csv"), pickle.load(open(CACHE / "cand_eval_ongoing.pkl", "rb")))
    else:
        main()
