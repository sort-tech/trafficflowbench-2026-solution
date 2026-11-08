"""(8) Demand-quantile out-of-fold check: does a low-demand hold-out predict the LB direction of past changes?

Per panel, train days are ranked by their mean flow (all links, all slots); the lowest 30% are evaluation days.
A window in fold k is scored with models fitted on rows of the other folds AND of non-evaluation days only (the
climatology is the fold's usual one). Evaluation windows: official-style monthly windows (cand_eval set) on
evaluation days. Fold models: models/cv_demand/<cond>_<variant>_<k>.txt (seed SEED, 300 rounds).
For each past change with a known LB direction, dS_q (aggregated like the LB over the evaluation windows) is
compared with that direction.

Usage: demand_oof.py fit | eval
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq

import features as FC
import features_v1 as F1
from build import MODELS, predict_mean
from build_dataset import keep_mask as keep_c
from build_dataset_v1 import keep_mask as keep_v1, onset_rows as onset_v1
from common import CACHE, H, PANELS, STEP, T0, Panel, aggregate, select_windows
from model import NON_FEATURES, SEED, VARIANTS, best_k_set, fit_dataset, iou, make_dataset
from postprocess import exp_iou, snap_shift
from thr_feats import THR_COLS, window_features

MD = MODELS / "cv_demand"
LOW = 0.30
# (name, table, condition, excluded feature set)
SPECS = [("onset_v1", "v1", 1, set()), ("onset_aug", "ds", 1, VARIANTS["aug"]), ("onset_v1_thr", "v1_thr", 1, set()),
         ("ongoing_v1", "v1", 0, set()), ("ongoing_phys", "ds", 0, VARIANTS["phys"]), ("ongoing_aug", "ds", 0, VARIANTS["aug"]),
         ("ongoing_onphys", "ds", 0, VARIANTS["onphys"])]
TABLE = {"v1": CACHE / "v1", "ds": CACHE, "v1_thr": CACHE / "v1_thr"}


def low_days() -> dict[str, set[int]]:
    path = CACHE / "demand_low_days.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    out = {}
    for p in PANELS:
        P = Panel(p)
        nd = P.n_t // 288
        dm = np.array([np.nanmean(P.flow[d * 288:(d + 1) * 288]) for d in range(nd)])
        out[p] = set(np.nonzero(dm <= np.nanquantile(dm, LOW))[0].tolist())
        del P
    pickle.dump(out, open(path, "wb"))
    return out


def fit() -> None:
    MD.mkdir(parents=True, exist_ok=True)
    low = low_days()
    for name, tab, cond, excl in SPECS:
        d0 = TABLE[tab]
        cols = pq.read_schema(d0 / f"ds_{PANELS[0]}.parquet").names
        feats = [c for c in cols if c not in NON_FEATURES and c not in excl and c != "jit"]
        for k in F1.FOLDS:
            path = MD / f"{name}_{k}.txt"
            if path.exists():
                continue
            filters = [("vis", "=", 1), ("fold", "!=", k)]
            if "cond" in cols:
                filters.append(("cond", "=", cond))
            if cond == 0:
                filters.append(("keep", "=", 1))
            parts = []
            for p in PANELS:
                d = pd.read_parquet(d0 / f"ds_{p}.parquet", columns=feats + ["y", "t"], filters=filters)
                parts.append(d[~np.isin(d.t.to_numpy() // 288, list(low[p]))].drop(columns=["t"]))
            df = pd.concat(parts, ignore_index=True); del parts
            ds = make_dataset(df, feats); del df
            fit_dataset(ds, 300).save_model(str(path)); del ds
            print("fit", name, k, flush=True)


def evaluate() -> None:
    low = low_days()
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    M = {(n, k): [lgb.Booster(model_file=str(MD / f"{n}_{k}.txt"))] for n, *_ in SPECS for k in F1.FOLDS}
    ex = pickle.load(open(CACHE / "pp_extra.pkl", "rb"))
    thr_all = pickle.load(open(CACHE / "v1_thr" / "thresholds.pkl", "rb"))
    recs = []
    for panel in PANELS:
        st1 = F1.Static(panel); stc = FC.Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
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
                if t // 288 not in low[panel]:
                    continue
                rng = np.random.default_rng([PANELS.index(panel), t])
                arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                online = FC.online_features(P.speed, P.vcut, t, start)
                sl = slice(t + 1, t + 1 + H)
                r = {"panel": panel, "month": mon, "condition": w.condition, "t": t, "y": P.q[sl], "e": P.elig[sl],
                     "pi": ex["pi"].get((panel, t))}
                d1 = F1.link_features(arr, st1, c1[k], t)
                dc = FC.link_features(arr, stc, cc[k], t, online)
                if w.condition == "queue_onset":
                    o1 = onset_v1(d1.copy())
                    r["v1"] = predict_mean(M[("onset_v1", k)], o1)
                    r["aug"] = predict_mean(M[("onset_aug", k)], FC.onset_rows(dc, cc[k], t))
                    tf = window_features(arr, st1.lanes, thr_all[panel][k])
                    o2 = o1.copy()
                    for c in THR_COLS:
                        o2[c] = tf[c].to_numpy()
                    r["v1_thr"] = predict_mean(M[("onset_v1_thr", k)], o2)
                    s = snap_shift(c1[k], t) if exp_iou(r["v1"]) < 0.35 else 0
                    r["v1_snap"] = predict_mean(M[("onset_v1", k)], onset_v1(F1.link_features(arr, st1, c1[k], t + s))) if s else r["v1"]
                else:
                    g1 = F1.horizon_rows(d1, c1[k], t)
                    p = predict_mean(M[("ongoing_v1", k)], g1); p[keep_v1(g1) == 0] = 0
                    a = np.zeros((H, st1.L)); a[g1.h.to_numpy() - 1, g1.link.to_numpy()] = p; r["v1"] = a
                    gc = FC.ongoing_rows(dc, cc[k], t)
                    kc = keep_c(gc) == 0
                    for v in ("phys", "aug", "onphys"):
                        p = predict_mean(M[(f"ongoing_{v}", k)], gc); p[kc] = 0
                        a = np.zeros((H, stc.L)); a[gc.h.to_numpy() - 1, gc.link.to_numpy()] = p; r[v] = a
                    sp = np.where(np.isnan(arr["speed"][-1]), arr["speed"][-2], arr["speed"][-1])
                    r["k_t"] = int((np.nan_to_num(sp, nan=1e9) <= st1.vcut).sum())
                recs.append(r)
        print(panel, flush=True)
    pickle.dump(recs, open(CACHE / "demand_oof.pkl", "wb"))
    report(recs)


def _logit_shift(p, d):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + d)))


def report(recs) -> None:
    meta = pd.DataFrame([{k: r[k] for k in ("panel", "month", "condition", "t")} for r in recs])
    on = [r["condition"] == "queue_onset" for r in recs]

    def mix(r, w):
        return sum(wv * r[v] for v, wv in w.items()) / sum(w.values())

    def score(on_rule, go_rule):
        out = []
        for r in recs:
            if r["condition"] == "queue_onset":
                pr = np.zeros(r["y"].shape, bool); pr[H - 1] = on_rule(r)
            else:
                pr = go_rule(r)
            out.append(iou(pr, r["y"], r["e"]))
        return aggregate(meta.assign(iou=out), "iou")["S_queue"]

    V1_ON = lambda r: best_k_set(r["v1"])
    THR_ON = lambda r: best_k_set(0.5 * r["v1"] + 0.5 * r["v1_thr"])

    def go(w, thr=0.37, small=None):
        return lambda r: mix(r, w) >= (small if small is not None and 1 <= r["k_t"] <= 5 else thr)

    def br(r, base_fn, a=0.3):
        pi_s, pi_t = r["pi"]
        return base_fn(_logit_shift(r["v1"], a * np.log(max(pi_s, 1e-5) / max(pi_t, 1e-5))))

    O3 = {"v1": .3, "phys": .7}; G3 = {"v1": .2, "phys": .6, "aug": .2}; G5 = {"v1": .2, "phys": .2, "aug": .6}
    pairs = [  # (change, LB direction, baseline (onset, ongoing), changed (onset, ongoing))
        ("ongphys: ongoing v1 -> v1 .5 + phys .5", "+", (V1_ON, go({"v1": 1}, 0.40)), (V1_ON, go({"v1": .5, "phys": .5}))),
        ("ongphys03: v1 .5 -> .3", "+", (V1_ON, go({"v1": .5, "phys": .5})), (V1_ON, go(O3))),
        ("ongphys0: v1 .3 -> 0 (phys only)", "-", (V1_ON, go(O3)), (V1_ON, go({"phys": 1}, 0.33))),
        ("g3: + aug .2", "+", (V1_ON, go(O3)), (V1_ON, go(G3))),
        ("g5: aug .2 -> .6", "+", (V1_ON, go(G3)), (V1_ON, go(G5))),
        ("ongpo26: aug -> onphys (vs g5)", "+", (V1_ON, go(G5)), (V1_ON, go({"v1": .2, "phys": .2, "onphys": .6}))),
        ("sq31: small-queue 0.31 (vs g5)", "+", (V1_ON, go(G5)), (V1_ON, go(G5, small=0.31))),
        ("onthr: onset v1 -> v1 .5 + v1_thr .5 (vs g5)", "+", (V1_ON, go(G5)), (THR_ON, go(G5))),
        ("g2 onaug: onset v1 -> v1 .6 + aug .4 (vs ongphys03)", "-", (V1_ON, go(O3)), (lambda r: best_k_set(.6 * r["v1"] + .4 * r["aug"]), go(O3))),
        ("todsnap: onset snap (vs ongphys)", "-", (V1_ON, go({"v1": .5, "phys": .5})), (lambda r: best_k_set(r["v1_snap"]), go({"v1": .5, "phys": .5}))),
        ("br03: base-rate a 0.3 on lgb_v1 (onset + ongoing)", "-", (V1_ON, go({"v1": 1}, 0.40)),
         (lambda r: br(r, best_k_set), lambda r: br(r, lambda p: p >= 0.40))),
    ]
    rows = []
    for name, lb, b, c in pairs:
        d = score(*c) - score(*b)
        rows.append({"change": name, "LB": lb, "dS_low_demand": d, "agrees": (d > 0) == (lb == "+")})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"evaluation windows {len(recs)} (onset {sum(on)}, ongoing {len(recs) - sum(on)})")
    print(res.round(4).to_string(index=False))
    print("agreement", int(res.agrees.sum()), "/", len(res))
    res.to_csv(CACHE / "demand_oof.csv", index=False)


if __name__ == "__main__":
    {"fit": fit, "eval": evaluate}[sys.argv[1]]()
