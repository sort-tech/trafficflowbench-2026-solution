"""B7 step 2 (upper bound): refit with the previous month added, score the next month.

For each consecutive train month pair (m, m+1) and every i5 model family, the fold model used for m+1 (trained on
the other two 3-month folds) is refitted with month m's rows added at weight 3 (rows already in the training set
get weight 3, rows of the excluded fold are added). Labels of month m are the full train labels, so this is an
upper bound for what three months of masked validation observations could give for private.
Families: onset v1, onset v1_thr; ongoing v1, phys, aug, onphys (same feature sets and parameters as before).
Models: models/cv_b7/<family>_<m+1>.txt. Evaluation: month m+1 official-style windows, same masking, the fold's
climatology; i5 decisions with the refitted probabilities vs the cached out-of-fold ones.

Usage: b7_refit.py fit <pair index 0..7> | eval
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
from common import CACHE, H, PANELS, STEP, T0, Panel, select_windows
from model import NON_FEATURES, PARAMS, SEED, VARIANTS, make_dataset
from thr_feats import THR_COLS, window_features

MD = MODELS / "cv_b7"
MONTHS = [m for ms in F1.FOLDS.values() for m in ms]
M2F = {m: f for f, ms in F1.FOLDS.items() for m in ms}
FAMS = [("onset_v1", CACHE / "v1", 1, set()), ("onset_v1_thr", CACHE / "v1_thr", 1, set()),
        ("ongoing_v1", CACHE / "v1", 0, set()), ("ongoing_phys", CACHE, 0, VARIANTS["phys"]),
        ("ongoing_aug", CACHE, 0, VARIANTS["aug"]), ("ongoing_onphys", CACHE, 0, VARIANTS["onphys"])]
WEIGHT = 3.0


def month_of(t: np.ndarray) -> np.ndarray:
    return pd.DatetimeIndex(T0 + t * STEP).strftime("%Y-%m").to_numpy()


def fit(i: int) -> None:
    m, m1 = MONTHS[i], MONTHS[i + 1]
    k = M2F[m1]
    MD.mkdir(parents=True, exist_ok=True)
    for name, d0, cond, excl in FAMS:
        path = MD / f"{name}_{m1}.txt"
        if path.exists():
            continue
        cols = pq.read_schema(d0 / f"ds_{PANELS[0]}.parquet").names
        feats = [c for c in cols if c not in NON_FEATURES and c not in excl and c != "jit"]
        filters = [("vis", "=", 1)] + ([("cond", "=", cond)] if "cond" in cols else []) + ([("keep", "=", 1)] if cond == 0 else [])
        parts = []
        for p in PANELS:
            d = pd.read_parquet(d0 / f"ds_{p}.parquet", columns=feats + ["y", "t", "fold"], filters=filters)
            mon = month_of(d.t.to_numpy())
            use = (d.fold.to_numpy() != k) | (mon == m)
            d = d[use]
            d["_w"] = np.where(mon[use] == m, WEIGHT, 1.0).astype(np.float32)
            parts.append(d.drop(columns=["t", "fold"]))
        df = pd.concat(parts, ignore_index=True); del parts
        ds = lgb.Dataset(df[feats], label=df["y"].to_numpy(), weight=df["_w"].to_numpy(), categorical_feature=["pid"],
                         params=dict(PARAMS), free_raw_data=True).construct()
        del df
        lgb.train(dict(PARAMS), ds, num_boost_round=300).save_model(str(path))
        del ds
        print("fit", name, m1, flush=True)


def evaluate() -> None:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    thr_all = pickle.load(open(CACHE / "v1_thr" / "thresholds.pkl", "rb"))
    on, go = {}, {}
    for m1 in MONTHS[1:]:
        k = M2F[m1]
        Mm = {n: [lgb.Booster(model_file=str(MD / f"{n}_{m1}.txt"))] for n, *_ in FAMS}
        for panel in PANELS:
            st1 = F1.Static(panel); stc = FC.Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
            onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
            others = [mm for f, ms in F1.FOLDS.items() if f != k for mm in ms]
            c1 = F1.Clim(P, qc, others, onset_t); cc = FC.Clim(P, qc, others, onset_t)
            start = int((pd.Timestamp(f"{m1}-01T00:00:00Z") - T0) / STEP)
            end = int((pd.Timestamp(f"{m1}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            for w in select_windows(P, qc, max_per_condition=5, start=max(start, 12), end=end).itertuples():
                t = int(w.t)
                rng = np.random.default_rng([PANELS.index(panel), t])
                arr = F1.train_window_arrays(P, t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                d1 = F1.link_features(arr, st1, c1, t)
                if w.condition == "queue_onset":
                    o1 = onset_v1(d1.copy())
                    tf = window_features(arr, st1.lanes, thr_all[panel][k])
                    o2 = o1.copy()
                    for c in THR_COLS:
                        o2[c] = tf[c].to_numpy()
                    on[(panel, t)] = {"v1": predict_mean(Mm["onset_v1"], o1), "v1_thr": predict_mean(Mm["onset_v1_thr"], o2)}
                else:
                    online = FC.online_features(P.speed, P.vcut, t, start)
                    rec = {}
                    g1 = F1.horizon_rows(d1, c1, t)
                    p = predict_mean(Mm["ongoing_v1"], g1); p[keep_v1(g1) == 0] = 0
                    a = np.zeros((H, st1.L), np.float32); a[g1.h.to_numpy() - 1, g1.link.to_numpy()] = p; rec["v1"] = a
                    gc = FC.ongoing_rows(FC.link_features(arr, stc, cc, t, online), cc, t)
                    kc = keep_c(gc) == 0
                    for v in ("phys", "aug", "onphys"):
                        p = predict_mean(Mm[f"ongoing_{v}"], gc); p[kc] = 0
                        a = np.zeros((H, stc.L), np.float32); a[gc.h.to_numpy() - 1, gc.link.to_numpy()] = p; rec[v] = a
                    go[(panel, t)] = rec
            del P
        print(m1, flush=True)
    pickle.dump({"on": on, "go": go}, open(CACHE / "b7_refit_oof.pkl", "wb"))


def report() -> None:
    from mix_eval import combine
    from mix_eval5 import load
    from common import aggregate
    from model import best_k_set, iou
    G, kt = load()
    R = pickle.load(open(CACHE / "b7_refit_oof.pkl", "rb"))
    W5 = {"v1": .15, "phys": .15, "aug": .35, "onphys": .35}

    def scores(src):
        on = []
        for k in G.ko:
            r = G.on[k]
            if src == "refit" and k in R["on"]:
                p = .5 * R["on"][k]["v1"] + .5 * R["on"][k]["v1_thr"]
            else:
                p = .5 * r["v1_0"] + .5 * r["v1_thr_0"]
            pr = np.zeros(r["y"].shape, bool); pr[H - 1] = best_k_set(p); on.append(iou(pr, r["y"], r["e"]))
        go = []
        for i, k in enumerate(G.kg):
            r = G.go[k]
            parts = R["go"][k] if (src == "refit" and k in R["go"]) else {v: r[f"{v}_0"] for v in W5}
            p = combine({v: parts[v] for v in W5}, W5)
            go.append(iou(p >= (0.31 if 1 <= kt[i] <= 5 else 0.37), r["y"], r["e"]))
        return np.array(on), np.array(go)

    b_on, b_go = scores("base"); a_on, a_go = scores("refit")
    om = np.array([G.on[k]["month"] for k in G.ko]); gm = np.array([G.go[k]["month"] for k in G.kg])
    rows = []
    for m1 in MONTHS[1:]:
        def S(on_s, go_s):
            df = pd.concat([G.meta.iloc[:len(G.ko)][om == m1].assign(iou=on_s[om == m1]),
                            G.meta.iloc[len(G.ko):][gm == m1].assign(iou=go_s[gm == m1])])
            return aggregate(df, "iou")["S_queue"]
        rows.append({"test_month": m1, "base": S(b_on, b_go), "refit": S(a_on, a_go), "gain": S(a_on, a_go) - S(b_on, b_go),
                     "gain_onset_only": S(a_on, b_go) - S(b_on, b_go), "gain_ongoing_only": S(b_on, a_go) - S(b_on, b_go)})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    g = res.gain
    print(f"mean gain {g.mean():+.4f} (sd {g.std():.4f}, se {g.std() / np.sqrt(len(g)):.4f}), up {int((g > 1e-9).sum())} / down {int((g < -1e-9).sum())}")
    res.to_csv(CACHE / "b7_refit.csv", index=False)


if __name__ == "__main__":
    {"fit": lambda: fit(int(sys.argv[2])), "eval": evaluate, "report": report}[sys.argv[1]]()
