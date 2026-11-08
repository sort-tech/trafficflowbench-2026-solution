"""Out-of-fold probabilities for the next LB probes (no refit): onset v1 / onphys / phys and ongoing notod,
with time-of-day lookups shifted by 0, +-45, +-60 min (online features stay at the true T).
Same windows and fold models as cand_eval.py; ongoing v1 / phys come from cache/cand_eval_ongoing.pkl.
Writes cache/cand_eval2.pkl and prints the mix grids.
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
from build_dataset import keep_mask as keep_c
from build_dataset_v1 import onset_rows as onset_v1
from common import CACHE, H, PANELS, STEP, T0, Panel, aggregate, select_windows
from model import best_k_set, iou

MD = Path(__file__).resolve().parent / "models" / "cv"
SHIFTS = (0, 9, -9, 12, -12)
PKL = CACHE / "cand_eval2.pkl"


def collect() -> dict:
    if PKL.exists():
        return pickle.load(open(PKL, "rb"))
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    M = {k: {n: [lgb.Booster(model_file=str(MD / f"{n}.txt".replace(".txt", f"_{k}.txt")))]
             for n in ("onset_v1", "onset_onphys", "onset_phys", "ongoing_notod")} for k in F1.FOLDS}
    on, go = {}, {}
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
                online = FC.online_features(P.speed, P.vcut, t, start)
                sl = slice(t + 1, t + 1 + H)
                rec = {"y": P.q[sl], "e": P.elig[sl], "month": mon, "condition": w.condition}
                for sh in SHIFTS:
                    if w.condition == "queue_onset":
                        rec[f"v1_{sh}"] = predict_mean(M[k]["onset_v1"], onset_v1(F1.link_features(arr, st1, c1[k], t + sh)))
                        dc = FC.onset_rows(FC.link_features(arr, stc, cc[k], t + sh, online), cc[k], t + sh)
                        rec[f"onphys_{sh}"] = predict_mean(M[k]["onset_onphys"], dc)
                        rec[f"phys_{sh}"] = predict_mean(M[k]["onset_phys"], dc)
                    else:
                        dc = FC.ongoing_rows(FC.link_features(arr, stc, cc[k], t + sh), cc[k], t + sh)
                        p = predict_mean(M[k]["ongoing_notod"], dc); p[keep_c(dc) == 0] = 0
                        a = np.zeros((H, stc.L), np.float32); a[dc.h.to_numpy() - 1, dc.link.to_numpy()] = p
                        rec[f"notod_{sh}"] = a
                (on if w.condition == "queue_onset" else go)[(panel, t)] = rec
        print(panel, flush=True)
    out = {"on": on, "go": go}
    pickle.dump(out, open(PKL, "wb"))
    return out


def main() -> None:
    d = collect()
    on, go2 = d["on"], d["go"]
    go1 = {(r["panel"], r["t"]): r for r in pickle.load(open(CACHE / "cand_eval_ongoing.pkl", "rb"))}
    keys_on, keys_go = list(on), list(go1)
    meta = pd.DataFrame([{"panel": p, "t": t, "month": on[(p, t)]["month"], "condition": "queue_onset"} for p, t in keys_on]
                        + [{"panel": p, "t": t, "month": go1[(p, t)]["month"], "condition": "queue_ongoing"} for p, t in keys_go])

    def onset_scores(wts: dict, sh: int) -> np.ndarray:
        out = []
        for key in keys_on:
            r = on[key]
            p = sum(w * r[f"{v}_{sh}"] for v, w in wts.items()) / sum(wts.values())
            pr = np.zeros(r["y"].shape, bool); pr[H - 1] = best_k_set(p)
            out.append(iou(pr, r["y"], r["e"]))
        return np.array(out)

    def ongoing_scores(wts: dict, thr: float, sh: int) -> np.ndarray:
        out = []
        for key in keys_go:
            r1 = go1[key]; r2 = go2[key]
            src = {"v1": r1[f"v1_{sh}"], "phys": r1[f"ph_{sh}"], "notod": r2[f"notod_{sh}"]}
            p = sum(w * src[v] for v, w in wts.items()) / sum(wts.values())
            out.append(iou(p >= thr, r1["y"], r1["e"]))
        return np.array(out)

    def S(sc):
        return aggregate(meta.assign(iou=sc), "iou")["S_queue"]

    base_on = onset_scores({"v1": 1}, 0)
    ref_go = ongoing_scores({"v1": 0.5, "phys": 0.5}, 0.37, 0)       # lgb_v1_ongphys
    base_go = ongoing_scores({"v1": 1}, 0.40, 0)
    S0 = S(np.concatenate([base_on, base_go]))
    Sref = S(np.concatenate([base_on, ref_go]))
    print(f"v1 S_q {S0:.4f}; lgb_v1_ongphys {Sref:.4f} ({Sref - S0:+.4f})")
    pd.set_option("display.width", 220)
    rows = []
    for name, wts in [("v1", {"v1": 1}), ("v1+onphys .5", {"v1": .5, "onphys": .5}), ("v1+onphys .7/.3", {"v1": .7, "onphys": .3}),
                      ("v1+phys .5", {"v1": .5, "phys": .5}), ("onphys", {"onphys": 1}), ("v1+onphys+phys", {"v1": .5, "onphys": .25, "phys": .25})]:
        r = {"onset": name}
        for sh in SHIFTS:
            sc = onset_scores(wts, sh)
            r[f"{sh * 5:+d}"] = S(np.concatenate([sc, ref_go])) - Sref
            if sh == 0:
                dd = sc - base_on
                r.update({"win": dd.mean(), "se": dd.std() / np.sqrt(len(dd)), "better": int((dd > 0).sum()), "worse": int((dd < 0).sum())})
        rows.append(r)
    print("onset mixes with ongoing = lgb_v1_ongphys (dS vs lgb_v1_ongphys unshifted):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    rows = []
    for name, wts in [("v1+phys .5", {"v1": .5, "phys": .5}), ("v1+notod .5", {"v1": .5, "notod": .5}),
                      ("v1 .5 phys .25 notod .25", {"v1": .5, "phys": .25, "notod": .25}),
                      ("v1 .4 phys .3 notod .3", {"v1": .4, "phys": .3, "notod": .3}),
                      ("v1 .34 phys .33 notod .33", {"v1": .34, "phys": .33, "notod": .33}),
                      ("phys .5 notod .5", {"phys": .5, "notod": .5}), ("notod", {"notod": 1})]:
        best = None
        for thr in (0.31, 0.33, 0.35, 0.37, 0.40, 0.42):
            sc = ongoing_scores(wts, thr, 0)
            v = S(np.concatenate([base_on, sc]))
            if best is None or v > best[0]:
                best = (v, thr, sc)
        r = {"ongoing": name, "thr": best[1], "dS_vs_v1": best[0] - S0, "dS_vs_ongphys": best[0] - Sref}
        dd = best[2] - ref_go
        r.update({"win_vs_ongphys": dd.mean(), "se": dd.std() / np.sqrt(len(dd))})
        for sh in SHIFTS[1:]:
            r[f"{sh * 5:+d}"] = S(np.concatenate([base_on, ongoing_scores(wts, best[1], sh)])) - S0
        rows.append(r)
    print("ongoing mixes with onset = v1 (shift columns: dS vs unshifted v1):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
