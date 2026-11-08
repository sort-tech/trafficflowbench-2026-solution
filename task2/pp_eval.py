"""Out-of-fold check of the CLAUDE.md post-processing rules on official-style monthly train windows (lgb_v1).

Step 1 (cached in cache/pp_windows.pkl): per window, fold-model probabilities, truth/eligibility [H, L], and the
queue at T: observed cells only, T-5 copy for empty cells (v1's own fill), and restorer fill (fold model).
Step 2: every rule is scored per window and compared with v1 in pairs; S_q uses the official aggregation.
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
from build import predict_mean
from build_dataset_v1 import keep_mask, onset_rows
from common import CACHE, H, PANELS, STEP, T0, Panel, aggregate, select_windows
from model import best_k_set, iou
from postprocess import anchor, closing, limit_growth, onset_adaptive_k, prune_islands
from restore_t import Restorer

MD = Path(__file__).resolve().parent / "models" / "cv"
PKL = CACHE / "pp_windows.pkl"


def collect() -> list[dict]:
    if PKL.exists():
        return pickle.load(open(PKL, "rb"))
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    models = {k: {n: [lgb.Booster(model_file=str(MD / f"{n}_v1_{k}.txt"))] for n in ("onset", "ongoing")} for k in F1.FOLDS}
    rest = {k: Restorer(k) for k in F1.FOLDS}
    out = []
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
                sp = arr["speed"][-1]
                obs = ~np.isnan(sp)
                q_obs = obs & (np.nan_to_num(sp, nan=1e9) <= st.vcut)
                sp5 = np.where(obs, sp, arr["speed"][-2])
                q_t5 = np.nan_to_num(sp5, nan=1e9) <= st.vcut
                q_rest = np.where(obs, sp, rest[k].fill(arr, st)["speed"]) <= st.vcut
                rec = {"panel": panel, "month": mon, "condition": w.condition, "t": int(w.t),
                       "y": P.q[sl].copy(), "e": P.elig[sl].copy(), "q_obs": q_obs, "q_t5": q_t5, "q_rest": q_rest}
                if w.condition == "queue_onset":
                    rec["p"] = predict_mean(models[k]["onset"], onset_rows(df)).astype(np.float32)
                else:
                    d = F1.horizon_rows(df, clims[k], w.t)
                    p = predict_mean(models[k]["ongoing"], d)
                    p[keep_mask(d) == 0] = 0
                    pa = np.zeros((H, L), np.float32)
                    pa[d.h.to_numpy() - 1, d.link.to_numpy()] = p
                    rec["p"] = pa
                out.append(rec)
        print(panel, flush=True)
    pickle.dump(out, open(PKL, "wb"))
    return out


def onset_pred(row: np.ndarray, L: int) -> np.ndarray:
    pr = np.zeros((H, L), bool)
    pr[H - 1] = row
    return pr


def rules() -> dict:
    """name -> (condition, f(rec) -> pred [H, L])."""
    R = {}
    R["v1"] = ("both", None)
    # 2-1 morphology
    for mins in (2, 3, 5):
        R[f"go_close_prune{mins}"] = ("queue_ongoing", lambda r, m=mins: prune_islands(closing(r["p"] >= 0.4), r["q_t5"], m))
        R[f"go_prune{mins}"] = ("queue_ongoing", lambda r, m=mins: prune_islands(r["p"] >= 0.4, r["q_t5"], m))
    R["go_close33"] = ("queue_ongoing", lambda r: closing(r["p"] >= 0.4))
    R["go_close13"] = ("queue_ongoing", lambda r: closing(r["p"] >= 0.4, (1, 3)))
    R["go_close31"] = ("queue_ongoing", lambda r: closing(r["p"] >= 0.4, (3, 1)))
    R["on_close13"] = ("queue_onset", lambda r: onset_pred(closing(best_k_set(r["p"])[None, :], (1, 3))[0], len(r["p"])))
    for mins in (2, 3):
        R[f"on_prune{mins}"] = ("queue_onset", lambda r, m=mins: onset_pred(
            prune_islands(best_k_set(r["p"])[None, :], np.zeros(len(r["p"]), bool), m)[0], len(r["p"])))
    # 2-2 anchoring and upstream growth limit
    for src in ("q_obs", "q_rest", "q_t5"):
        R[f"go_anchor12_{src}"] = ("queue_ongoing", lambda r, s=src: anchor(r["p"] >= 0.4, r[s], (1, 2)))
        R[f"go_anchor1_{src}"] = ("queue_ongoing", lambda r, s=src: anchor(r["p"] >= 0.4, r[s], (1,)))
    for m in (1, 2, 3):
        R[f"go_limit{m}"] = ("queue_ongoing", lambda r, m=m: limit_growth(r["p"] >= 0.4, r["q_t5"], m))
        R[f"go_limit{m}_rest"] = ("queue_ongoing", lambda r, m=m: limit_growth(r["p"] >= 0.4, r["q_rest"] | r["q_t5"], m))
    # 2-3 adaptive k (never empty)
    for c in (0.25, 0.4, 0.6):
        R[f"on_topk_or1_{c}"] = ("queue_onset", lambda r, c=c: onset_pred(onset_adaptive_k(r["p"], c), len(r["p"])))
    return R


def v1_pred(r: dict) -> np.ndarray:
    if r["condition"] == "queue_onset":
        return onset_pred(best_k_set(r["p"]), len(r["p"]))
    return r["p"] >= 0.4


def main() -> None:
    recs = collect()
    base = np.array([iou(v1_pred(r), r["y"], r["e"]) for r in recs])
    meta = pd.DataFrame([{k: r[k] for k in ("panel", "month", "condition", "t")} for r in recs])
    S0 = aggregate(meta.assign(iou=base), "iou")["S_queue"]
    rows = []
    for name, (cond, f) in rules().items():
        if f is None:
            continue
        sc = base.copy()
        for i, r in enumerate(recs):
            if r["condition"] == cond:
                sc[i] = iou(f(r), r["y"], r["e"])
        d = (sc - base)[meta.condition == cond]
        S = aggregate(meta.assign(iou=sc), "iou")["S_queue"]
        # monthly S differences (9 months) as a second, coarser pairing
        mo = [aggregate(meta[meta.month == m].assign(iou=sc[meta.month == m]), "iou")["S_queue"]
              - aggregate(meta[meta.month == m].assign(iou=base[meta.month == m]), "iou")["S_queue"] for m in sorted(meta.month.unique())]
        rows.append({"rule": name, "cond": cond, "dS": S - S0, "win_mean": d.mean(), "win_se": d.std() / np.sqrt(len(d)),
                     "t": d.mean() / max(d.std() / np.sqrt(len(d)), 1e-9), "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
                     "months_up": int(sum(x > 0 for x in mo)), "months_down": int(sum(x < 0 for x in mo))})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"v1 S_q {S0:.4f} ({len(recs)} windows)")
    print(res.round(4).to_string(index=False))
    res.to_csv(CACHE / "pp_eval.csv", index=False)


if __name__ == "__main__":
    main()
