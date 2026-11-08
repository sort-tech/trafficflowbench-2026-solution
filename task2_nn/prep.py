"""Data for the Task 2 corridor network (task2_nn). Reads task2/ code and caches only; writes task2_nn/data/.

Per panel, one npz with
  train arrays     speed, flow, occ [n_t, L] float32 (NaN = not observed), pct uint8, label q (observed speed <= v_cut),
                   elig (pct >= 75)
  static           frac, lanes, length, vcut, cap, pid
  climatology      clim_<k>_q [2, 288, L], clim_<k>_any [2, 288], clim_<k>_onset [2, 288, L], clim_<k>_rate [2, 288]
                   for k in A, B, C (months outside fold k) and all (all train months); features_v1.Clim
  thresholds       thr_<k> [G, 4] (first link, last link, ref link, c per lane); thr_feats.fit_thresholds
  train origins    cand_t, cand_cond (1 onset / 0 ongoing), cand_fold (0/1/2) - every origin meeting a condition
                   (windows.candidates with stride 1)
  eval windows     ev_t, ev_cond, ev_fold and ev_speed/flow/occ/pct [n, 13, L]: the official-style monthly windows
                   (5 per condition and month) with the same T-row masks as task2/monthly_v1.py
  release windows  rel_<split>_* : window_history + the masked T row (features_v1.release_windows), t_idx, cond
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T2 = Path(os.environ.get("T2_DIR", HERE.parent / "task2"))   # task2 code + cache (run_all: the fresh task2 build)
sys.path.insert(0, str(T2))
DATA = Path(os.environ.get("NN_DATA", HERE / "data"))         # prep.py output
import numpy as np
import pandas as pd

import features_v1 as F1
from common import CACHE, PANELS, STEP, T0, Panel, select_windows
from thr_feats import day_mask_for_months, fit_thresholds
from windows import candidates

OUT = DATA
FK = list(F1.FOLDS)


def month_of(t: np.ndarray) -> np.ndarray:
    return pd.DatetimeIndex(T0 + np.asarray(t) * STEP).strftime("%Y-%m").to_numpy()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: i for i, (f, ms) in enumerate(F1.FOLDS.items()) for m in ms}
    months = [m for ms in F1.FOLDS.values() for m in ms]
    for panel in PANELS:
        P = Panel(panel); st = F1.Static(panel)
        qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        z = {"speed": P.speed.astype(np.float32), "flow": P.flow.astype(np.float32), "occ": P.occ.astype(np.float32),
             "pct": P.pct.astype(np.uint8), "q": P.q, "elig": P.elig,
             "frac": st.frac, "lanes": st.lanes, "length": st.length, "vcut": st.vcut, "cap": st.cap,
             "pid": np.array(st.pid)}
        for k in FK + ["all"]:
            ms = months if k == "all" else [m for f, mm in F1.FOLDS.items() if f != k for m in mm]
            c = F1.Clim(P, qc, ms, onset_t)
            z[f"clim_{k}_q"], z[f"clim_{k}_any"] = c.q, c.any.astype(np.float32)
            z[f"clim_{k}_onset"], z[f"clim_{k}_rate"] = c.onset, c.onset_rate
            dm = np.ones(P.n_t // 288, bool) if k == "all" else day_mask_for_months(P, ms)
            z[f"thr_{k}"] = np.array(fit_thresholds(P, dm), np.float64).reshape(-1, 4)
        c = candidates(P, qc, stride=1)
        z["cand_t"] = c.t.to_numpy(np.int64)
        z["cand_cond"] = (c.condition == "queue_onset").to_numpy(np.int8)
        z["cand_fold"] = np.array([m2f[m] for m in month_of(c.t.to_numpy())], np.int8)
        ev = {k: [] for k in ("t", "cond", "fold", "speed", "flow", "occ", "pct")}
        for mon in months:
            start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
            end = int((pd.Timestamp(f"{mon}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            for w in select_windows(P, qc, max_per_condition=5, start=max(start, 12), end=end).itertuples():
                rng = np.random.default_rng([PANELS.index(panel), int(w.t)])
                arr = F1.train_window_arrays(P, w.t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                ev["t"].append(int(w.t)); ev["cond"].append(int(w.condition == "queue_onset")); ev["fold"].append(m2f[mon])
                for a in ("speed", "flow", "occ", "pct"):
                    ev[a].append(arr[a])
        for a, v in ev.items():
            z[f"ev_{a}"] = np.array(v) if a in ("t", "cond", "fold") else np.stack(v)
        for split in ("validation", "private"):
            r = {k: [] for k in ("wid", "t", "cond", "speed", "flow", "occ", "pct")}
            for w, arr, t_idx in F1.release_windows(panel, split, st):
                r["wid"].append(w.window_id); r["t"].append(t_idx); r["cond"].append(int(w.condition == "queue_onset"))
                for a in ("speed", "flow", "occ", "pct"):
                    r[a].append(arr[a])
            for a, v in r.items():
                z[f"rel_{split}_{a}"] = np.array(v) if a in ("wid", "t", "cond") else np.stack(v)
        np.savez(OUT / f"{panel}.npz", **z)
        print(panel, P.n_l, "cand", len(c), "onset", int(z["cand_cond"].sum()), "eval", len(ev["t"]),
              "thr groups", len(z["thr_all"]), flush=True)


if __name__ == "__main__":
    main()
