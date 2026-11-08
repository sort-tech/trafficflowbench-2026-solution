"""Per-window strength of the climatology gate (11-05 candidate, main session request).

h = P(observed queue | train climatology >= 0.5) on the split's own masked observations of the complete days before
the window's day (private windows also use all of validation), cells = every observed (speed not NaN) cell of those
rows, queue = speed <= v_cut, climatology = clim_all_q (9 train months) at the row's time of day and daytype.
Shrunk to the panel's train value: h_s = (n_q + K h_ref) / (n + K). The ratio r = h_s / h_ref picks the gate
(fixed before any LB result):
  r >= R_HI           c 0.5           (no new queue on free links with clim >= 0.5; = cpu3g_go50)
  R_LO <= r < R_HI    c 0.0           (no new queue on any link free at T)
  r < R_LO            c 0.0 + wide    (on links free at T a queue needs both i5 and the network alone)
h_ref: the same statistic on train, every row of the 9 train months (unmasked observations), each month read
with the climatology table that leaves its own 3-month fold out (prep.py clim_A/B/C_q).
The split's masked layer hides Task 1 cells by a state-independent hash (docs/MASK_SPEC.md); it also removes the
30-min horizon + 1 h after each scored Task 2 window, which lowers h slightly. Rows are read through task2/split_clim (SplitPanel), so the T-truncation test sees every read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import features_v1 as F1
import split_clim
from common import STEP, T0

K = 500
R_HI, R_LO = 0.9, 0.7
H_REF = {"D12_I5_N": 0.653, "D12_I5_S": 0.678, "D7_I10_E": 0.677, "D7_I10_W": 0.460, "D7_I210_E": 0.915,
         "D7_I210_W": 0.873, "D7_I405_N": 0.694, "D7_I405_S": 0.706}
_cache: dict = {}
_static: dict = {}


def split_counts(panel: str, days: dict[str, int], clim: np.ndarray) -> tuple[int, int]:
    if sum(days.values()) == 0:
        return 0, 0
    st = _static.setdefault(panel, F1.Static(panel))
    P = split_clim.SplitPanel(panel, st, days)
    n = q = 0
    for a, b in P.ranges:
        ti = np.arange(a, b)
        dw = ((ti // 288 + T0.day_of_week) % 7 >= 5).astype(int)
        hi = (clim[dw, ti % 288] >= 0.5) & P.visible[a:b]
        n += int(hi.sum()); q += int((hi & P.q[a:b]).sum())
    return n, q


def gate_from_ratio(r: float) -> tuple[float, bool]:
    if r >= R_HI:
        return 0.5, False
    if r >= R_LO:
        return 0.0, False
    return 0.0, True


def window_gate(panel: str, split: str, t_idx: int, clim: np.ndarray) -> dict:
    a = int((split_clim.START[split] - T0) / STEP)
    days = {split: (t_idx - a) // 288}
    if split == "private":
        days["validation"] = split_clim.N_DAYS["validation"]
    key = (panel, split, days[split])
    if split_clim.CUR_T is not None or key not in _cache:      # no reuse inside the truncation test
        _cache[key] = split_counts(panel, days, clim)
    n, q = _cache[key]
    ref = H_REF[panel]
    hs = (q + K * ref) / (n + K)
    c, wide = gate_from_ratio(hs / ref)
    return {"panel": panel, "split": split, "day": days[split], "n": n, "h_raw": q / n if n else None,
            "h": hs, "ratio": hs / ref, "c": c, "wide": wide}
