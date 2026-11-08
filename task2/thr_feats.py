"""Bottleneck threshold features for onset (method from KAGGLE/sim, thr_lib.py / onset_choice.py, re-written on the
task2 Panel arrays; sim/ is not modified).

Threshold fit (train, unmasked observations, chosen days only):
  events   per day, speed interpolated in time; connected components (8-neighbour) of speed <= 0.6 v_f with >= 6 cells;
           start slot t0 and head link (most downstream link among the start cells)
  groups   head links with >= MIN_EV_PER_DAY * n_days events, adjacent heads (gap <= 3 links) merged
  c_h      per group: member link whose per-lane flow (3-slot mean, time-interpolated) just before the day's first
           onset in the group has the smallest coefficient of variation; c_h = median of those flows
Window features (inputs at or before T only; per-lane flow, causal mean of observed values in the last 3 slots,
else the last observed value):
  thr_dist      links to the nearest bottleneck group at or downstream of the link (0 inside a group, 99 if none)
  thr_r         q(T) / c_h at that group's reference link
  thr_trend     (q(T) - q(T-15 min)) / c_h
  thr_rmax30    max over T-30..T of q / c_h
  thr_gap       1 - thr_r (remaining share to the threshold)
  thr_rank      rank of that group's thr_r among the panel's groups (0 = highest)
  thr_rl        the link's own q(T) / c_h of its nearest group (within 3 links, else NaN)
  thr_rl_nb     max thr_rl over links l-3..l+3
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import label

from common import NHIST, T0, STEP

MIN_EV_PER_DAY = 20 / 273
GAP = 3
THR_COLS = ["thr_dist", "thr_r", "thr_trend", "thr_rmax30", "thr_gap", "thr_rank", "thr_rl", "thr_rl_nb"]


def _interp_day(a: np.ndarray) -> np.ndarray:
    return pd.DataFrame(a).interpolate(limit_direction="both").to_numpy()


def fit_thresholds(P, day_mask: np.ndarray) -> list[tuple[int, int, int, float]]:
    """day_mask: [n_days] bool, days used for the fit. Returns [(first link, last link, ref link, c per lane)]."""
    vf = P.free.astype(np.float32)
    lanes = np.maximum(P.lanes, 1).astype(np.float32)
    days = np.nonzero(day_mask)[0]
    ev = []
    qd = {}
    for d in days:
        sl = slice(d * 288, (d + 1) * 288)
        v = _interp_day(P.speed[sl])
        lab, n = label(v <= 0.6 * vf, structure=np.ones((3, 3)))
        for e in range(1, n + 1):
            ts, ls = np.nonzero(lab == e)
            if len(ts) < 6:
                continue
            t0 = ts.min()
            ev.append((d, int(t0), int(ls[ts == t0].max())))
        q = _interp_day(P.flow[sl] / lanes)
        cs = np.cumsum(np.vstack([np.zeros((1, q.shape[1])), q]), 0)
        lo = np.maximum(np.arange(288) - 2, 0)
        qd[d] = (cs[np.arange(288) + 1] - cs[lo]) / (np.arange(288) + 1 - lo)[:, None]
    ev = pd.DataFrame(ev, columns=["day", "t0", "head"])
    minev = max(int(round(MIN_EV_PER_DAY * len(days))), 5)
    heads = sorted(h for h, g in ev.groupby("head") if len(g) >= minev)
    groups: list[list[int]] = []
    for h in heads:
        if groups and h - groups[-1][-1] <= GAP:
            groups[-1].append(h)
        else:
            groups.append([h])
    out = []
    for G in groups:
        first = ev[ev["head"].isin(G)].sort_values("t0").groupby("day").first()
        best = None
        for h in G:
            pre = np.array([qd[d][t0 - 1, h] for d, t0 in zip(first.index, first.t0) if t0 > 0])
            if len(pre) < 5:
                continue
            cv = pre.std() / pre.mean()
            if best is None or cv < best[3]:
                best = (G[0], G[-1], h, float(np.median(pre)), cv)
        if best:
            out.append(best[:4])
    return out


def causal_q(flow: np.ndarray, lanes: np.ndarray) -> np.ndarray:
    """[rows, L] per-lane flow -> causal 3-slot mean of observed values, else last observed (no look-ahead)."""
    q = flow / np.maximum(lanes, 1)
    ok = np.isfinite(q)
    out = np.full_like(q, np.nan)
    last = np.full(q.shape[1], np.nan)
    for i in range(q.shape[0]):
        a = max(0, i - 2)
        n = ok[a:i + 1].sum(0)
        s = np.where(ok[a:i + 1], q[a:i + 1], 0).sum(0)
        last = np.where(ok[i], q[i], last)
        out[i] = np.where(n > 0, s / np.maximum(n, 1), last)
    return out


def window_features(arr: dict, lanes: np.ndarray, thr: list) -> pd.DataFrame:
    """arr: window arrays [13, L] (rows T-60..T); only these rows are read."""
    L = arr["flow"].shape[1]
    F = {c: np.full(L, np.nan, np.float32) for c in THR_COLS}
    F["thr_dist"][:] = 99
    if not thr:
        return pd.DataFrame(F)
    q = causal_q(arr["flow"].astype(np.float64), lanes)
    g_r = np.array([q[NHIST, h] / c for _, _, h, c in thr])
    g_tr = np.array([(q[NHIST, h] - q[NHIST - 3, h]) / c for _, _, h, c in thr])
    g_mx = np.array([np.nanmax(q[NHIST - 6:NHIST + 1, h]) / c for _, _, h, c in thr])
    order = np.argsort(-np.nan_to_num(g_r, nan=-9))
    rank = np.empty(len(thr)); rank[order] = np.arange(len(thr))
    for l in range(L):
        cand = [(max(a - l, 0), k) for k, (a, b, h, c) in enumerate(thr) if b >= l]
        if cand:
            dist, k = min(cand)
            F["thr_dist"][l] = dist
            F["thr_r"][l] = g_r[k]; F["thr_trend"][l] = g_tr[k]; F["thr_rmax30"][l] = g_mx[k]
            F["thr_gap"][l] = 1 - g_r[k]; F["thr_rank"][l] = rank[k]
        near = [(min(abs(l - a), abs(l - b)) if not a <= l <= b else 0, k) for k, (a, b, h, c) in enumerate(thr)]
        dn, kn = min(near)
        if dn <= 3:
            F["thr_rl"][l] = q[NHIST, l] / thr[kn][3]
    rl = F["thr_rl"]
    F["thr_rl_nb"] = np.array([np.nanmax(rl[max(l - 3, 0):l + 4]) if np.isfinite(rl[max(l - 3, 0):l + 4]).any() else np.nan
                               for l in range(L)], np.float32)
    return pd.DataFrame(F)


def day_mask_for_months(P, months: list[str]) -> np.ndarray:
    n_days = P.n_t // 288
    mon = pd.DatetimeIndex(T0 + np.arange(n_days) * 288 * STEP).strftime("%Y-%m").to_numpy()
    return np.isin(mon, months)
