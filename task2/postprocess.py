"""Decision post-processing rules from KAGGLE/CLAUDE.md (guide 2-1..2-3), applied to lgb_v1 outputs.

All rules use only the window input: the model probabilities and the queue state at T (observed, or
filled by the causal T-row restorer). Arrays are [H, L] with links ordered upstream -> downstream.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label

import features_v1 as F1
from common import H
from model import best_k_set


def closing(pred: np.ndarray, size: tuple[int, int] = (3, 3)) -> np.ndarray:
    """Morphological closing with edge padding (no artificial erosion at the array border)."""
    ph, pl = size[0] // 2, size[1] // 2
    a = np.pad(pred, ((ph, ph), (pl, pl)), mode="edge")
    s = np.ones(size, bool)
    a = binary_erosion(binary_dilation(a, structure=s), structure=s)
    return a[ph:ph + pred.shape[0], pl:pl + pred.shape[1]]


def prune_islands(pred: np.ndarray, q_t: np.ndarray, min_size: int) -> np.ndarray:
    """Drop 8-connected components smaller than min_size that do not touch the queue at T in the first row."""
    out = pred.copy()
    lab, n = label(pred, structure=np.ones((3, 3)))
    for i in range(1, n + 1):
        m = lab == i
        touches = (m[0] & binary_dilation(q_t, structure=np.ones(3, bool))).any()
        if not touches and m.sum() < min_size:
            out[m] = False
    return out


def anchor(pred: np.ndarray, q_t: np.ndarray, hs: tuple[int, ...] = (1, 2)) -> np.ndarray:
    """Force the queue observed at T to 1 at the given horizons (1-based)."""
    out = pred.copy()
    for h in hs:
        out[h - 1] |= q_t
    return out


def limit_growth(pred: np.ndarray, q_t: np.ndarray, links_per_step: int) -> np.ndarray:
    """Keep a predicted cell (h, l) only if a link queued at T lies within links_per_step * h links downstream
    (queues grow upstream, i.e. toward lower link index). Cells with no queue at T anywhere are left alone."""
    if not q_t.any():
        return pred
    L = len(q_t)
    idx = np.where(q_t, np.arange(L), L + 10**6)
    # nearest queued link at or downstream of l
    nxt = np.minimum.accumulate(idx[::-1])[::-1]
    out = pred.copy()
    for h in range(1, pred.shape[0] + 1):
        out[h - 1] &= (nxt - np.arange(L)) <= links_per_step * h
    return out


def onset_adaptive_k(p: np.ndarray, conf: float) -> np.ndarray:
    """Guide 2-3 without the empty set (every truth window has a queue): top-1 when max p < conf."""
    if p.max() < conf:
        out = np.zeros(len(p), bool)
        out[int(np.argmax(p))] = True
        return out
    return best_k_set(p)


def exp_iou(p: np.ndarray) -> float:
    """Expected IoU of the top-k set chosen by best_k_set (ratio-of-expectations form)."""
    s = best_k_set(p)
    o = np.sort(p)[::-1]; k = int(s.sum())
    return float(o[:k].sum() / (k + o[k:].sum()))


def snap_shift(clim, t_idx: int, rng: int = 24) -> int:
    """Offset s (steps, |s| <= rng, same day so the weekday/weekend climatology is kept) maximizing the v1 onset
    frequency onset_rate[dt, tod + s + 30 min]; ties -> smallest |s|. Train and release times are both UTC."""
    tod = t_idx % 288
    dt = int(F1.dow_of(t_idx) >= 5)
    cand = [s for s in range(-rng, rng + 1) if 0 <= tod + s < 288]
    rate = np.array([clim.onset_rate[dt, (tod + s + H) % 288] for s in cand])
    ok = [s for s, r in zip(cand, rate) if r >= rate.max() - 1e-12]
    return min(ok, key=lambda s: (abs(s), s))
