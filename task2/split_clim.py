"""Climatology from a release split's own masked observations (B7 b), for the frozen v1 features.

SplitPanel mimics common.Panel on the absolute time index (T0 = 2030-06-01) but only holds rows of the chosen days
of validation/private (masked observations: mainline_states_masked speed/flow/pct). From it:
  qc       clean-queue proxy inside the chosen days only (speed forward/back filled within the split, <= v_cut)
  onset_t  onset windows of those days (the organizer-style selector on the observations, no cap)
  Clim     features_v1.Clim on those months -> q[dt, tod, l], onset[dt, tod, l], onset_rate[dt, tod]
blend(train, split, lam) returns a Clim-like object with lam * train + (1 - lam) * split for those arrays.
Only rows of the chosen days are loaded, so a caller that passes days before T uses nothing after T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.dataset as pds

import features_v1 as F1
from common import H, REL, STEP, T0, select_windows

START = {"validation": pd.Timestamp("2031-03-01T00:00:00Z"), "private": pd.Timestamp("2031-04-01T00:00:00Z")}
N_DAYS = {"validation": 31, "private": 30}
_tables: dict = {}
CUR_T: pd.Timestamp | None = None     # set by trunc_test.py: rows after this time must not exist


def split_table(panel: str, split: str) -> pd.DataFrame:
    if (panel, split) not in _tables:
        t = pds.dataset(str(REL / "corridors" / panel / split / "mainline_states_masked"), format="parquet",
                        partitioning="hive").to_table(columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed"]).to_pandas()
        t["ti"] = ((pd.to_datetime(t.timestamp, utc=True) - T0) // STEP).astype(np.int64)
        _tables[(panel, split)] = t
    return _tables[(panel, split)]


class SplitPanel:
    def __init__(self, panel: str, st, days: dict[str, int]):
        """days: split -> number of leading days of that split to load (rows of later days are never read)."""
        self.panel = panel
        last = max(int((START[s] - T0) / STEP) + 288 * n for s, n in days.items() if n > 0)
        self.n_t, self.n_l = last, st.L
        self.speed = np.full((self.n_t, st.L), np.nan, np.float32)
        self.flow = np.full((self.n_t, st.L), np.nan, np.float32)
        self.occ = np.full((self.n_t, st.L), np.nan, np.float32)
        pct = np.zeros((self.n_t, st.L), np.uint8)
        pos = {k: i for i, k in enumerate(st.link_ids)}
        self.ranges = []
        for s, n in days.items():
            if n <= 0:
                continue
            a = int((START[s] - T0) / STEP); b = a + 288 * n
            t = split_table(panel, s)
            t = t[(t.ti >= a) & (t.ti < b)]
            if CUR_T is not None:
                assert int((t.ti.max() if len(t) else 0)) <= int((CUR_T - T0) / STEP), "row after T read"
            li = t.link_id.astype(str).map(pos).to_numpy()
            self.speed[t.ti.to_numpy(), li] = t.speed_kmh.to_numpy(np.float32)
            self.flow[t.ti.to_numpy(), li] = t.flow_vph.to_numpy(np.float32)
            self.occ[t.ti.to_numpy(), li] = t.occupancy.to_numpy(np.float32)
            pct[t.ti.to_numpy(), li] = t.pct_observed.to_numpy().astype(np.uint8)
            self.ranges.append((a, b))
        self.vcut = st.vcut
        self.capacity = st.cap
        self.pct = pct
        self.elig = pct >= 75
        self.visible = ~np.isnan(self.speed)
        self.q = np.nan_to_num(self.speed, nan=1e9) <= self.vcut

    def clean_queue(self) -> np.ndarray:
        qc = np.zeros((self.n_t, self.n_l), bool)
        for a, b in self.ranges:
            s = pd.DataFrame(self.speed[a:b]).ffill().bfill().to_numpy()
            qc[a:b] = s <= self.vcut
        return qc


def split_clim(panel: str, st, days: dict[str, int]):
    """Clim from the leading `days` of the given splits (None if no onset window was found there)."""
    P = SplitPanel(panel, st, days)
    qc = P.clean_queue()
    onset_t = []
    for a, b in P.ranges:
        w = select_windows(P, qc, start=a + 12, end=b)
        if len(w):
            onset_t += w[w.condition == "queue_onset"].t.tolist()
    months = sorted({pd.Timestamp(T0 + a * STEP).strftime("%Y-%m") for a, _ in P.ranges})
    if not onset_t:
        return None, 0
    return F1.Clim(P, qc, months, np.array(onset_t, int)), len(onset_t)


class Blend:
    """lam * train + (1 - lam) * split for the arrays the v1 features read."""

    def __init__(self, train, split, lam: float):
        self.q = lam * train.q + (1 - lam) * split.q
        self.any = self.q.max(axis=2)
        self.onset = lam * train.onset + (1 - lam) * split.onset
        self.onset_rate = lam * train.onset_rate + (1 - lam) * split.onset_rate


def onset_split_clim(panel: str, split: str, st, t_idx: int, use_val_for_private: bool = True):
    """Split climatology for an onset window at t_idx: complete days of its own split before T's day, plus all of
    validation for private windows. Returns (LightClim or None, number of onset windows used)."""
    from split_clim_oof import light_clim
    a = int((START[split] - T0) / STEP)
    day = (t_idx - a) // 288
    days = {split: day}
    if split == "private" and use_val_for_private:
        days["validation"] = N_DAYS["validation"]
    if sum(days.values()) == 0:
        return None
    P = SplitPanel(panel, st, days)
    qc = P.clean_queue()
    rows, onset_t = [], []
    for a0, b0 in P.ranges:
        rows.append(np.arange(a0, b0))
        w = select_windows(P, qc, start=a0 + 12, end=b0)
        if len(w):
            onset_t += w[w.condition == "queue_onset"].t.tolist()
    return light_clim(qc, np.concatenate(rows), np.array(onset_t, int), st.L)


_full_cache: dict = {}


def split_clims(panel: str, split: str, st, t_idx: int, use_val_for_private: bool = True):
    """Both climatologies (frozen-v1 Clim and current-feature Clim) from the complete days of the window's split before
    T's day (private: plus all of validation). Rows of later days are never loaded. Returns (v1 clim, fc clim,
    days loaded, onset windows) or None when fewer than two days are available. Cached per (panel, split, day)."""
    import features as FC
    a = int((START[split] - T0) / STEP)
    day = (t_idx - a) // 288
    key = (panel, split, day, use_val_for_private)
    if key in _full_cache:
        return _full_cache[key]
    days = {split: day}
    if split == "private" and use_val_for_private:
        days["validation"] = N_DAYS["validation"]
    if sum(days.values()) < 2:
        _full_cache[key] = None
        return None
    P = SplitPanel(panel, st, days)
    qc = P.clean_queue()
    onset_t = []
    for a0, b0 in P.ranges:
        w = select_windows(P, qc, start=a0 + 12, end=b0)
        if len(w):
            onset_t += w[w.condition == "queue_onset"].t.tolist()
    onset_t = np.array(onset_t, int)
    months = sorted({pd.Timestamp(T0 + a0 * STEP).strftime("%Y-%m") for a0, _ in P.ranges})
    out = (F1.Clim(P, qc, months, onset_t), FC.Clim(P, qc, months, onset_t), sum(days.values()), len(onset_t))
    _full_cache[key] = out
    return out


class BlendAll:
    """lam * train + (1 - lam) * split for every climatology array the features read. Location priors (onset,
    onset_rate, onset_static) keep the train values when the split has fewer than min_onsets onset windows."""

    ARRS = ("q", "onset", "onset_rate", "onset_static", "onset_static_all", "rfree", "fclim", "p11", "p01")

    def __init__(self, train, split, lam: float, n_onsets: int, min_onsets: int = 5):
        for a in self.ARRS:
            if not hasattr(train, a):
                continue
            lt = lam if (n_onsets >= min_onsets or a not in ("onset", "onset_rate", "onset_static", "onset_static_all")) else 1.0
            setattr(self, a, (lt * getattr(train, a) + (1 - lt) * getattr(split, a)).astype(np.float32))
        self.any = self.q.max(axis=2)


class _View:
    """Panel arrays cut at row n (rows from n on do not exist)."""

    def __init__(self, P, n):
        self.n_t, self.n_l = n, P.n_l
        for a in ("speed", "flow", "visible", "q", "elig"):
            setattr(self, a, getattr(P, a)[:n])
        self.vcut, self.capacity = P.vcut, P.capacity


class TrainSplitClims:
    """Train analogue of split_clims for building training tables: for a train window at t, the climatology of
    "the previous month + the window's own month before its day" (kind 'v1' -> features_v1.Clim, 'fc' -> features.Clim),
    blended with the fold climatology like BlendAll. Cached per day."""

    def __init__(self, P, qc, onset_t: np.ndarray, kind: str, lam: float):
        self.P, self.qc, self.onset_t, self.kind, self.lam = P, qc, onset_t, kind, lam
        self.cache: dict = {}
        self.months = [m for ms in F1.FOLDS.values() for m in ms]

    def split(self, t: int):
        day0 = t - t % 288
        if day0 in self.cache:
            return self.cache[day0]
        mon = pd.Timestamp(T0 + day0 * STEP).strftime("%Y-%m")
        i = self.months.index(mon)
        start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
        mm = ([self.months[i - 1]] if i else []) + ([mon] if day0 > start else [])
        first = int((pd.Timestamp(f"{mm[0]}-01T00:00:00Z") - T0) / STEP) if mm else day0
        if not mm or day0 - first < 2 * 288:
            self.cache[day0] = None
            return None
        V = _View(self.P, day0)
        ot = self.onset_t[(self.onset_t >= first) & (self.onset_t + H + 1 < day0)]
        if self.kind == "v1":
            c = F1.Clim(V, self.qc[:day0], mm, ot)
        else:
            import features as FC
            c = FC.Clim(V, self.qc[:day0], mm, ot)
        self.cache[day0] = (c, len(ot))
        return self.cache[day0]

    def blend(self, fold_clim, t: int):
        sc = self.split(t)
        return fold_clim if sc is None else BlendAll(fold_clim, sc[0], self.lam, sc[1])
