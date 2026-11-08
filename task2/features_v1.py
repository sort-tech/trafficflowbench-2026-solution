"""Frozen lgb_v1 feature code (as of 2026-10-25 12:47 KST), kept so lgb_v1 can be rebuilt from scratch.

Differences from features.py: the onset prior is looked up at tod + 6, the onset location label is the
clean state at T+30, and there are no deviation / transition / online / physics features. The only edits to
the original are (a) the T-row read is filtered to timestamp == T (same values, nothing later is loaded) and
(b) the optional t_fill argument used by the causal T-row restorer (restore_t.py).

Per-link window features for Task 2 (shared by train pseudo windows and release windows).

A window input is four [13, L] arrays (rows T-60 .. T, row 12 = the released row at T):
speed, flow, occ, pct. Nothing after T is ever read here.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from common import CACHE, H, NHIST, PANELS, REL, STEP, T0, Panel

NROW = NHIST + 1          # T-60 .. T
FOLDS = {"A": ["2030-06", "2030-07", "2030-08"], "B": ["2030-09", "2030-10", "2030-11"],
         "C": ["2030-12", "2031-01", "2031-02"]}
CLIM_HALF = 3             # +-15 min smoothing of the time-of-day climatology
OFFS = [-8, -5, -3, -2, -1, 1, 2, 3, 5, 8]


class Static:
    def __init__(self, panel: str):
        net = REL / "corridors" / panel / "network"
        topo = pd.read_csv(net / "lwr_mainline_topology.csv")
        links = pd.read_csv(net / "links.csv").merge(topo[["link_id", "order_index"]], on="link_id", how="left")
        links = links.sort_values("order_index").reset_index(drop=True)
        self.panel = panel
        self.link_ids = links.link_id.astype(str).to_numpy()
        self.L = len(links)
        self.vcut = (0.6 * links.free_speed_kmh.astype(float)).to_numpy(np.float32)
        self.cap = links.capacity_vph.to_numpy(np.float32)
        self.lanes = links.lanes.to_numpy(np.float32)
        self.length = links.length_km.to_numpy(np.float32)
        self.pos_km = (np.cumsum(self.length) - self.length / 2).astype(np.float32)
        self.frac = np.linspace(0, 1, self.L).astype(np.float32)
        self.pid = PANELS.index(panel)


def dow_of(t: np.ndarray | int) -> np.ndarray:
    return (np.asarray(t) // 288 + T0.day_of_week) % 7


class Clim:
    """Time-of-day climatology of the clean queue state, by daytype, from chosen train months.

    q[dt, tod, l] = P(queued); onset[dt, tod, l] = P(link queued at the first-queue step | onset there).
    """

    def __init__(self, P: Panel, qc: np.ndarray, months: list[str], onset_t: np.ndarray):
        ts = T0 + np.arange(P.n_t) * STEP
        mon = pd.DatetimeIndex(ts).strftime("%Y-%m").to_numpy()
        use = np.isin(mon, months)
        t_idx = np.arange(P.n_t)
        tod = t_idx % 288
        dt = (dow_of(t_idx) >= 5).astype(int)
        L = P.n_l
        num = np.zeros((2, 288, L))
        den = np.zeros((2, 288, 1))
        for d in (0, 1):
            m = use & (dt == d)
            np.add.at(num[d], tod[m], qc[m])
            np.add.at(den[d], tod[m], 1)
        # circular smoothing over +-CLIM_HALF steps
        k = np.arange(-CLIM_HALF, CLIM_HALF + 1)
        ns = sum(np.roll(num, s, axis=1) for s in k)
        ds = sum(np.roll(den, s, axis=1) for s in k)
        self.q = (ns / np.maximum(ds, 1)).astype(np.float32)
        self.any = (self.q.max(axis=2))
        # onset location prior with a +-60 min kernel
        o = onset_t[np.isin(mon[onset_t], months)]
        on = np.zeros((2, 288, L))
        oc = np.zeros((2, 288, 1))
        for t in o:
            lab = qc[t + H]
            d = int(dow_of(t) >= 5)
            for s in range(-12, 13):
                w = np.exp(-0.5 * (s / 6.0) ** 2)
                on[d, (t + s) % 288] += w * lab
                oc[d, (t + s) % 288] += w
        self.onset = ((on + 0.05 * on.sum(axis=1, keepdims=True) / max(len(o), 1))
                      / (oc + 0.05 * oc.sum(axis=1, keepdims=True) / max(len(o), 1) + 1e-6)).astype(np.float32)
        self.onset_rate = (oc[:, :, 0] / max(len(o), 1)).astype(np.float32)   # how often onsets occur at this tod


def ffill_rows(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaN along axis 0 (time), then back-fill."""
    out = a.copy()
    for i in range(1, out.shape[0]):
        m = np.isnan(out[i])
        out[i, m] = out[i - 1, m]
    for i in range(out.shape[0] - 2, -1, -1):
        m = np.isnan(out[i])
        out[i, m] = out[i + 1, m]
    return out


def fill_space(v: np.ndarray, default: float) -> np.ndarray:
    """Linear interpolation along links for remaining NaN (1-D)."""
    m = np.isnan(v)
    if not m.any():
        return v
    if m.all():
        return np.full_like(v, default)
    x = np.arange(len(v))
    out = v.copy()
    out[m] = np.interp(x[m], x[~m], v[~m])
    return out


def runs(q: np.ndarray):
    """Contiguous queued runs as (start, end) inclusive, upstream->downstream order."""
    d = np.diff(np.concatenate([[0], q.astype(np.int8), [0]]))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0] - 1
    return list(zip(s, e))


def dist_features(q: np.ndarray, pos: np.ndarray):
    """For each link: km distance to nearest queued link downstream (>=0) and upstream (>=0)."""
    L = len(q)
    BIG = 99.0
    dd = np.full(L, BIG, np.float32)
    du = np.full(L, BIG, np.float32)
    idx = np.where(q)[0]
    if len(idx):
        j = np.searchsorted(idx, np.arange(L))            # first queued index >= l
        ok = j < len(idx)
        dd[ok] = pos[idx[j[ok]]] - pos[np.arange(L)[ok]]
        j2 = np.searchsorted(idx, np.arange(L), side="right") - 1   # last queued index <= l
        ok2 = j2 >= 0
        du[ok2] = pos[np.arange(L)[ok2]] - pos[idx[j2[ok2]]]
    return dd, du


def run_features(q: np.ndarray, pos: np.ndarray, length: np.ndarray):
    L = len(q)
    in_len = np.zeros(L, np.float32)
    to_head = np.zeros(L, np.float32)
    to_tail = np.zeros(L, np.float32)
    for s, e in runs(q):
        ln = pos[e] - pos[s] + (length[s] + length[e]) / 2
        in_len[s:e + 1] = ln
        to_head[s:e + 1] = pos[e] - pos[s:e + 1]
        to_tail[s:e + 1] = pos[s:e + 1] - pos[s]
    return in_len, to_head, to_tail


def tail_track(qhist: np.ndarray, pos: np.ndarray):
    """Per link: upstream-tail position (km) of the run that is nearest downstream (or contains l),
    at T and 30 min earlier, giving a tail velocity (km/step, positive = moving upstream)."""
    L = qhist.shape[1]
    out_now = np.full(L, np.nan, np.float32)
    out_vel = np.zeros(L, np.float32)
    out_vel15 = np.zeros(L, np.float32)
    rT = runs(qhist[-1])
    if not rT:
        return out_now, out_vel, out_vel15
    # tail of each run at earlier steps: run in the earlier frame that overlaps the head region
    def tail_at(frame, s, e):
        rr = runs(frame)
        best = None
        for s2, e2 in rr:
            if e2 >= s and s2 <= e:      # overlap
                best = s2 if best is None else min(best, s2)
        return best
    starts = np.array([s for s, _ in rT])
    for s, e in rT:
        t6 = tail_at(qhist[-7], s, e)
        t3 = tail_at(qhist[-4], s, e)
        v30 = (pos[t6] - pos[s]) / 6.0 if t6 is not None else (pos[e] - pos[s]) / 6.0 + 0.1
        v15 = (pos[t3] - pos[s]) / 3.0 if t3 is not None else (pos[e] - pos[s]) / 3.0 + 0.1
        # links governed by this run: from previous run's head+1 (exclusive) .. e
        prev_e = max([e2 for s2, e2 in rT if e2 < s], default=-1)
        sl = slice(prev_e + 1, e + 1)
        out_now[sl] = pos[s]
        out_vel[sl] = v30
        out_vel15[sl] = v15
    return out_now, out_vel, out_vel15


def link_features(arr: dict, st: Static, clim: Clim, t_idx: int, t_fill: dict | None = None) -> pd.DataFrame:
    """arr: speed/flow/occ/pct [13, L] for T-60..T. t_idx: step index of T from T0 (tod/dow only).
    t_fill: optional speed/flow/occ [L] estimates for the empty cells of the T row (from T-row inputs only)."""
    L = st.L
    sp = arr["speed"].astype(np.float32)
    fl = arr["flow"].astype(np.float32)
    oc = arr["occ"].astype(np.float32)
    pct = arr["pct"]
    ratio_raw = sp / st.vcut
    if t_fill is not None:
        sp, fl, oc = sp.copy(), fl.copy(), oc.copy()
        for a, k in ((sp, "speed"), (fl, "flow"), (oc, "occ")):
            m = np.isnan(a[-1])
            a[-1, m] = t_fill[k][m]
    r = ffill_rows(sp / st.vcut)
    for i in range(NROW):
        r[i] = fill_space(r[i], 1.67)
    f = ffill_rows(fl / st.cap)
    for i in range(NROW):
        f[i] = fill_space(f[i], 0.5)
    o = ffill_rows(oc)
    for i in range(NROW):
        o[i] = fill_space(o[i], 0.05)
    q = r <= 1.0
    qe = (ratio_raw <= 1.0) & (pct >= 75)
    tod = t_idx % 288
    dow = int(dow_of(t_idx))
    dt = int(dow >= 5)

    F = {}
    F["pid"] = np.full(L, st.pid, np.int16)
    F["frac"] = st.frac
    F["lanes"] = st.lanes
    F["length"] = st.length
    F["vcut"] = st.vcut
    F["tod"] = np.full(L, tod, np.int16)
    F["dow"] = np.full(L, dow, np.int8)
    for k in (0, 1, 2, 3, 6, 12):
        F[f"r_{k}"] = r[NROW - 1 - k]
    F["r_min3"] = r[-4:].min(0)
    F["r_mean12"] = r[:-1].mean(0)
    F["miss_T"] = np.isnan(ratio_raw[-1]).astype(np.int8)
    F["f_0"] = f[-1]
    F["f_m3"] = f[-4:].mean(0)
    F["f_tr6"] = f[-1] - f[-7]
    F["f_tr12"] = f[-1] - f[0]
    F["o_0"] = o[-1]
    F["qe_n12"] = qe[:-1].sum(0).astype(np.int8)
    F["q_n12"] = q[:-1].sum(0).astype(np.int8)
    for k in OFFS:
        idx = np.clip(np.arange(L) + k, 0, L - 1)
        F[f"r0_o{k}"] = r[-1, idx]
        F[f"r6_o{k}"] = r[-7, idx]
        if abs(k) <= 3:
            F[f"f0_o{k}"] = f[-1, idx]
    dd0, du0 = dist_features(q[-1], st.pos_km)
    dd6, du6 = dist_features(q[-7], st.pos_km)
    dd3, _ = dist_features(q[-4], st.pos_km)
    F["dd_0"], F["du_0"], F["dd_6"], F["du_6"], F["dd_3"] = dd0, du0, dd6, du6, dd3
    F["dd_v6"] = np.where((dd0 < 90) & (dd6 < 90), (dd6 - dd0) / 6.0, 0.0)
    F["dd_v3"] = np.where((dd0 < 90) & (dd3 < 90), (dd3 - dd0) / 3.0, 0.0)
    rl, th, tt = run_features(q[-1], st.pos_km, st.length)
    F["run_len"], F["to_head"], F["to_tail"] = rl, th, tt
    tail_now, tv30, tv15 = tail_track(q, st.pos_km)
    F["tail_rel"] = np.where(np.isnan(tail_now), 99.0, st.pos_km - tail_now)   # <0 upstream of tail
    F["tail_v30"], F["tail_v15"] = tv30, tv15
    F["nq_0"] = np.full(L, q[-1].sum(), np.int16)
    F["nq_6"] = np.full(L, q[-7].sum(), np.int16)
    F["nq_12"] = np.full(L, q[0].sum(), np.int16)
    F["clim_0"] = clim.q[dt, tod]
    F["clim_any0"] = np.full(L, clim.any[dt, tod], np.float32)
    for h in (3, 6):
        F[f"clim_{h}"] = clim.q[dt, (tod + h) % 288]
    F["clim_m6"] = clim.q[dt, (tod - 6) % 288]
    F["onset_prior"] = clim.onset[dt, (tod + H) % 288]
    F["onset_rate"] = np.full(L, clim.onset_rate[dt, (tod + H) % 288], np.float32)
    df = pd.DataFrame(F)
    df["link"] = np.arange(L, dtype=np.int16)
    return df


def horizon_rows(df: pd.DataFrame, clim: Clim, t_idx: int) -> pd.DataFrame:
    """Expand per-link features to (link, h) rows for h=1..6 with horizon-specific features."""
    L = len(df)
    tod = t_idx % 288
    dt = int(dow_of(t_idx) >= 5)
    parts = []
    for h in range(1, H + 1):
        d = df.copy()
        d["h"] = np.int8(h)
        d["clim_h"] = clim.q[dt, (tod + h) % 288]
        d["clim_dh"] = d["clim_h"] - d["clim_0"]
        d["tail_ext30"] = d["tail_rel"] + d["tail_v30"] * h       # >0 means inside predicted queue extent
        d["tail_ext15"] = d["tail_rel"] + d["tail_v15"] * h
        d["dd_ext"] = d["dd_0"] - d["dd_v6"] * h
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- inputs

def train_window_arrays(P: Panel, t: int, rng: np.random.Generator, mask_rate: float | None) -> dict:
    sl = slice(t - NHIST, t + 1)
    arr = {"speed": P.speed[sl].copy(), "flow": P.flow[sl].copy(), "occ": P.occ[sl].copy(), "pct": P.pct[sl].copy()}
    if mask_rate:
        m = (arr["pct"][-1] >= 75) & (rng.random(P.n_l) < mask_rate)
        for k in ("speed", "flow", "occ"):
            arr[k][-1, m] = np.nan
    return arr


def release_windows(panel: str, split: str, st: Static):
    """Yield (window row, arrays) for release windows: window_history + the masked row at T."""
    base = REL / "task2" / panel / split
    wi = pd.read_csv(base / "window_index.csv")
    hist = pd.read_parquet(base / "window_history.parquet")
    pos = {k: i for i, k in enumerate(st.link_ids)}
    for w in wi.itertuples(index=False):
        T = pd.Timestamp(w.forecast_origin)
        times = [(T - (NHIST - i) * STEP).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(NROW)]
        arr = {k: np.full((NROW, st.L), np.nan, np.float32) for k in ("speed", "flow", "occ")}
        arr["pct"] = np.zeros((NROW, st.L), np.uint8)
        h = hist[hist.window_id == w.window_id]
        ti = h.timestamp.map({s: i for i, s in enumerate(times)}).to_numpy()
        li = h.link_id.astype(str).map(pos).to_numpy()
        assert not np.isnan(ti.astype(float)).any() and ti.max() <= NHIST - 1, "history must end at T-5"
        arr["speed"][ti, li] = h.speed_kmh.to_numpy()
        arr["flow"][ti, li] = h.flow_vph.to_numpy()
        arr["occ"][ti, li] = h.occupancy.to_numpy()
        arr["pct"][ti, li] = h.pct_observed.to_numpy()
        # the released row at T (allowed by the rules); the read is filtered to timestamp == T
        d = T.strftime("%Y_%m_%d")
        f = glob.glob(str(REL / "corridors" / panel / split / "mainline_states_masked" / "*" / f"synthetic_mainline_{d}.parquet"))
        if f:
            r = pd.read_parquet(f[0], columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed"],
                                filters=[("timestamp", "=", times[-1])])
            li = r.link_id.astype(str).map(pos).to_numpy()
            arr["speed"][NHIST, li] = r.speed_kmh.to_numpy()
            arr["flow"][NHIST, li] = r.flow_vph.to_numpy()
            arr["occ"][NHIST, li] = r.occupancy.to_numpy()
            arr["pct"][NHIST, li] = r.pct_observed.to_numpy()
        t_idx = int((T - T0) / STEP)
        yield w, arr, t_idx
