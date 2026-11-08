"""Per-link window features for Task 2 (shared by train pseudo windows and release windows).

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
        fd = pd.read_csv(net / "fd_parameters.csv").set_index("link_id").loc[self.link_ids]
        self.k_crit = fd.critical_density.to_numpy(np.float32)
        self.k_jam = fd.k_jam.to_numpy(np.float32)
        self.w_cong = (self.cap / np.maximum(self.k_jam - self.k_crit, 1)).astype(np.float32)   # km/h backward wave


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
            # gaps at T+30 take the next observation (onset queues persist), else the clean state
            lab = np.where(P.visible[t + H], P.q[t + H],
                           np.where(P.visible[t + H + 1], P.q[t + H + 1], qc[t + H]))
            d = int(dow_of(t) >= 5)
            for s in range(-12, 13):
                w = np.exp(-0.5 * (s / 6.0) ** 2)
                on[d, (t + s) % 288] += w * lab
                oc[d, (t + s) % 288] += w
        self.onset = ((on + 0.05 * on.sum(axis=1, keepdims=True) / max(len(o), 1))
                      / (oc + 0.05 * oc.sum(axis=1, keepdims=True) / max(len(o), 1) + 1e-6)).astype(np.float32)
        self.onset_rate = (oc[:, :, 0] / max(len(o), 1)).astype(np.float32)   # how often onsets occur at this tod
        # time-of-day free onset propensity per link (bottleneck locations are network properties)
        cnt = np.zeros((2, L)); nd = np.zeros(2)
        for t in o:
            d = int(dow_of(t) >= 5)
            cnt[d] += np.where(P.visible[t + H], P.q[t + H], np.where(P.visible[t + H + 1], P.q[t + H + 1], qc[t + H]))
            nd[d] += 1
        self.onset_static = ((cnt + 0.5 * cnt.sum(0) / max(nd.sum(), 1)) / (nd[:, None] + 0.5)).astype(np.float32)
        self.onset_static_all = (cnt.sum(0) / max(nd.sum(), 1)).astype(np.float32)
        # free-flow speed ratio per link and time-of-day mean flow/capacity
        ratio = P.speed[use] / P.vcut
        free = np.where(qc[use] | np.isnan(ratio), np.nan, ratio)
        self.rfree = np.nan_to_num(np.nanmedian(free, axis=0), nan=1.67).astype(np.float32)
        fr = np.nan_to_num(P.flow / P.capacity, nan=0.0)
        has = ~np.isnan(P.flow)
        fnum = np.zeros((2, 288, L))
        fden = np.zeros((2, 288, L))
        for d in (0, 1):
            m = use & (dt == d)
            np.add.at(fnum[d], tod[m], fr[m])
            np.add.at(fden[d], tod[m], has[m])
        fns = sum(np.roll(fnum, s, axis=1) for s in k)
        fds = sum(np.roll(fden, s, axis=1) for s in k)
        self.fclim = (fns / np.maximum(fds, 1)).astype(np.float32)
        # transition climatology: P(q[t+h] | q[t]) by daytype, time of day of t, link, h
        self.p11 = np.zeros((H, 2, 288, L), np.float32)
        self.p01 = np.zeros((H, 2, 288, L), np.float32)
        qi = qc.astype(np.float32)
        for h in range(1, H + 1):
            src = qi[:-h]
            dst = qi[h:]
            uu = use[:-h] & use[h:]
            for d in (0, 1):
                m = uu & (dt[:-h] == d)
                tt = tod[:-h][m]
                n1 = np.zeros((288, L)); n11 = np.zeros((288, L)); n0 = np.zeros((288, L)); n01 = np.zeros((288, L))
                np.add.at(n1, tt, src[m])
                np.add.at(n11, tt, src[m] * dst[m])
                np.add.at(n0, tt, 1 - src[m])
                np.add.at(n01, tt, (1 - src[m]) * dst[m])
                n1, n11, n0, n01 = (sum(np.roll(a, s, axis=0) for s in k) for a in (n1, n11, n0, n01))
                prior = self.q[d, (np.arange(288) + h) % 288]
                self.p11[h - 1, d] = (n11 + 2 * prior) / (n1 + 2)
                self.p01[h - 1, d] = (n01 + 2 * prior) / (n0 + 2)


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


def rh_features(q: np.ndarray, speed: np.ndarray, flow: np.ndarray, st: Static):
    """Rankine-Hugoniot tail speed of each queued run at T (km/h, negative = growing upstream).

    Upstream arrival state from the 3 links above the tail, queue state from the 3 links at the tail.
    Returns per link: tail position (km) and wave speed of the run that governs it (nearest downstream).
    """
    L = st.L
    tail = np.full(L, np.nan, np.float32)
    wv = np.full(L, np.nan, np.float32)
    head = np.full(L, np.nan, np.float32)
    rr = runs(q)
    prev_e = -1
    for s, e in rr:
        up = slice(max(s - 3, 0), s)
        qq = slice(s, min(s + 3, e + 1))
        if up.stop > up.start:
            qu = float(np.mean(flow[up])); vu = float(np.mean(speed[up]))
        else:
            qu = float(flow[s]); vu = float(st.vcut[s] / 0.6)
        qd = float(np.mean(flow[qq])); vd = float(np.mean(speed[qq]))
        ku = qu / max(vu, 1.0); kd = qd / max(vd, 1.0)
        w = (qd - qu) / (kd - ku) if abs(kd - ku) > 1e-3 else 0.0
        w = float(np.clip(w, -40.0, 40.0))
        sl = slice(prev_e + 1, e + 1)
        tail[sl] = st.pos_km[s]; wv[sl] = w; head[sl] = st.pos_km[e]
        prev_e = e
    return tail, wv, head


ONLINE_DAYS = 10   # validation/private windows fall on days 1-12, so at most ~10 earlier days exist


def online_features(speed: np.ndarray, vcut: np.ndarray, t: int, split_start: int, max_days: int = ONLINE_DAYS) -> dict:
    """Same-split statistics computed strictly from rows before the window (index < t - NHIST for today,
    whole earlier days otherwise). speed: [n, L] array of the split; t: index of T in that array.

    These adapt to the split's own timing (validation/private have their own scenario seed).
    """
    L = speed.shape[1]
    tod = t % 288
    ndays = int(min(max((t - tod - split_start) // 288, 0), ONLINE_DAYS, max_days))
    a = t - tod - 288 * ndays                 # only rows from here to T are touched
    sp = speed[a:t + 1]
    q = np.nan_to_num(sp, nan=1e9) <= vcut
    vis = ~np.isnan(sp)
    t = t - a
    day_start = t - tod
    F = {"on_ndays": np.full(L, ndays, np.int8)}
    for k in (0, 6, 12):
        num = np.zeros(L); den = np.zeros(L); anyq = []
        for d in range(1, ndays + 1):
            c = t - 288 * d + k
            sl = slice(c - 3, c + 4)
            num += q[sl].sum(0); den += vis[sl].sum(0)
            anyq.append(q[sl].any())
        F[f"oclim_{k}"] = np.where(den > 0, num / np.maximum(den, 1), np.nan).astype(np.float32)
        F[f"oclim_any_{k}"] = np.full(L, np.mean(anyq) if anyq else np.nan, np.float32)
    # first queued time of day per link on earlier days, relative to T+30
    dist = []; had = np.zeros(L)
    for d in range(1, ndays + 1):
        ds = day_start - 288 * d
        qd = q[ds + 36:ds + 288]          # from 03:00, skip night noise
        anyl = qd.any(0)
        first = np.where(anyl, qd.argmax(0) + 36, np.nan)
        dist.append(np.abs(first - (tod + H)))
        had += anyl
    if ndays:
        dv = np.vstack(dist)
        cnt = np.isfinite(dv).sum(0)
        med = np.full(L, np.nan, np.float32)
        if (cnt > 0).any():
            med[cnt > 0] = np.nanmedian(dv[:, cnt > 0], axis=0)
        F["ofirst_dist"] = med
        F["ofirst_frac"] = (had / ndays).astype(np.float32)
    else:
        F["ofirst_dist"] = np.full(L, np.nan, np.float32)
        F["ofirst_frac"] = np.full(L, np.nan, np.float32)
    # earlier today, before the released history window
    pre_end = max(day_start, t - NHIST)
    qp = q[day_start:pre_end]
    if len(qp):
        anyl = qp.any(0)
        last = np.where(anyl, len(qp) - 1 - qp[::-1].argmax(0), np.nan)
        F["today_q"] = anyl.astype(np.int8)
        F["today_since"] = np.where(anyl, (pre_end - day_start) - last, np.nan).astype(np.float32)
        F["today_nq"] = np.full(L, int(anyl.sum()), np.int16)
    else:
        F["today_q"] = np.zeros(L, np.int8)
        F["today_since"] = np.full(L, np.nan, np.float32)
        F["today_nq"] = np.zeros(L, np.int16)
    for k in (-3, -1, 1, 3):
        idx = np.clip(np.arange(L) + k, 0, L - 1)
        F[f"oclim6_o{k}"] = F["oclim_6"][idx]
        F[f"today_q_o{k}"] = F["today_q"][idx]
    return F


def link_features(arr: dict, st: Static, clim: Clim, t_idx: int, online: dict | None = None) -> pd.DataFrame:
    """arr: speed/flow/occ/pct [13, L] for T-60..T. t_idx: step index of T from T0 (tod/dow only)."""
    L = st.L
    sp = arr["speed"].astype(np.float32)
    fl = arr["flow"].astype(np.float32)
    oc = arr["occ"].astype(np.float32)
    pct = arr["pct"]
    ratio_raw = sp / st.vcut
    r = ffill_rows(ratio_raw)
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
    F["onset_prior"] = clim.onset[dt, tod]
    F["onset_rate"] = np.full(L, clim.onset_rate[dt, tod], np.float32)
    rdev = r[-1] - clim.rfree
    F["r0_dev"] = rdev
    F["rm3_dev"] = r[-4:].mean(0) - clim.rfree
    F["rm12_dev"] = r[:-1].mean(0) - clim.rfree
    fdev = f[-1] - clim.fclim[dt, tod]
    F["f0_dev"] = fdev
    for k in (-3, -2, -1, 1, 2, 3):
        idx = np.clip(np.arange(L) + k, 0, L - 1)
        F[f"rdev_o{k}"] = rdev[idx]
    F["fdev_mean"] = np.full(L, float(np.mean(fdev)), np.float32)
    F["rdev_min"] = np.full(L, float(np.min(rdev)), np.float32)
    F["rdev_rank"] = (np.argsort(np.argsort(rdev)) / L).astype(np.float32)
    sp_T = r[-1] * st.vcut
    fl_T = f[-1] * st.cap
    tail_p, w_rh, head_p = rh_features(q[-1], sp_T, fl_T, st)
    F["rh_w"] = w_rh
    F["rh_tail_rel"] = np.where(np.isnan(tail_p), 99.0, st.pos_km - tail_p)
    F["rh_head_rel"] = np.where(np.isnan(head_p), 99.0, head_p - st.pos_km)
    F["w_cong"] = st.w_cong
    F["cap_margin"] = (1.0 - f[-1]).astype(np.float32)
    F["cap_margin_dmin"] = np.array([np.min(1.0 - f[-1, l:min(l + 6, L)]) for l in range(L)], np.float32)
    F["onset_static"] = clim.onset_static[dt]
    F["onset_static_all"] = clim.onset_static_all
    if online is not None:
        F.update(online)
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
        d["p11_h"] = clim.p11[h - 1, dt, tod]
        d["p01_h"] = clim.p01[h - 1, dt, tod]
        d["tr_h"] = np.where(d["r_0"].to_numpy() <= 1.0, d["p11_h"], d["p01_h"])
        d["clim_dh"] = d["clim_h"] - d["clim_0"]
        d["tail_ext30"] = d["tail_rel"] + d["tail_v30"] * h       # >0 means inside predicted queue extent
        d["tail_ext15"] = d["tail_rel"] + d["tail_v15"] * h
        d["dd_ext"] = d["dd_0"] - d["dd_v6"] * h
        d["rh_ext"] = np.where(d["rh_tail_rel"] > 90, np.nan, d["rh_tail_rel"] + d["rh_w"] * h / 12.0)
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
        # the released row at T (allowed by the rules); the read is filtered to timestamp == T,
        # so no later observation of that day is ever loaded
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


def online_source(panel: str, split: str, st: Static):
    """Return f(t_idx) -> online feature dict for release windows of a split (strictly pre-T rows)."""
    if split == "train":
        P = Panel(panel)
        def f(t_idx: int) -> dict:
            ts = T0 + t_idx * STEP
            month_start = int((pd.Timestamp(ts.strftime("%Y-%m-01T00:00:00Z")) - T0) / STEP)
            return online_features(P.speed, P.vcut, t_idx, month_start)
        return f
    z = np.load(CACHE / f"{panel}_{split}.npz")
    speed = z["speed"]
    start = {"validation": pd.Timestamp("2031-03-01T00:00:00Z"), "private": pd.Timestamp("2031-04-01T00:00:00Z")}[split]
    offset = int((start - T0) / STEP)
    def f(t_idx: int) -> dict:
        return online_features(speed, st.vcut, t_idx - offset, 0)
    return f


AUG_LINK = ["tod", "clim_0", "clim_any0", "clim_3", "clim_6", "clim_m6", "onset_prior", "onset_rate", "f0_dev", "fdev_mean"]
AUG_H = ["clim_h", "clim_dh", "p11_h", "p01_h", "tr_h"]


def aug_link_cols(df_shift: pd.DataFrame) -> pd.DataFrame:
    """Time-of-day columns computed at a shifted origin, renamed aug_* (training augmentation)."""
    return df_shift[AUG_LINK].add_prefix("aug_")


def aug_h_cols(d: pd.DataFrame, clim: Clim, t_aug: int) -> pd.DataFrame:
    """Horizon-specific time-of-day columns at the shifted origin for rows of d (needs h, r_0, aug_clim_0)."""
    tod = t_aug % 288
    dt = int(dow_of(t_aug) >= 5)
    h = d["h"].to_numpy().astype(int)
    link = d["link"].to_numpy().astype(int)
    out = pd.DataFrame(index=d.index)
    out["aug_clim_h"] = clim.q[dt, (tod + h) % 288, link]
    out["aug_clim_dh"] = out["aug_clim_h"] - d["aug_clim_0"].to_numpy()
    out["aug_p11_h"] = clim.p11[h - 1, dt, tod, link]
    out["aug_p01_h"] = clim.p01[h - 1, dt, tod, link]
    out["aug_tr_h"] = np.where(d["r_0"].to_numpy() <= 1.0, out["aug_p11_h"], out["aug_p01_h"])
    return out


def add_aug_identity(d: pd.DataFrame) -> pd.DataFrame:
    """At inference the augmented columns equal the unshifted ones."""
    for c in AUG_LINK + AUG_H:
        if c in d.columns:
            d["aug_" + c] = d[c]
    return d


def onset_rows(df: pd.DataFrame, clim: Clim, t_idx: int) -> pd.DataFrame:
    """Per-link rows for an onset window (only the T+30 cell is predicted)."""
    df = df.copy()
    df["h"] = np.int8(H)
    df["clim_h"] = df["clim_6"]
    df["clim_dh"] = df["clim_6"] - df["clim_0"]
    df["tail_ext30"] = df["tail_rel"]
    df["tail_ext15"] = df["tail_rel"]
    df["dd_ext"] = df["dd_0"]
    dt = int(dow_of(t_idx) >= 5)
    df["p11_h"] = clim.p11[H - 1, dt, t_idx % 288]
    df["p01_h"] = clim.p01[H - 1, dt, t_idx % 288]
    df["tr_h"] = df["p01_h"]
    df["rh_ext"] = np.where(df["rh_tail_rel"] > 90, np.nan, df["rh_tail_rel"] + df["rh_w"] * H / 12.0)
    return add_aug_identity(df)


def ongoing_rows(df: pd.DataFrame, clim: Clim, t_idx: int) -> pd.DataFrame:
    return add_aug_identity(horizon_rows(df, clim, t_idx))
