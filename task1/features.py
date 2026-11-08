"""Spatio-temporal features for masked cells.

Arrays are laid out (time, link) with time running across consecutive days and
links upstream -> downstream. Everything is computed from the masked layer
only, so the same code serves train (with truth for labels) and the scored
splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common

OFFSETS = (-3, -2, -1, 1, 2, 3)


def prev_next(x: np.ndarray):
    """For each cell, the nearest finite value strictly before / after along axis 0,
    and the distance to it in steps (inf where none)."""
    n, m = x.shape
    ok = np.isfinite(x)
    ar = np.arange(n)[:, None]
    pi = np.where(ok, ar, -1)
    pi = np.maximum.accumulate(pi, axis=0)
    pi = np.vstack([np.full((1, m), -1), pi[:-1]])  # strictly before
    ni = np.where(ok, ar, n)
    ni = np.minimum.accumulate(ni[::-1], axis=0)[::-1]
    ni = np.vstack([ni[1:], np.full((1, m), n)])  # strictly after
    cols = np.arange(m)[None, :]
    pv = np.where(pi >= 0, x[np.clip(pi, 0, n - 1), cols], np.nan)
    nv = np.where(ni < n, x[np.clip(ni, 0, n - 1), cols], np.nan)
    gp = np.where(pi >= 0, ar - pi, np.inf).astype(np.float32)
    gn = np.where(ni < n, ni - ar, np.inf).astype(np.float32)
    return pv.astype(np.float32), nv.astype(np.float32), gp, gn


def interp(pv, nv, gp, gn):
    with np.errstate(invalid="ignore", divide="ignore"):
        w = gn / (gp + gn)
    lin = np.where(np.isfinite(gp) & np.isfinite(gn), w * pv + (1 - w) * nv, np.nan)
    lin = np.where(np.isfinite(lin), lin, np.where(np.isfinite(pv), pv, nv))
    return lin.astype(np.float32)


def window_mean(x: np.ndarray, h: int):
    """Mean of finite values within +-h steps, excluding the centre."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0).astype(np.float64)
    c = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(ok, 0)])
    s = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(v, 0)])
    n = x.shape[0]
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    ss = s[hi] - s[lo] - v
    cc = c[hi] - c[lo] - ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cc > 0, ss / cc, np.nan).astype(np.float32), cc.astype(np.float32)


def window_slope(x: np.ndarray, h: int):
    """Least-squares slope of finite values within +-h, excluding centre (per step)."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0).astype(np.float64)
    n = x.shape[0]
    num = np.zeros_like(v)
    den = np.zeros_like(v)
    sm = np.zeros_like(v)
    cnt = np.zeros_like(v)
    for k in range(-h, h + 1):
        if k == 0:
            continue
        sh = np.roll(v, -k, axis=0)
        so = np.roll(ok, -k, axis=0)
        if k > 0:
            so[n - k:] = False
        else:
            so[:-k] = False
        num += np.where(so, k * sh, 0)
        den += np.where(so, k * k, 0)
        sm += np.where(so, sh, 0)
        cnt += so
    # slope through the centre, adjusted for the window mean offset
    sk = np.zeros_like(v)
    for k in range(-h, h + 1):
        if k == 0:
            continue
        so = np.roll(ok, -k, axis=0)
        if k > 0:
            so[n - k:] = False
        else:
            so[:-k] = False
        sk += np.where(so, k, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_k = sk / cnt
        mean_x = sm / cnt
        cov = num / cnt - mean_k * mean_x
        var = den / cnt - mean_k ** 2
        slope = np.where((cnt >= 3) & (var > 0), cov / var, np.nan)
        level = mean_x - slope * mean_k  # value at k=0
    return slope.astype(np.float32), level.astype(np.float32)


def shift_links(a: np.ndarray, o: int):
    """b[:, l] = a[:, l+o], NaN outside."""
    b = np.full_like(a, np.nan)
    if o > 0:
        b[:, :-o] = a[:, o:]
    elif o < 0:
        b[:, -o:] = a[:, :o]
    else:
        b[:] = a
    return b


def profiles(datasets: list[dict], lanes: np.ndarray):
    """weekday x tod median-ish (mean) of observed per-lane values, pooled over datasets."""
    nl = len(lanes)
    sv = np.zeros((7, common.T, nl)); sq = np.zeros((7, common.T, nl)); c = np.zeros((7, common.T, nl))
    for d in datasets:
        wd = common.slot_weekday(d["dates"])
        v = d["obs_speed_kmh"]; q = d["obs_flow_vph"] / lanes
        ok = np.isfinite(v) & np.isfinite(q) & (d["pct"] >= 75)
        for w in range(7):
            s = wd == w
            sv[w] += np.where(ok[s], v[s], 0).sum(0)
            sq[w] += np.where(ok[s], q[s], 0).sum(0)
            c[w] += ok[s].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pv = sv / c
        pq = sq / c
    # smooth over +-1 slot and fill holes
    for a in (pv, pq):
        flat = a.reshape(7 * common.T, nl)
        m, _ = window_mean(flat, 1)
        both = np.where(np.isfinite(flat), (flat + 2 * np.nan_to_num(m, nan=0) * np.isfinite(m)) /
                        (1 + 2 * np.isfinite(m)), m)
        flat[:] = both
        lm = np.nanmean(flat, 0)
        flat[:] = np.where(np.isfinite(flat), flat, lm[None, :])
    return pv.astype(np.float32), pq.astype(np.float32)


def ramp_profile(parts) -> np.ndarray:
    """(7, 288, n_ramps) weekday x slot mean of observed ramp flow over (flow, dates)
    pairs, smoothed over +-1 slot; holes take the ramp's overall mean, then 0."""
    nr = parts[0][0].shape[2]
    s = np.zeros((7, common.T, nr)); c = np.zeros((7, common.T, nr))
    for flow, dates in parts:
        wd = common.slot_weekday(dates)
        ok = np.isfinite(flow)
        for w in range(7):
            m = wd == w
            s[w] += np.where(ok[m], flow[m], 0).sum(0)
            c[w] += ok[m].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = s / c
    flat = p.reshape(7 * common.T, nr)
    m, _ = window_mean(flat, 1)
    flat[:] = np.where(np.isfinite(flat), (flat + 2 * np.nan_to_num(m) * np.isfinite(m)) / (1 + 2 * np.isfinite(m)), m)
    with np.errstate(invalid="ignore"):
        lm = np.nanmean(flat, 0)
    flat[:] = np.where(np.isfinite(flat), flat, np.nan_to_num(lm)[None, :])
    return p.astype(np.float32)


def fit_bottlenecks(d: dict, net: pd.DataFrame, day_mask: np.ndarray, min_ev_per_day=20 / 273, gap=3):
    """Bottleneck groups and their hidden thresholds c_h (method of KAGGLE/sim and
    task2/thr_feats.py, on the unmasked train layer of the chosen days).

    events: per day, speed interpolated in time, 8-connected components of v <= 0.6 v_f with
    >= 6 cells; start slot t0 and head link (most downstream start cell). Groups: head links
    with enough events, heads <= gap links apart merged. c_h: the member link whose per-lane
    flow (3-slot mean) just before the day's first onset varies least; c_h its median.
    Returns [(first link, last link, reference link, c_h per lane)]."""
    from scipy.ndimage import label
    vf = net.free_speed_kmh.to_numpy(np.float64)
    lanes = net.lanes.to_numpy(np.float64)
    days = np.nonzero(day_mask)[0]
    ev, qd = [], {}
    for dd in days:
        v = pd.DataFrame(d["true_speed"][dd]).interpolate(limit_direction="both").to_numpy()
        lab, n = label(v <= 0.6 * vf, structure=np.ones((3, 3)))
        for e in range(1, n + 1):
            ts, ls = np.nonzero(lab == e)
            if len(ts) < 6:
                continue
            t0 = ts.min()
            ev.append((dd, int(t0), int(ls[ts == t0].max())))
        q = pd.DataFrame(d["true_flow"][dd] / lanes).interpolate(limit_direction="both").to_numpy()
        cs = np.cumsum(np.vstack([np.zeros((1, q.shape[1])), q]), 0)
        lo = np.maximum(np.arange(common.T) - 2, 0)
        qd[dd] = (cs[np.arange(common.T) + 1] - cs[lo]) / (np.arange(common.T) + 1 - lo)[:, None]
    ev = pd.DataFrame(ev, columns=["day", "t0", "head"])
    minev = max(int(round(min_ev_per_day * len(days))), 5)
    heads = sorted(h for h, g in ev.groupby("head") if len(g) >= minev)
    groups = []
    for h in heads:
        if groups and h - groups[-1][-1] <= gap:
            groups[-1].append(h)
        else:
            groups.append([h])
    out = []
    for G in groups:
        first = ev[ev["head"].isin(G)].sort_values("t0").groupby("day").first()
        best = None
        for h in G:
            pre = np.array([qd[dd][t0 - 1, h] for dd, t0 in zip(first.index, first.t0) if t0 > 0])
            if len(pre) < 5:
                continue
            cv = pre.std() / pre.mean()
            if best is None or cv < best[4]:
                best = (G[0], G[-1], h, float(np.median(pre)), cv)
        if best:
            out.append(best[:4])
    return out


class Block:
    """Feature source for a run of consecutive days of one panel."""

    def __init__(self, d: dict, days: np.ndarray, net: pd.DataFrame, prof, ramps=None, extra_blank=None,
                 version: int = 3, windows=None, thr=None):
        """windows: (flat origin index within the block, condition code 1=onset 2=ongoing) pairs
        of Task 2 windows whose horizon + buffer (T+5..T+90) is blank."""
        self.version = version
        self.days = days
        lanes = net.lanes.to_numpy(np.float32)
        self.lanes = lanes
        nd = len(days)
        nl = len(lanes)
        v = d["obs_speed_kmh"][days].reshape(nd * common.T, nl).copy()
        q = (d["obs_flow_vph"][days] / lanes).reshape(nd * common.T, nl).copy()
        o = d["obs_density_occ_linear_vehpkm"][days].reshape(nd * common.T, nl) / lanes
        if extra_blank is not None:
            eb = extra_blank.reshape(nd * common.T, nl)
            v[eb] = np.nan; q[eb] = np.nan
            o = np.where(eb, np.nan, o)
        bad = ~(np.isfinite(v) & np.isfinite(q))
        v[bad] = np.nan; q[bad] = np.nan
        self.v, self.q, self.o = v, q, o.astype(np.float32)
        k = q / np.clip(v, 1, None)
        self.k = k
        wd = common.slot_weekday(d["dates"][days])
        self.tod = np.tile(np.arange(common.T), nd)
        self.wd = np.repeat(wd, common.T)
        pv, pq = prof
        self.pv = pv[wd].reshape(nd * common.T, nl)
        self.pq = pq[wd].reshape(nd * common.T, nl)
        self.f = {}
        for name, x in (("v", v), ("q", q), ("k", k), ("o", self.o)):
            pvv, nvv, gp, gn = prev_next(x)
            self.f[f"{name}_prev"] = pvv
            self.f[f"{name}_next"] = nvv
            if name == "v":
                self.f["gp"] = gp
                self.f["gn"] = gn
            self.f[f"{name}_I"] = interp(pvv, nvv, gp, gn)
            for h in (2, 6):
                m, c = window_mean(x, h)
                self.f[f"{name}_W{h}"] = m
                if name == "v":
                    self.f[f"cnt_W{h}"] = c
            if name in ("v", "q"):
                sl, lv = window_slope(x, 4)
                self.f[f"{name}_lv4"] = lv
        self.ramps = ramps
        nT = nd * common.T
        self.cov = np.isfinite(v).mean(1).astype(np.float32)
        self.win_cond = np.zeros(nT, np.int8)
        self.win_pos = np.full(nT, -1, np.int16)
        for t0, cond in (windows or []):
            a, b = max(0, t0 + 1), min(nT, t0 + 19)
            if a < b:
                self.win_cond[a:b] = cond
                self.win_pos[a:b] = np.arange(a, b) - t0
        # same-time residual fields and their distance-kernel averages (v2)
        x = (net.length_km.cumsum() - net.length_km / 2).to_numpy(np.float64)
        self.x = x.astype(np.float32)
        if version >= 5:
            self.gapf = self.window_fields(net, windows or [])
        if version >= 6:
            self.rampf = self.ramp_fields(ramps, d["dates"][days], windows or [])
        if version >= 7:
            self.consf = self.conservation_fields(net)
        if version >= 8:
            self.thrf = self.gap_threshold_fields(net, windows or [], thr or [])
        dist = np.abs(x[:, None] - x[None, :])
        up = x[:, None] < x[None, :]  # column j sees row i upstream of it
        self.kern = {}
        res = {"rv": v - self.f["v_W2"], "rq": q - self.f["q_W2"], "rk": k - self.f["k_W2"]}
        kernels = {}
        for s in (0.5, 2.0, 8.0):
            W = np.exp(-dist / s)
            np.fill_diagonal(W, 0)
            kernels[f"s{s}"] = W
        W = np.ones((nl, nl)); np.fill_diagonal(W, 0)
        kernels["all"] = W
        Wu = np.exp(-dist / 2.0) * up; Wd = np.exp(-dist / 2.0) * up.T
        kernels["up2"] = Wu
        kernels["dn2"] = Wd
        self.res = res
        for name, r in res.items():
            M = np.isfinite(r)
            R = np.where(M, r, 0).astype(np.float64)
            Mf = M.astype(np.float64)
            for kn, W in kernels.items():
                num = R @ W
                den = Mf @ W
                with np.errstate(invalid="ignore", divide="ignore"):
                    self.kern[f"{name}_{kn}"] = np.where(den > 1e-3, num / den, np.nan).astype(np.float32)
                if name == "rv":
                    self.kern[f"wsum_{kn}"] = den.astype(np.float32)

    def features(self, ti: np.ndarray, li: np.ndarray, net: pd.DataFrame) -> pd.DataFrame:
        """Feature frame for cells (flat time index ti, link index li)."""
        F = {}
        nl = len(self.lanes)
        F["tod"] = self.tod[ti].astype(np.int16)
        F["wd"] = self.wd[ti].astype(np.int8)
        F["lanes"] = self.lanes[li]
        F["vf"] = net.free_speed_kmh.to_numpy(np.float32)[li]
        F["cap_l"] = (net.capacity_vph.to_numpy(np.float32) / self.lanes)[li]
        F["len"] = net.length_km.to_numpy(np.float32)[li]
        F["pos"] = (li / max(nl - 1, 1)).astype(np.float32)
        F["pv"] = self.pv[ti, li]
        F["pq"] = self.pq[ti, li]
        F["gp"] = np.minimum(self.f["gp"][ti, li], 999)
        F["gn"] = np.minimum(self.f["gn"][ti, li], 999)
        F["cnt_W2"] = self.f["cnt_W2"][ti, li]
        F["cnt_W6"] = self.f["cnt_W6"][ti, li]
        own_keys = ["v_prev", "v_next", "v_I", "v_W2", "v_W6", "v_lv4",
                    "q_prev", "q_next", "q_I", "q_W2", "q_W6", "q_lv4",
                    "k_I", "k_W2", "k_W6", "o_I", "o_W2"]
        for key in own_keys:
            F[key] = self.f[key][ti, li]
        bv = np.where(np.isfinite(F["v_I"]), F["v_I"], F["pv"])
        bq = np.where(np.isfinite(F["q_I"]), F["q_I"], F["pq"])
        F["base_v"] = bv.astype(np.float32)
        F["base_q"] = bq.astype(np.float32)
        for key in ("v_prev", "v_next", "v_W2", "v_W6", "v_lv4", "pv"):
            F[f"d_{key}"] = F[key] - bv
        for key in ("q_prev", "q_next", "q_W2", "q_W6", "q_lv4", "pq"):
            F[f"d_{key}"] = F[key] - bq
        for o in OFFSETS:
            lj = li + o
            okl = (lj >= 0) & (lj < nl)
            ljc = np.clip(lj, 0, nl - 1)

            def g(a):
                return np.where(okl, a[ti, ljc], np.nan).astype(np.float32)

            F[f"v_{o}"] = g(self.v)
            F[f"q_{o}"] = g(self.q)
            F[f"k_{o}"] = g(self.k)
            F[f"o_{o}"] = g(self.o)
            F[f"v_{o}_I"] = g(self.f["v_I"])
            F[f"q_{o}_I"] = g(self.f["q_I"])
            F[f"k_{o}_I"] = g(self.f["k_I"])
            F[f"v_{o}_W2"] = g(self.f["v_W2"])
            F[f"q_{o}_W2"] = g(self.f["q_W2"])
            F[f"rv_{o}"] = F[f"v_{o}"] - F[f"v_{o}_W2"]
            F[f"rq_{o}"] = F[f"q_{o}"] - F[f"q_{o}_W2"]
            F[f"rk_{o}"] = F[f"k_{o}"] - g(self.f["k_W2"])
        for key, a in self.kern.items():
            F[key] = a[ti, li]
        if self.version >= 3:
            F.update(self.advected(ti, li, F["base_v"]))
        if self.version >= 4:
            F["cov_t"] = self.cov[ti]
            F["win_cond"] = self.win_cond[ti]
            F["win_pos"] = self.win_pos[ti]
        if self.version >= 5:
            for key, a in self.gapf.items():
                F[key] = a[ti, li]
            F["g5_d_mf_v"] = F["g5_mf_v"] - F["base_v"]
            F["g5_d_mf_q"] = F["g5_mf_q"] - F["base_q"]
            F["g5_d_lin_v"] = F["g5_lin_v"] - F["base_v"]
            F["g7_d_mf_v"] = F["g7_mf_v"] - F["base_v"]
            F["g7_d_mf_q"] = F["g7_mf_q"] - F["base_q"]
        if self.version >= 6:
            for o in (-2, -1, 1, 2):
                lj = li + o
                okl = (lj >= 0) & (lj < nl)
                ljc = np.clip(lj, 0, nl - 1)
                for key in ("on", "off"):
                    F[f"g8_{key}_{o}"] = np.where(okl, self.rampf[f"g8_{key}"][ti, ljc], np.nan).astype(np.float32)
            for key, a in self.rampf.items():
                F[key] = a[ti, li]
        if self.version >= 7:
            for key, a in self.consf.items():
                F[key] = a[ti, li]
            F["g9_d_est"] = F["g9_est"] - F["base_q"]
            F["g9_d_est_up"] = F["g9_est_up"] - F["base_q"]
            F["g9_d_est_dn"] = F["g9_est_dn"] - F["base_q"]
        if self.version >= 8:
            for key, a in self.thrf.items():
                F[key] = a[ti, li]
        nT = self.v.shape[0]
        for o in (-2, -1, 1, 2):
            lj = li + o
            okl = (lj >= 0) & (lj < nl)
            ljc = np.clip(lj, 0, nl - 1)
            F[f"dist_{o}"] = np.where(okl, np.abs(self.x[ljc] - self.x[li]), np.nan).astype(np.float32)
            for dt in (-1, 1):
                tj = ti + dt
                okt = okl & (tj >= 0) & (tj < nT)
                tjc = np.clip(tj, 0, nT - 1)
                F[f"v_{o}_t{dt}"] = np.where(okt, self.v[tjc, ljc], np.nan).astype(np.float32)
                F[f"q_{o}_t{dt}"] = np.where(okt, self.q[tjc, ljc], np.nan).astype(np.float32)
        return pd.DataFrame(F)


    def slot_field(self, s: int):
        """Speed, per-lane flow and density of every link at flat slot s: the observation,
        else the mean of the observed slots within +-2 (one-sided next to a gap), else
        linear in milepost between observed links, else the weekday profile."""
        out = []
        for name, a in (("v", self.v), ("q", self.q)):
            f = a[s].astype(np.float64).copy()
            m = ~np.isfinite(f)
            f[m] = self.f[f"{name}_W2"][s][m]
            m = np.isfinite(f)
            if 0 < m.sum() < len(f):
                f[~m] = np.interp(self.x[~m], self.x[m], f[m])
            elif m.sum() == 0:
                f = (self.pv if name == "v" else self.pq)[s].astype(np.float64)
            out.append(f)
        v, q = out
        return v, q, q / np.clip(v, 1, None)

    def window_fields(self, net: pd.DataFrame, windows, step=0.2, vcut=0.6):
        """Task 2 gap features (v5) on the slots T+5..T+90 of each window.

        tau = (t-T)/95 between the field at T and at T+95.
        Queue polygon: the main queue run (speed <= vcut * free speed, 1-link holes
        merged, largest speed deficit) at T and at T+95, its ends joined linearly in tau;
        inside flag, signed distance to the nearer end (km, > 0 inside), relative position.
        Moving frame: shift dx of the speed-deficit profile (below 0.8 free speed) from T to T+95 by
        cross-correlation, then v(x, tau) = (1-tau) v_T(x - tau dx) + tau v_T95(x + (1-tau) dx),
        same for flow and density."""
        nT, nl = self.v.shape
        keys = ("g5_tau", "g5_lin_v", "g5_lin_q", "g5_lin_k", "g5_mf_v", "g5_mf_q", "g5_mf_k", "g5_dx",
                "g5_corr", "g5_pstate", "g5_in", "g5_sd", "g5_rel", "g5_plen", "g5_sd0", "g5_sd2",
                "g6_in_any", "g6_sd_any", "g6_rel_any", "g7_mf_v", "g7_mf_q", "g7_mf_k")
        G = {k: np.full((nT, nl), np.nan, np.float32) for k in keys}
        x = self.x.astype(np.float64)
        ln = net.length_km.to_numpy(np.float64)
        vf = net.free_speed_kmh.to_numpy(np.float64)
        lo_e, hi_e = x[0] - ln[0] / 2, x[-1] + ln[-1] / 2
        grid = np.arange(lo_e, hi_e + step / 2, step)
        smax = int(min(30.0, 0.6 * (hi_e - lo_e)) / step)

        def queue_run(v):
            dv = np.clip(vf - v, 0, None)
            c = v <= vcut * vf
            if not c.any():
                return None
            idx = np.nonzero(c)[0]
            runs, a = [], idx[0]
            for i, j in zip(idx[:-1], idx[1:]):
                if j - i > 2:
                    runs.append((a, i)); a = j
            runs.append((a, idx[-1]))
            a, b = max(runs, key=lambda r: dv[r[0]:r[1] + 1].sum())
            return x[a] - ln[a] / 2, x[b] + ln[b] / 2

        for t0, _ in windows:
            if t0 < 0 or t0 + 19 >= nT:
                continue
            v0, q0, k0 = self.slot_field(t0)
            v2, q2, k2 = self.slot_field(t0 + 19)
            a = np.interp(grid, x, np.clip(0.8 * vf - v0, 0, None))
            b = np.interp(grid, x, np.clip(0.8 * vf - v2, 0, None))
            dx, corr = 0.0, np.nan
            if a.max() >= 10 and b.max() >= 10:
                c = np.correlate(b, a, mode="full")  # c[k + n - 1] = sum a[i] b[i + k]
                n = len(a)
                lags = np.arange(-smax, smax + 1)
                cc = c[lags + n - 1]
                kb = lags[int(np.argmax(cc))]
                dx = kb * step
                corr = cc.max() / np.sqrt((a * a).sum() * (b * b).sum())
            r0, r2 = queue_run(v0), queue_run(v2)
            kp = [self.pq[t] / np.clip(self.pv[t], 1, None) for t in (t0, t0 + 19)]
            pstate = (r0 is not None) + 2 * (r2 is not None)
            for p in range(1, 19):
                s = t0 + p
                tau = p / 19.0
                G["g5_tau"][s] = tau
                G["g5_lin_v"][s] = (1 - tau) * v0 + tau * v2
                G["g5_lin_q"][s] = (1 - tau) * q0 + tau * q2
                G["g5_lin_k"][s] = (1 - tau) * k0 + tau * k2
                xa, xb = x - tau * dx, x + (1 - tau) * dx
                for nm, f0, f2 in (("v", v0, v2), ("q", q0, q2), ("k", k0, k2)):
                    G[f"g5_mf_{nm}"][s] = (1 - tau) * np.interp(xa, x, f0) + tau * np.interp(xb, x, f2)
                # g7: the same moving frame on deviations from a reference (free speed for v,
                # weekday profile for q and k), the own link's reference added back
                for nm, f0, f2, r0_, r2_, rs in (("v", v0, v2, vf, vf, vf),
                                                 ("q", q0, q2, self.pq[t0], self.pq[t0 + 19], self.pq[s]),
                                                 ("k", k0, k2, kp[0], kp[1], self.pq[s] / np.clip(self.pv[s], 1, None))):
                    G[f"g7_mf_{nm}"][s] = rs + (1 - tau) * np.interp(xa, x, f0 - r0_) + tau * np.interp(xb, x, f2 - r2_)
                G["g5_dx"][s] = dx
                G["g5_corr"][s] = corr
                G["g5_pstate"][s] = pstate
                for r, key in ((r0, "g5_sd0"), (r2, "g5_sd2")):
                    if r is not None:
                        G[key][s] = np.minimum(x - r[0], r[1] - x)
                # one-sided cases: onset triangle grows upstream from the T+95 queue head,
                # dissipation triangle recovers from the T queue head upstream
                tri = None
                if r0 is None and r2 is not None:
                    tri = (r2[1] - tau * (r2[1] - r2[0]), r2[1])
                elif r0 is not None and r2 is None:
                    tri = (r0[0], r0[1] - tau * (r0[1] - r0[0]))
                elif r0 is not None:
                    tri = ((1 - tau) * r0[0] + tau * r2[0], (1 - tau) * r0[1] + tau * r2[1])
                if tri is not None:
                    G["g6_in_any"][s] = (x >= tri[0]) & (x <= tri[1])
                    G["g6_sd_any"][s] = np.minimum(x - tri[0], tri[1] - x)
                    G["g6_rel_any"][s] = (x - tri[0]) / max(tri[1] - tri[0], 1e-3)
                if r0 is not None and r2 is not None:
                    up = (1 - tau) * r0[0] + tau * r2[0]
                    dn = (1 - tau) * r0[1] + tau * r2[1]
                    G["g5_in"][s] = (x >= up) & (x <= dn)
                    G["g5_sd"][s] = np.minimum(x - up, dn - x)
                    G["g5_rel"][s] = (x - up) / max(dn - up, 1e-3)
                    G["g5_plen"][s] = dn - up
        return G

    def ramp_fields(self, ramps: dict, dates, windows):
        """Ramp flow features (v6), on every slot.

        ramps: flow (days, 288, n_ramps) with NaN where not observed, li (attached link),
        on (on-ramp flag), prof (7, 288, n_ramps) weekday x slot profile. A missing ramp
        value is the mean of its observed slots within +-2, else the profile.
        Per link: on / off sums at t, their 3-slot means, observed share, ratio to the
        profile; cumulative net inflow sum_{j <= l} (on - off) and its gap from the
        profile. Inside Task 2 gaps also the change since T and the accumulated
        change of the cumulative net inflow (vehicles, sum over the gap slots so far)."""
        nT, nl = self.v.shape
        f = ramps["flow"].reshape(nT, -1).astype(np.float32)
        nr = f.shape[1]
        wd = common.slot_weekday(dates)
        prof = ramps["prof"][wd].reshape(nT, nr)
        obs = np.isfinite(f)
        w2, _ = window_mean(f, 2)
        ff = np.where(obs, f, np.where(np.isfinite(w2), w2, prof))
        A = np.zeros((nr, nl), np.float32)
        ok = ramps["li"] >= 0
        A[np.nonzero(ok)[0], ramps["li"][ok]] = 1
        on = ramps["on"].astype(bool)
        Aon, Aoff = A * on[:, None], A * ~on[:, None]
        n_on, n_off = Aon.sum(0), Aoff.sum(0)
        G = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for key, M, n in (("on", Aon, n_on), ("off", Aoff, n_off)):
                s = ff @ M
                p = prof @ M
                if key == "on":
                    self.ramp_on_prof = p
                G[f"g8_{key}"] = s
                sm = s.copy()
                sm[1:-1] = (s[:-2] + s[1:-1] + s[2:]) / 3
                G[f"g8_{key}_s3"] = sm
                G[f"g8_{key}_obs"] = np.where(n > 0, (obs.astype(np.float32) @ M) / n, np.nan)
                G[f"g8_{key}_r"] = np.where(p > 1, s / p, np.nan)
            net_in = ff @ (Aon - Aoff)
            cum = np.cumsum(net_in, axis=1)
            cum_p = np.cumsum(prof @ (Aon - Aoff), axis=1)
            G["g8_cum"] = cum
            cs = cum.copy()
            cs[1:-1] = (cum[:-2] + cum[1:-1] + cum[2:]) / 3
            G["g8_cum_s3"] = cs
            G["g8_cum_dp"] = cum - cum_p
            G["g8_cum_obs"] = np.cumsum(obs.astype(np.float32) @ A, axis=1) / np.maximum(np.cumsum(A.sum(0)), 1)
        self.ramp_obs_count = (obs.astype(np.float32) @ A).astype(np.float64)
        self.ramp_count = A.sum(0).astype(np.float64)
        for key in ("g8_d_on", "g8_d_off", "g8_d_cum", "g8_acc"):
            G[key] = np.full((nT, nl), np.nan, np.float32)
        for t0, _ in windows:
            if t0 < 0 or t0 + 19 > nT:
                continue
            sl = slice(t0 + 1, t0 + 19)
            G["g8_d_on"][sl] = G["g8_on"][sl] - G["g8_on"][t0]
            G["g8_d_off"][sl] = G["g8_off"][sl] - G["g8_off"][t0]
            dc = cum[sl] - cum[t0]
            G["g8_d_cum"][sl] = dc
            G["g8_acc"][sl] = np.cumsum(dc, axis=0) * common.DT_H
        return {k: a.astype(np.float32) for k, a in G.items()}

    def conservation_fields(self, net: pd.DataFrame):
        """Mainline flow conservation through the ramps (v7), every slot.

        Flow on link l = flow on link l-1 + on-ramp - off-ramp flow attached to l. From
        the nearest observed links u < l and d > l at the same slot:
            up estimate = Q_u + sum_{u < j <= l} (on_j - off_j)
            down estimate = Q_d - sum_{l < j <= d} (on_j - off_j)
        per lane of l; their mean and difference, the link distances to u and d, and the
        observed share of the ramps attached in (u, d]. Ramp gaps are filled as in
        ramp_fields (+-2 slot mean, then profile), never zero."""
        nT, nl = self.q.shape
        lanes = self.lanes.astype(np.float64)
        Q = self.q.astype(np.float64) * lanes[None, :]
        C = np.cumsum((self.rampf["g8_on"] - self.rampf["g8_off"]).astype(np.float64), axis=1)
        idx = np.arange(nl)
        ok = np.isfinite(Q)
        pu = np.maximum.accumulate(np.where(ok, idx, -1), axis=1)
        up = np.full((nT, nl), -1)
        up[:, 1:] = pu[:, :-1]
        pd_ = np.minimum.accumulate(np.where(ok, idx, nl)[:, ::-1], axis=1)[:, ::-1]
        dn = np.full((nT, nl), nl)
        dn[:, :-1] = pd_[:, 1:]
        rows = np.arange(nT)[:, None]
        hu, hd = up >= 0, dn < nl
        upc, dnc = np.clip(up, 0, nl - 1), np.clip(dn, 0, nl - 1)
        est_up = np.where(hu, (Q[rows, upc] + C - C[rows, upc]) / lanes, np.nan)
        est_dn = np.where(hd, (Q[rows, dnc] - (C[rows, dnc] - C)) / lanes, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            est = np.where(hu & hd, (est_up + est_dn) / 2, np.where(hu, est_up, est_dn))
            # observed share of the ramps attached in (u, d]; u = -1 / d = nl mean the corridor end
            O = np.cumsum(self.ramp_obs_count, axis=1)
            A = np.cumsum(np.broadcast_to(self.ramp_count, (nT, nl)), axis=1)
            o_lo, a_lo = np.where(hu, O[rows, upc], 0), np.where(hu, A[rows, upc], 0)
            o_hi, a_hi = O[rows, dnc], A[rows, dnc]
            share = np.where(a_hi - a_lo > 0, (o_hi - o_lo) / (a_hi - a_lo), np.nan)
        G = {"g9_q_up": np.where(hu, Q[rows, upc] / lanes, np.nan), "g9_q_dn": np.where(hd, Q[rows, dnc] / lanes, np.nan),
             "g9_net_up": np.where(hu, C - C[rows, upc], np.nan), "g9_net_dn": np.where(hd, C[rows, dnc] - C, np.nan),
             "g9_est_up": est_up, "g9_est_dn": est_dn, "g9_est": est, "g9_est_diff": est_up - est_dn,
             "g9_lu": np.where(hu, idx[None, :] - up, np.nan), "g9_ld": np.where(hd, dn - idx[None, :], np.nan),
             "g9_ramp_obs": share}
        return {k: np.asarray(a, np.float32) for k, a in G.items()}

    def gap_threshold_fields(self, net: pd.DataFrame, windows, thr, w_kmh=18.0):
        """Bottleneck threshold features inside the Task 2 gaps (v8, g10_*).

        thr: [(first link, last link, reference link, c_h per lane)] bottleneck groups fitted
        on train (fit_bottlenecks). For a gap slot s of the window at T (T2 = T+95):
          q_h(T)   causal mean of the observed per-lane flow at the group's reference link
                   over T-10..T (else the last value within an hour); q_h(T2) the same forward
          Qhat_h(s) = q_h(T) * on-ramp total(s) / on-ramp total(T)  (ramps are observed in the
                   gap; their total follows the demand)
        per cell (link l, slot s), for the nearest group at or downstream of l:
          g10_dist links to the group, g10_rT q_h(T)/c, g10_rT2 q_h(T2)/c, g10_rhat Qhat/c,
          g10_rlin ((1-tau) q_h(T) + tau q_h(T2))/c, g10_rmax running max of rhat since T,
          g10_since slots since rhat first reached 1 (NaN if not yet), g10_up km upstream of
          the group's downstream end, g10_wedge = w*since - up (> 0 inside a wedge growing
          upstream at w km/h from the crossing), g10_in = wedge > 0 (or rT >= 1 and inside)."""
        nT, nl = self.v.shape
        keys = ("g10_dist", "g10_rT", "g10_rT2", "g10_rhat", "g10_rlin", "g10_rmax", "g10_since", "g10_up",
                "g10_wedge", "g10_in")
        G = {k: np.full((nT, nl), np.nan, np.float32) for k in keys}
        if not thr:
            return G
        x = self.x.astype(np.float64)
        ln = net.length_km.to_numpy(np.float64)
        on = self.rampf["g8_on"].sum(1).astype(np.float64)
        # nearest group at or downstream of each link
        near = np.full(nl, -1)
        dist = np.full(nl, np.nan)
        for l in range(nl):
            cand = [(max(a - l, 0), k) for k, (a, b, h, c) in enumerate(thr) if b >= l]
            if cand:
                dd, k = min(cand)
                near[l], dist[l] = k, dd
        has = near >= 0
        kk = np.clip(near, 0, None)
        x_end = np.array([x[b] + ln[b] / 2 for a, b, h, c in thr])
        cvec = np.array([c for a, b, h, c in thr])

        def side_mean(t, h, forward):
            for span in (2, 12):
                sl = slice(t, min(nT, t + span + 1)) if forward else slice(max(0, t - span), t + 1)
                q = self.q[sl, h]
                q = q[np.isfinite(q)]
                if len(q):
                    return float(q[:3].mean() if forward else q[-3:].mean())
            return np.nan

        for t0, _ in windows:
            if t0 < 0 or t0 + 19 >= nT:
                continue
            qT = np.array([side_mean(t0, h, False) for a, b, h, c in thr])
            q2 = np.array([side_mean(t0 + 19, h, True) for a, b, h, c in thr])
            rT, r2 = qT / cvec, q2 / cvec
            rmax = np.where(np.isfinite(rT), rT, -np.inf)
            since = np.full(len(thr), np.nan)
            for p in range(1, 19):
                s = t0 + p
                tau = p / 19.0
                rhat = rT * (on[s] / on[t0] if on[t0] > 0 else 1.0)
                rlin = (1 - tau) * rT + tau * r2
                rmax = np.maximum(rmax, np.nan_to_num(rhat, nan=-np.inf))
                cross = (rhat >= 1) & np.isnan(since)
                since[cross] = 0
                since[~np.isnan(since) & ~cross] += 1
                since_eff = np.where(rT >= 1, p, since)  # already above the threshold at T
                up = x_end[kk] - x
                wedge = w_kmh * since_eff[kk] * common.DT_H - up
                G["g10_dist"][s] = np.where(has, dist, np.nan)
                G["g10_rT"][s] = np.where(has, rT[kk], np.nan)
                G["g10_rT2"][s] = np.where(has, r2[kk], np.nan)
                G["g10_rhat"][s] = np.where(has, rhat[kk], np.nan)
                G["g10_rlin"][s] = np.where(has, rlin[kk], np.nan)
                G["g10_rmax"][s] = np.where(has, np.where(np.isfinite(rmax[kk]), rmax[kk], np.nan), np.nan)
                G["g10_since"][s] = np.where(has, since_eff[kk], np.nan)
                G["g10_up"][s] = np.where(has, up, np.nan)
                G["g10_wedge"][s] = np.where(has, wedge, np.nan)
                G["g10_in"][s] = np.where(has, np.nan_to_num(wedge, nan=-1) > 0, np.nan)
        return G

    def advected(self, ti, li, speed, scale=1.5, chunk=20000):
        """Flow/density residuals along the free-flow characteristic.

        Flow fluctuations travel downstream at the traffic speed, so the platoon
        at (x, t) was near x - v*dt at t-1 and will be near x + v*dt at t+1.
        Kernel-weighted residual means at those points, for lags 1 and 2.
        """
        out = {}
        nT, nl = self.v.shape
        x = self.x.astype(np.float64)
        D = np.clip(speed, 5, 130).astype(np.float64) * common.DT_H
        for name in ("rq", "rk"):
            r = self.res[name]
            M = np.isfinite(r)
            R = np.where(M, r, 0).astype(np.float32)
            for lag in (-2, -1, 1, 2):
                num = np.full(len(ti), np.nan, np.float32)
                wsum = np.zeros(len(ti), np.float32)
                for s in range(0, len(ti), chunk):
                    tt = ti[s:s + chunk] + lag
                    ok = (tt >= 0) & (tt < nT)
                    ttc = np.clip(tt, 0, nT - 1)
                    centre = x[li[s:s + chunk]] + lag * D[s:s + chunk]
                    W = np.exp(-np.abs(x[None, :] - centre[:, None]) / scale).astype(np.float32)
                    W[np.arange(len(tt)), li[s:s + chunk]] = 0
                    nu = (W * R[ttc]).sum(1)
                    de = (W * M[ttc]).sum(1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        num[s:s + chunk] = np.where(ok & (de > 1e-3), nu / de, np.nan)
                    wsum[s:s + chunk] = np.where(ok, de, 0)
                out[f"adv_{name}_{lag}"] = num
                if name == "rq":
                    out[f"advw_{lag}"] = wsum
        return out
