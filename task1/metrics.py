"""Array-based scorers for Task 1 and a Task 3 proxy on train.

S_state follows score_task1.py exactly. S_FD follows score_task3.py exactly.
S_LWR uses the observation-matched boundary flux: the organizer flux is
projected onto the released observations, so the right-hand side of the
conservation check equals the observed dN, and

    S_LWR ~= 1 - sum|dN_sub - dN_obs| / sum|dN_obs|

over the transitions score_task3.py keeps (consecutive eligible cells of the
regime's days, attached ramps valid at the interval start).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

sys.path.insert(0, str(common.SRC))
from task3.score_task3 import load_ramp_flows, network_parameters  # noqa: E402
from task1.baseline_task1_historical_mean import station_lanes  # noqa: E402

SPEED_N, FLOW_N, W_SPEED, W_FLOW = 25.0, 600.0, 0.54, 0.46


def panel_params(panel: str, links: np.ndarray) -> dict:
    pdir = common.REL / "corridors" / panel
    p = network_parameters(pdir).set_index("link_id").loc[list(links)]
    lanes_map = station_lanes(pdir)
    score_lanes = np.array([lanes_map.get(l, 1.0) for l in links], float).clip(min=1.0)
    return {
        "length": p.length_km.to_numpy(float),
        "lanes": p.lanes.clip(lower=1.0).to_numpy(float),
        "vf": p.free_speed_kmh.to_numpy(float),
        "cap": p.capacity_vph.to_numpy(float),
        "kcrit": p.critical_density.to_numpy(float),
        "kjam": p.k_jam.to_numpy(float),
        "score_lanes": score_lanes,
    }


def ramp_valid(panel: str, split: str, dates: np.ndarray, links: np.ndarray):
    """Boolean (nd, T, nl) arrays: on/off ramp validity as score_task3 derives it."""
    cp = common.CACHE / f"{panel}_{split}_ramp_{common.CACHE_VERSION}.npz"
    if cp.exists():
        z = np.load(cp)
        return z["on"], z["off"]
    pdir = common.REL / "corridors" / panel
    rf = load_ramp_flows(pdir, split)
    nd, nl = len(dates), len(links)
    on = np.zeros((nd, common.T, nl), bool)
    off = np.zeros((nd, common.T, nl), bool)
    if len(rf):
        ts = rf.timestamp.dt.strftime("%Y-%m-%d")
        day_of = {d: i for i, d in enumerate(dates)}
        di = ts.map(day_of)
        link_of = {l: i for i, l in enumerate(links)}
        li = rf.link_id.map(link_of)
        ok = di.notna() & li.notna()
        rf = rf[ok]
        di = di[ok].astype(int).to_numpy()
        li = li[ok].astype(int).to_numpy()
        ti = (rf.timestamp.dt.hour * 12 + rf.timestamp.dt.minute // 5).to_numpy()
        on[di, ti, li] = rf.on_ramp_valid.to_numpy() > 0
        off[di, ti, li] = rf.off_ramp_valid.to_numpy() > 0
    topo = pd.read_csv(pdir / "network" / "lwr_mainline_topology.csv", dtype=str)
    on_att = set(topo.loc[topo.on_ramp_link_ids.fillna("").str.len() > 0, "link_id"])
    off_att = set(topo.loc[topo.off_ramp_link_ids.fillna("").str.len() > 0, "link_id"])
    on |= ~np.isin(links, list(on_att))[None, None, :]
    off |= ~np.isin(links, list(off_att))[None, None, :]
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cp, on=on, off=off)
    return on, off


class Scorer:
    """Scores predictions for a set of days of one panel's train split."""

    def __init__(self, panel: str, data: dict, days=None):
        self.panel = panel
        self.d = data
        nd = len(data["dates"])
        self.days = np.arange(nd) if days is None else np.asarray(days)
        self.p = panel_params(panel, data["links"])
        on, off = ramp_valid(panel, "train", data["dates"], data["links"])
        self.ramp_ok = (on & off)[self.days]
        self.v = data["true_speed"][self.days].astype(np.float64)
        self.q = data["true_flow"][self.days].astype(np.float64)
        self.el = data["elig"][self.days]
        self.tg = data["target"][self.days]
        self.reg = data["regime"][self.days]
        dts = pd.to_datetime(pd.Series(data["dates"][self.days])).to_numpy()
        self.dates = dts

    def score(self, ps: np.ndarray, pq: np.ndarray, physics: bool = True) -> pd.DataFrame:
        """ps, pq: (n_sel_days, T, nl) predictions, used only at target cells."""
        rows = []
        P = self.p
        for r in common.REGIMES:
            sel = np.where(self.reg == r)[0]
            if len(sel) == 0:
                continue
            tg = self.tg[sel]
            ds = (ps[sel] - self.v[sel])[tg]
            lanes = np.broadcast_to(P["score_lanes"], tg.shape)[tg]
            dq = ((pq[sel] - self.q[sel])[tg]) / lanes
            rmse_s = float(np.sqrt(np.mean(ds * ds)))
            rmse_q = float(np.sqrt(np.mean(dq * dq)))
            s_speed = max(0.0, 1 - rmse_s / SPEED_N)
            s_flow = max(0.0, 1 - rmse_q / FLOW_N)
            row = {"panel": self.panel, "regime": r, "n_days": len(sel), "n_target": int(tg.sum()),
                   "rmse_speed": rmse_s, "rmse_flow_lane": rmse_q,
                   "S_state": W_SPEED * s_speed + W_FLOW * s_flow}
            if physics:
                row.update(self.physics(sel, ps[sel], pq[sel]))
            rows.append(row)
        return pd.DataFrame(rows)

    def physics(self, sel, ps, pq) -> dict:
        P = self.p
        el, tg = self.el[sel], self.tg[sel]
        v = np.where(tg, ps, self.v[sel])
        q = np.where(tg, pq, self.q[sel])
        # FD on all eligible cells, per lane
        lanes = P["lanes"]
        k = q / np.clip(v, 1.0, None)
        kl = k / lanes
        ql = q / lanes
        capl, kcl, kjl = P["cap"] / lanes, P["kcrit"] / lanes, P["kjam"] / lanes
        qfd = np.where(kl <= kcl, P["vf"] * kl, (capl / np.maximum(kjl - kcl, 1e-9)) * np.maximum(kjl - kl, 0))
        s_fd = max(0.0, 1 - np.abs(ql - qfd)[el].sum() / (np.abs(ql)[el].sum() + 1e-9))
        if (q[tg & el] < 50).mean() > 0.2:
            s_fd = 0.0
        # LWR: N = k L, forward transitions within the regime's day sequence
        N = k * P["length"]
        k0 = self.v[sel]
        N0 = self.q[sel] / np.clip(k0, 1.0, None) * P["length"]
        ro = self.ramp_ok[sel]
        nd = len(sel)
        # flatten time across the selected days; link consecutive days
        Nf = N.reshape(nd * common.T, -1)
        N0f = N0.reshape(nd * common.T, -1)
        ef = el.reshape(nd * common.T, -1)
        rof = ro.reshape(nd * common.T, -1)
        cont = np.ones(nd * common.T - 1, bool)
        dates = self.dates[sel]
        gap = (np.diff(dates) != np.timedelta64(1, "D"))
        idx = np.arange(1, nd) * common.T - 1
        cont[idx[gap]] = False
        ok = ef[:-1] & ef[1:] & rof[:-1] & cont[:, None]
        dn = Nf[1:] - Nf[:-1]
        dn0 = N0f[1:] - N0f[:-1]
        num = np.abs(dn - dn0)[ok].sum()
        den = np.abs(dn0)[ok].sum()
        s_lwr = max(0.0, 1 - min(1.0, num / (den + 1e-9)))
        return {"S_FD": s_fd, "S_LWR": s_lwr, "S_physics": (s_fd + 2 * s_lwr) / 3}


def summarize(df: pd.DataFrame) -> dict:
    """panel mean over regimes -> family mean -> overall."""
    cols = [c for c in ["S_state", "S_FD", "S_LWR", "S_physics"] if c in df]
    pm = df.groupby("panel")[cols].mean()
    pm["family"] = pm.index.map(common.FAMILY)
    fm = pm.groupby("family")[cols].mean()
    return {c: float(fm[c].mean()) for c in cols}


def contribution(s: dict) -> float:
    """Task 1 + Task 3 contribution to S_total."""
    return 0.35 * s.get("S_state", 0) + 0.15 * s.get("S_physics", 0)
