"""Shared loaders for Task 1/3 work.

Every daily parquet file holds 288 timestamps x n_links rows sorted by
(timestamp, link_id), one station per link, and the masked and unmasked layers
are row-aligned. A panel/split is therefore loaded into dense arrays of shape
(n_days, 288, n_links) with links reordered upstream -> downstream by the
topology order_index.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

KAGGLE = Path(__file__).resolve().parent.parent
# Data and code roots can be overridden so the same code runs on the Mac and on GB10.
REL = Path(os.environ.get("TFB_REL", KAGGLE / "data" / "kaggle_public"))
SRC = Path(os.environ.get("TFB_SRC", KAGGLE / "trafficflowbench-public" / "src"))
CACHE = Path(os.environ.get("TFB_CACHE", Path(__file__).resolve().parent / "cache"))
THREADS = int(os.environ.get("TFB_THREADS", 3))
CACHE_VERSION = "v3"  # v3: milepost order (v2 reversed the S/W corridors by mistake)
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S",
          "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]
FAMILY = {p: p.rsplit("_", 1)[0] for p in PANELS}
REGIMES = ("R1", "R2", "R3")
T = 288
DT_H = 5.0 / 60.0

COLS = ["speed_kmh", "flow_vph", "occupancy", "density_occ_linear_vehpkm"]


def downstream_decreasing(panel: str) -> bool:
    """True when the topology's outgoing links sit at lower order_index (S/W corridors)."""
    nd = REL / "corridors" / panel / "network"
    topo = pd.read_csv(nd / "lwr_mainline_topology.csv", dtype={"link_id": str})
    pos = dict(zip(topo.link_id, topo.order_index))
    s = [np.sign(pos[x] - pos[l]) for l, o in zip(topo.link_id, topo.outgoing_link_ids.fillna(""))
         for x in o.split(";") if x in pos]
    return bool(np.mean(s) < 0) if s else False


def network(panel: str) -> pd.DataFrame:
    """Per-link parameters in topology order (upstream first)."""
    nd = REL / "corridors" / panel / "network"
    topo = pd.read_csv(nd / "lwr_mainline_topology.csv", dtype={"link_id": str})
    fd = pd.read_csv(nd / "fd_parameters.csv", dtype={"link_id": str})
    links = pd.read_csv(nd / "links.csv", dtype={"link_id": str}).drop_duplicates("link_id")
    net = topo[["link_id", "order_index", "incoming_link_ids", "outgoing_link_ids",
                "on_ramp_link_ids", "off_ramp_link_ids"]].merge(
        fd[["link_id", "lanes", "free_speed_kmh", "capacity_vph", "k_jam"]], on="link_id", how="left")
    net = net.merge(links[["link_id", "length_km"]], on="link_id", how="left")
    # order_index follows milepost. The generator's traffic runs toward increasing
    # milepost on every corridor (flow fluctuations advect that way, checked on
    # train), even where lwr_mainline_topology.csv lists outgoing links the other
    # way (the S/W corridors), so milepost order is upstream -> downstream.
    net = net.sort_values("order_index").reset_index(drop=True)
    net["critical_density"] = net.capacity_vph / net.free_speed_kmh.clip(lower=1)
    return net


def _day_files(panel: str, split: str):
    base = REL / "corridors" / panel / split
    masked = sorted(glob.glob(str(base / "mainline_states_masked" / "*" / "*.parquet")))
    masked = sorted(masked, key=os.path.basename)
    truth = {os.path.basename(p): p for p in glob.glob(str(base / "mainline_states" / "*" / "*.parquet"))}
    return masked, truth


def load(panel: str, split: str, use_cache: bool = True) -> dict:
    """Dense arrays for one panel and split.

    Keys: dates, regime (per day), links, obs_<col> (masked layer, NaN where
    blank), pct, elig, target (template cells), and on train true_speed,
    true_flow (unmasked layer).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"{panel}_{split}_{CACHE_VERSION}.npz"
    if use_cache and cp.exists():
        z = np.load(cp, allow_pickle=True)
        return {k: z[k] for k in z.files}
    net = network(panel)
    masked, truth = _day_files(panel, split)
    first = pq.read_table(masked[0], columns=["link_id"]).column(0).to_pylist()
    file_links = first[: len(set(first))]
    nl = len(file_links)
    pos = {l: i for i, l in enumerate(file_links)}
    perm = np.array([pos[l] for l in net.link_id])  # topology order -> file column
    nd = len(masked)
    out = {f"obs_{c}": np.full((nd, T, nl), np.nan, np.float32) for c in COLS}
    out["pct"] = np.zeros((nd, T, nl), np.int8)
    out["elig"] = np.zeros((nd, T, nl), bool)
    has_truth = bool(truth)
    if has_truth:
        out["true_speed"] = np.full((nd, T, nl), np.nan, np.float32)
        out["true_flow"] = np.full((nd, T, nl), np.nan, np.float32)
    dates, regimes = [], []
    for d, mp in enumerate(masked):
        tb = pq.read_table(mp, columns=["date", "link_id", "pct_observed", "is_score_eligible", "mask_regime"] + COLS)
        assert tb.num_rows == T * nl, (mp, tb.num_rows)
        lk = np.asarray(tb.column("link_id").to_pylist()[:nl])
        assert (lk == np.asarray(file_links)).all(), mp
        dates.append(tb.column("date")[0].as_py())
        regimes.append(tb.column("mask_regime")[0].as_py())
        for c in COLS:
            out[f"obs_{c}"][d] = tb.column(c).to_numpy(zero_copy_only=False).reshape(T, nl)[:, perm]
        out["pct"][d] = tb.column("pct_observed").to_numpy().reshape(T, nl)[:, perm]
        out["elig"][d] = tb.column("is_score_eligible").to_numpy().reshape(T, nl)[:, perm] == 1
        if has_truth:
            tt = pq.read_table(truth[os.path.basename(mp)], columns=["speed_kmh", "flow_vph"])
            out["true_speed"][d] = tt.column(0).to_numpy(zero_copy_only=False).reshape(T, nl)[:, perm]
            out["true_flow"][d] = tt.column(1).to_numpy(zero_copy_only=False).reshape(T, nl)[:, perm]
    out["dates"] = np.asarray(dates)
    out["regime"] = np.asarray(regimes)
    out["links"] = net.link_id.to_numpy().astype(str)
    out["target"] = template_mask(panel, split, out["dates"], out["links"])
    np.savez(cp, **out)
    return out


def template_path(panel: str, split: str) -> Path:
    return REL / "task1" / panel / split / "sample_submission_state.csv"


def read_template(panel: str, split: str) -> pd.DataFrame:
    tb = pacsv.read_csv(template_path(panel, split),
                        convert_options=pacsv.ConvertOptions(column_types={c: "string" for c in
                                                                           ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]}))
    return tb.select(["panel", "timestamp", "station_id", "link_id", "mask_regime"]).to_pandas()


def template_index(tpl: pd.DataFrame, dates: np.ndarray, links: np.ndarray):
    """(day, slot, link) index arrays for every template row."""
    ts = tpl.timestamp.str
    day_of = {d: i for i, d in enumerate(dates)}
    di = ts.slice(0, 10).map(day_of).to_numpy()
    hh = ts.slice(11, 13).astype(int).to_numpy()
    mm = ts.slice(14, 16).astype(int).to_numpy()
    ti = hh * 12 + mm // 5
    link_of = {l: i for i, l in enumerate(links)}
    li = tpl.link_id.map(link_of).to_numpy()
    return di.astype(np.int64), ti.astype(np.int64), li.astype(np.int64)


def template_mask(panel: str, split: str, dates: np.ndarray, links: np.ndarray) -> np.ndarray:
    tpl = read_template(panel, split)
    di, ti, li = template_index(tpl, dates, links)
    m = np.zeros((len(dates), T, len(links)), bool)
    m[di, ti, li] = True
    return m


def slot_weekday(dates: np.ndarray) -> np.ndarray:
    return pd.to_datetime(pd.Series(dates)).dt.weekday.to_numpy()


COND = {"queue_onset": 1, "queue_ongoing": 2}


def split_windows(panel: str, split: str, dates: np.ndarray):
    """Task 2 windows of a scored split as (day, slot, condition code)."""
    f = REL / "task2" / panel / split / "window_index.csv"
    if not f.exists():
        return []
    w = pd.read_csv(f, dtype=str)
    day_of = {d: i for i, d in enumerate(dates)}
    out = []
    for fo, cond in zip(w.forecast_origin, w.condition):
        di = day_of.get(fo[:10])
        if di is None:
            continue
        out.append((di, int(fo[11:13]) * 12 + int(fo[14:16]) // 5, COND[cond]))
    return out


def load_ramps(panel: str, split: str, dates: np.ndarray, links: np.ndarray, use_cache: bool = True) -> dict:
    """Ramp flows of one panel/split aligned with the mainline days.

    flow: (n_days, 288, n_ramps) vph, NaN where not observed (missing is not zero).
    li: mainline link index (topology order) each ramp attaches to, -1 if none;
    on: True for on-ramps (OR), False for off-ramps (FR).
    Day files are read one at a time (reading the partitioned directory fails on a
    mask_regime type clash)."""
    cp = CACHE / f"{panel}_{split}_ramp_v1.npz"
    if use_cache and cp.exists():
        z = np.load(cp, allow_pickle=True)
        return {k: z[k] for k in z.files}
    amap = pd.read_csv(REL / "corridors" / panel / "network" / "ramp_attachment_map.csv", dtype=str)
    rid = {r: i for i, r in enumerate(amap.ramp_link_id)}
    link_of = {l: i for i, l in enumerate(links)}
    li = np.array([link_of.get(l, -1) for l in amap.nearest_mainline_link_id], np.int64)
    on = (amap.ramp_type == "OR").to_numpy()
    day_of = {str(d): i for i, d in enumerate(dates)}
    flow = np.full((len(dates), T, len(amap)), np.nan, np.float32)
    files = sorted(glob.glob(str(REL / "corridors" / panel / split / "ramp_states" / "*" / "*.parquet")),
                   key=os.path.basename)
    for f in files:
        tb = pq.read_table(f, columns=["date", "timestamp", "ramp_link_id", "flow_vph"]).to_pandas()
        di = day_of.get(str(tb.date.iloc[0]))
        if di is None:
            continue
        ts = tb.timestamp.str
        ti = ts.slice(11, 13).astype(int).to_numpy() * 12 + ts.slice(14, 16).astype(int).to_numpy() // 5
        ri = tb.ramp_link_id.map(rid).to_numpy()
        ok = ~pd.isna(ri)
        flow[di, ti[ok], ri[ok].astype(np.int64)] = tb.flow_vph.to_numpy(np.float32)[ok]
    out = {"flow": flow, "li": li, "on": on}
    np.savez(cp, **out)
    return out


def train_windows(panel: str, d: dict, rng: np.random.Generator, per_day: float = 10 / 31):
    """Simulated Task 2 windows on train, imitating the organizer selector.

    queue(t): some eligible link observed at speed <= 0.6 * free speed.
    onset at T: no queue in T-60..T, queue in T+5..T+30.
    ongoing at T: queue in at least two steps of T-60..T-5, queue in T+5..T+30.
    About per_day windows a day (Poisson), each a random candidate of that day.
    The two D12_I405 corridors have no Task 2 windows and get none.
    """
    if panel.startswith("D12_I405"):
        return []
    vf = network(panel).free_speed_kmh.to_numpy(np.float32)
    v = d["true_speed"].reshape(-1, len(vf))
    el = d["elig"].reshape(-1, len(vf))
    q = (np.nan_to_num(v, nan=1e9) <= 0.6 * vf) & el
    anyq = q.any(1).astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(anyq)])
    n = len(anyq)
    t = np.arange(12, n - 19)
    hist = cs[t] - cs[t - 12]          # T-60..T-5
    hor = cs[t + 7] - cs[t + 1]        # T+5..T+30
    onset = (hist == 0) & (anyq[t] == 0) & (hor > 0)
    ongoing = (hist >= 2) & (hor > 0)
    out = []
    nd = n // T
    for day in range(nd):
        k = rng.poisson(per_day)
        if k == 0:
            continue
        sel = (t // T == day)
        for _ in range(k):
            cond = 1 if rng.random() < 0.5 else 2
            cand = t[sel & (onset if cond == 1 else ongoing)]
            if len(cand) == 0:
                cond = 3 - cond
                cand = t[sel & (onset if cond == 1 else ongoing)]
            if len(cand) == 0:
                continue
            t0 = int(rng.choice(cand))
            out.append((t0 // T, t0 % T, cond))
    return out
