"""Shared loaders, pseudo-window selector and local scorer for Task 2."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KAGGLE = HERE.parent
REL = Path(os.environ.get("TFB_REL", KAGGLE / "data" / "kaggle_public"))
CACHE = HERE / "cache"
PANELS = ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]
FAMILY = {p: p.rsplit("_", 1)[0] for p in PANELS}
T0 = pd.Timestamp("2030-06-01T00:00:00Z")
STEP = pd.Timedelta("5min")
H = 6          # horizon steps T+5..T+30
NHIST = 12     # history steps T-60..T-5
SPACING = 72   # 360 minutes in steps


class Panel:
    """Dense train arrays for one panel, links ordered upstream -> downstream."""

    def __init__(self, panel: str):
        z = np.load(CACHE / f"{panel}_train.npz", allow_pickle=False)
        self.panel = panel
        self.speed = z["speed"]
        self.flow = z["flow"]
        self.occ = z["occ"]
        self.pct = z["pct"]
        net = REL / "corridors" / panel / "network"
        topo = pd.read_csv(net / "lwr_mainline_topology.csv")
        links = pd.read_csv(net / "links.csv").merge(topo[["link_id", "order_index"]], on="link_id", how="left")
        self.link_ids = links.sort_values("order_index").link_id.astype(str).to_numpy()
        self.vcut = z["vcut"]
        self.lanes = z["lanes"]
        self.length = z["length"]
        self.capacity = z["capacity"]
        self.free = z["free"]
        self.n_t, self.n_l = self.speed.shape
        self.elig = self.pct >= 75
        self.visible = ~np.isnan(self.speed)
        self.q = np.nan_to_num(self.speed, nan=1e9) <= self.vcut   # observed queue, any non-null cell

    def t_index(self, ts) -> int:
        return int((pd.Timestamp(ts) - T0) / STEP)


def clean_queue(P: Panel) -> np.ndarray:
    """Proxy for the noise-free queue state: observed speed, gaps forward-filled in time."""
    s = pd.DataFrame(P.speed).ffill().bfill().to_numpy()
    return s <= P.vcut


def select_windows(P: Panel, qc: np.ndarray, max_per_condition: int | None = None, start: int = NHIST,
                   persistence_cap: float = 0.9, min_steps: int = 2, end: int | None = None) -> pd.DataFrame:
    """Greedy chronological selector imitating the organizer rules on observed train data.

    Reproduces 72 of the 80 official train windows with:
    onset   : no eligible queue in T-60..T-5 nor at T, clean queue only at T+30 (earliest origin).
    ongoing : eligible queue visible in >= 2 history steps, clean queue in horizon,
              clean persistence IoU of the state at T <= 0.9 (all cells).
    Consecutive origins (any condition) are >= 360 minutes apart.
    """
    qv = P.q & P.elig
    anyq = qv.any(1)
    cs = np.concatenate([[0], np.cumsum(anyq)])
    end = P.n_t - H - 1 if end is None else min(end, P.n_t - H - 1)
    rows = []
    t = start
    last = -10**9
    n_on = n_go = 0
    while t < end:
        if t - last < SPACING:
            t = last + SPACING
            continue
        nsteps = cs[t] - cs[t - NHIST]
        hor = qc[t + 1:t + 1 + H]
        cond = None
        if hor.any():
            if nsteps == 0 and not anyq[t]:
                if not qc[t + 1:t + H].any() and (max_per_condition is None or n_on < max_per_condition):
                    cond = "queue_onset"
            elif nsteps >= min_steps and (max_per_condition is None or n_go < max_per_condition):
                pers = qc[t]
                union = (hor | pers).sum()
                iou = (hor & pers).sum() / union if union else 1.0
                if iou <= persistence_cap:
                    cond = "queue_ongoing"
        if cond is not None:
            rows.append({"panel": P.panel, "t": t, "condition": cond,
                         "forecast_origin": (T0 + t * STEP).strftime("%Y-%m-%dT%H:%M:%SZ")})
            last = t
            if cond == "queue_onset":
                n_on += 1
            else:
                n_go += 1
            if max_per_condition is not None and n_on >= max_per_condition and n_go >= max_per_condition:
                break
        t += 1
    return pd.DataFrame(rows)


def window_iou(pred: np.ndarray, truth: np.ndarray, elig: np.ndarray) -> float:
    """pred/truth/elig: bool arrays [H, L]. Official convention: union over eligible cells."""
    if not elig.any():
        return 0.0
    p = pred & elig
    t = truth & elig
    union = (p | t).sum()
    return 1.0 if union == 0 else float((p & t).sum() / union)


def aggregate(df: pd.DataFrame, col: str = "iou") -> dict:
    """Official aggregation: windows -> (panel, condition) -> panel -> family -> overall."""
    pc = df.groupby(["panel", "condition"])[col].mean().reset_index()
    pm = pc.groupby("panel")[col].mean()
    fam = pm.groupby(pm.index.map(FAMILY)).mean()
    by_cond = pc.groupby("condition")[col].mean()
    return {"S_queue": float(fam.mean()), "onset": float(by_cond.get("queue_onset", np.nan)),
            "ongoing": float(by_cond.get("queue_ongoing", np.nan)), "per_panel": pm.round(4).to_dict()}
