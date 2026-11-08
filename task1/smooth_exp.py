"""Temporal smoothing of predicted density inside the target field (flow carries the change).

N_s(t) = (1-2a) N(t) + a (N(t-1) + N(t+1)), neighbours taken from the filled field
(observed value where observed, prediction where a target). Applied after reconcile.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import common, metrics
from train_gbm import PanelEval
from pipeline import reconcile

tag, dsn = sys.argv[1], sys.argv[2]
alphas = [float(x) for x in sys.argv[3].split(",")]
HERE = Path(__file__).resolve().parent
res = {a: [] for a in alphas}
for p in common.PANELS:
    ev = pd.read_parquet(HERE / dsn / f"eval_{p}.parquet", columns=["day", "t", "li"])
    pr = pd.read_parquet(HERE / "preds" / f"{tag}_{p}.parquet")
    pe = PanelEval(p, ev)
    d = common.load(p, "train")
    lanes = pe.lanes
    v, q = reconcile(np.clip(pr.pv.to_numpy(), 1, 120.75), pr.pq.to_numpy(), pr.pk.to_numpy(), 1.0, 0.3, 0.8, 80.0)
    di, ti, li = pe.idx
    nd, nl = len(pe.days), len(lanes)
    # filled per-lane density field: observed where available (masked layer), prediction at targets
    ov = d["obs_speed_kmh"][pe.days].astype(float); oq = d["obs_flow_vph"][pe.days] / lanes
    K = oq / np.clip(ov, 1, None)
    K[di, ti, li] = q / v
    tg = np.zeros((nd, common.T, nl), bool); tg[di, ti, li] = True
    Kf = K.reshape(nd * common.T, nl)
    tgf = tg.reshape(nd * common.T, nl)
    prev = np.vstack([np.full((1, nl), np.nan), Kf[:-1]])
    nxt = np.vstack([Kf[1:], np.full((1, nl), np.nan)])
    flat = (di * common.T + ti, li)
    for a in alphas:
        kp, kn, k0 = prev[flat], nxt[flat], Kf[flat]
        ks = np.where(np.isfinite(kp) & np.isfinite(kn), (1 - 2 * a) * k0 + a * (kp + kn), k0)
        res[a].append(pe(v, ks * v))
    print(p, flush=True)
for a, rows in res.items():
    s = metrics.summarize(pd.concat(rows))
    print(f"a={a}", {k: round(x, 5) for k, x in s.items()}, round(metrics.contribution(s), 5), flush=True)
