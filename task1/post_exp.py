"""Density reconciliation grid on eval-fold predictions.

Given model speed pv, per-lane flow pq and density pk, move the implied density
q/v toward pk by a fraction beta, putting a share alpha of the log change into
flow and the rest into speed:
    d = log(pk) - log(pq/pv)
    q* = pq * exp(alpha*beta*d),  v* = pv * exp(-(1-alpha)*beta*d)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import common, metrics
from train_gbm import PanelEval

HERE = Path(__file__).resolve().parent
tag = sys.argv[1]
ds = HERE / (sys.argv[2] if len(sys.argv) > 2 else "ds2")
grid = [(a, b) for b in (0.0, 0.5, 0.75, 1.0) for a in (0.0, 0.5, 1.0)] if len(sys.argv) <= 3 else \
    [tuple(map(float, x.split(":"))) for x in sys.argv[3].split(",")]


def reconcile(pv, pq, pk, alpha, beta, beta_c=None, vthr=80.0):
    """beta in free flow, beta_c where the predicted speed is below vthr."""
    pv = np.maximum(pv, 1.0); pq = np.maximum(pq, 1.0); pk = np.maximum(pk, 1e-3)
    if beta_c is not None:
        beta = np.where(pv < vthr, beta_c, beta)
    d = np.log(pk) - np.log(pq / pv)
    return pv * np.exp(-(1 - alpha) * beta * d), pq * np.exp(alpha * beta * d)


rows = {g: [] for g in grid}
for p in common.PANELS:
    ev = pd.read_parquet(ds / f"eval_{p}.parquet", columns=["day", "t", "li"])
    pr = pd.read_parquet(HERE / "preds" / f"{tag}_{p}.parquet")
    assert (pr.day.to_numpy() == ev.day.to_numpy()).all()
    pe = PanelEval(p, ev)
    for g in grid:
        v, q = reconcile(pr.pv.to_numpy(), pr.pq.to_numpy(), pr.pk.to_numpy(), *g)
        rows[g].append(pe(v, q))
    print(p, flush=True)
for g in grid:
    df = pd.concat(rows[g])
    s = metrics.summarize(df)
    reg = df.groupby("regime")[["S_state", "S_LWR"]].mean()
    print("cfg=" + ":".join(str(x) for x in g), " ".join(f"{k}={v:.4f}" for k, v in s.items()),
          f"contrib={metrics.contribution(s):.5f}", "| R3 LWR", round(reg.loc["R3", "S_LWR"], 4), flush=True)
