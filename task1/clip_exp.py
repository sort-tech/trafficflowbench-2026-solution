"""Effect of clipping predictions to the generator's speed cap and per-link flow range."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import common, metrics
from train_gbm import PanelEval
from pipeline import reconcile

tag, dsn = sys.argv[1], sys.argv[2]
HERE = Path(__file__).resolve().parent
res = {k: [] for k in ("rc", "rc+clip", "clip_then_rc")}
for p in common.PANELS:
    ev = pd.read_parquet(HERE / dsn / f"eval_{p}.parquet", columns=["day", "t", "li", "lanes"])
    pr = pd.read_parquet(HERE / "preds" / f"{tag}_{p}.parquet")
    d = common.load(p, "train")
    lanes = common.network(p).lanes.to_numpy(np.float32)
    qmax = np.nanmax(d["true_flow"], axis=(0, 1)) / lanes
    vmin = np.nanmin(d["true_speed"], axis=(0, 1))
    pe = PanelEval(p, ev)
    v, q = reconcile(pr.pv.to_numpy(), pr.pq.to_numpy(), pr.pk.to_numpy(), 1.0, 0.3, 0.8, 80.0)
    res["rc"].append(pe(v, q))
    li = ev.li.to_numpy()
    v2 = np.clip(v, vmin[li], 120.75)
    q2 = np.clip(q, 0, 1.1 * qmax[li])
    res["rc+clip"].append(pe(v2, q2))
    vc = np.clip(pr.pv.to_numpy(), vmin[li], 120.75)
    v3, q3 = reconcile(vc, pr.pq.to_numpy(), pr.pk.to_numpy(), 1.0, 0.3, 0.8, 80.0)
    q3 = np.clip(q3, 0, 1.1 * qmax[li])
    res["clip_then_rc"].append(pe(v3, q3))
    print(p, (v > 120.75).mean(), (q > 1.1 * qmax[li]).mean(), flush=True)
for k, rows in res.items():
    s = metrics.summarize(pd.concat(rows))
    print(k, {a: round(b, 5) for a, b in s.items()}, round(metrics.contribution(s), 5))
