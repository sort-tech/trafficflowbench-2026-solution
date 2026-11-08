"""Ongoing: speed-ratio regression (multi-horizon) vs queue classifier, month-fold CV on greedy windows."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import CACHE, PANELS, Panel
from features import FOLDS
from model import PARAMS, VARIANTS, feature_cols, iou

VAR = sys.argv[1] if len(sys.argv) > 1 else "aug"
parts = []
for p in PANELS:
    d = pd.read_parquet(CACHE / f"ds_{p}.parquet", filters=[("cond", "=", 0)])
    P = Panel(p)
    d["ratio"] = np.clip(P.speed[d.t.to_numpy() + d.h.to_numpy(), d.link.to_numpy()] / P.vcut[d.link.to_numpy()], 0, 2)
    d["panel"] = p
    parts.append(d)
df = pd.concat(parts, ignore_index=True)
feats = [f for f in feature_cols(df) if f not in VARIANTS[VAR] and f != "ratio"]
res = []
for k in FOLDS:
    tr = df[(df.fold != k) & (df.vis == 1) & (df.keep == 1)]
    te = df[(df.fold == k) & (df.greedy == 1)].copy()
    clf = lgb.train(PARAMS, lgb.Dataset(tr[feats], label=tr.y.to_numpy(), categorical_feature=["pid"]), 300)
    reg = lgb.train(dict(PARAMS, objective="huber", alpha=0.3), lgb.Dataset(tr[feats], label=tr.ratio.to_numpy(), categorical_feature=["pid"]), 300)
    te["p"] = clf.predict(te[feats], num_threads=3)
    te["r"] = reg.predict(te[feats], num_threads=3)
    te.loc[te.keep == 0, "p"] = 0; te.loc[te.keep == 0, "r"] = 2
    for (panel, t), g in te.groupby(["panel", "t"], sort=False):
        y, e = g.y.to_numpy(), g.elig.to_numpy()
        pr, rr = g.p.to_numpy(), g.r.to_numpy()
        sig = 1 / (1 + np.exp(-(1 - rr) / 0.08))
        row = {"panel": panel, "t": t, "clf35": iou(pr >= 0.35, y, e), "clf42": iou(pr >= 0.42, y, e)}
        for tau in (0.95, 1.0, 1.05, 1.1):
            row[f"reg{tau}"] = iou(rr <= tau, y, e)
        for thr in (0.35, 0.4, 0.45):
            row[f"mix{thr}"] = iou((pr + sig) / 2 >= thr, y, e)
        res.append(row)
    print("fold", k, flush=True)
r = pd.DataFrame(res)
print(r.drop(columns="t").groupby("panel").mean().mean().sort_values(ascending=False).round(4).to_string())
