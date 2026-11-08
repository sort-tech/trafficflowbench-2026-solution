"""Density model variants on ds4: objective / weighting. Saves preds/<tag>_<panel> with
pv, pq copied from a base tag and pk from the new density model.

    python -I kmodel_exp.py <tag> <base_tag> <objective> <weighting: lenlanes|none> [ds]
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb
import common
import train_gbm as TG

HERE = Path(__file__).resolve().parent
tag, base_tag, obj, wmode = sys.argv[1:5]
DS = HERE / (sys.argv[5] if len(sys.argv) > 5 else "ds4")
TG.DS = DS
fit = TG.load_fit(common.PANELS)
cols = TG.feat_cols(fit)
y = TG.k_true(fit) - TG.k_base(fit)
w = (fit.lanes * fit.len).to_numpy(np.float32) if wmode == "lenlanes" else None
rng = np.random.default_rng(TG.SEED)
va = rng.random(len(fit)) < 0.05
prm = TG.params(obj)
prm["learning_rate"] = 0.1
if obj == "quantile":
    prm["alpha"] = 0.5
if obj == "huber":
    prm["alpha"] = 0.5  # residual scale of per-lane density is ~0.5 veh/km
dtr = lgb.Dataset(fit.loc[~va, cols], y[~va], weight=None if w is None else w[~va])
dva = lgb.Dataset(fit.loc[va, cols], y[va], weight=None if w is None else w[va], reference=dtr)
t0 = time.time()
m = lgb.train(prm, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
print("best", m.best_iteration, f"{time.time()-t0:.0f}s", flush=True)
del fit, dtr, dva
for p in common.PANELS:
    ev = pd.read_parquet(DS / f"eval_{p}.parquet")
    ev["rate"] = ev.regime.map(TG.RATE).astype(np.float32)
    pr = pd.read_parquet(HERE / "preds" / f"{base_tag}_{p}.parquet")
    pr["pk"] = (TG.k_base(ev) + m.predict(ev[cols], num_iteration=m.best_iteration)).astype(np.float32)
    pr.to_parquet(HERE / "preds" / f"{tag}_{p}.parquet", index=False)
print("done")
