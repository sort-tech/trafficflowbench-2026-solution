"""Predict a dataset's eval fold with saved models and score it.

    python -I score_models.py <tag> <ds_dir> [out_tag]
Also prints RMSE split by whether the target sits in a simulated Task 2 gap.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import lightgbm as lgb
import common, metrics
from train_gbm import PanelEval, k_base, RATE
from make_dataset import gap_spans, SEED

HERE = Path(__file__).resolve().parent
tag, dsn = sys.argv[1], sys.argv[2]
out_tag = sys.argv[3] if len(sys.argv) > 3 else f"{tag}@{dsn}"
models = {}
for t in ("y_v", "y_q", "y_k"):
    f = HERE / "models" / f"{tag}_{t}.txt"
    if f.exists():
        models[t] = lgb.Booster(model_file=str(f))
cols = models["y_v"].feature_name()
rows = []
gap_stats = []
for p in common.PANELS:
    ev = pd.read_parquet(HERE / dsn / f"eval_{p}.parquet")
    ev["rate"] = ev.regime.map(RATE).astype(np.float32)
    X = ev[cols]
    pv = ev.base_v.to_numpy() + models["y_v"].predict(X, num_threads=common.THREADS)
    pq = ev.base_q.to_numpy() + models["y_q"].predict(X, num_threads=common.THREADS)
    out = pd.DataFrame({"day": ev.day, "t": ev.t, "li": ev.li, "pv": pv.astype(np.float32), "pq": pq.astype(np.float32)})
    if "y_k" in models:
        out["pk"] = (k_base(ev) + models["y_k"].predict(X, num_threads=common.THREADS)).astype(np.float32)
    out.to_parquet(HERE / "preds" / f"{out_tag}_{p}.parquet", index=False)
    pe = PanelEval(p, ev)
    r = pe(pv, pq)
    rows.append(r)
    nd = len(common.load(p, "train")["dates"])
    g = gap_spans(nd, np.random.default_rng(SEED + 100 + common.PANELS.index(p)))
    ing = g[ev.day.to_numpy(), ev.t.to_numpy()]
    ev_v = (pv - ev.y_v.to_numpy()) ** 2
    ev_q = (pq - ev.y_q.to_numpy()) ** 2
    gap_stats.append((p, ing.mean(), np.sqrt(ev_v[~ing].mean()), np.sqrt(ev_v[ing].mean()) if ing.any() else np.nan,
                      np.sqrt(ev_q[~ing].mean()), np.sqrt(ev_q[ing].mean()) if ing.any() else np.nan))
    print(p, metrics.summarize(r), flush=True)
df = pd.concat(rows)
df.to_csv(HERE / "logs" / f"eval_{out_tag}.csv", index=False)
gs = pd.DataFrame(gap_stats, columns=["panel", "gap_frac", "rmse_v_nogap", "rmse_v_gap", "rmse_q_nogap", "rmse_q_gap"])
print(gs.to_string())
print(gs.mean(numeric_only=True).to_dict())
s = metrics.summarize(df)
print("SUMMARY", out_tag, s, metrics.contribution(s))
print(df.groupby("regime")[["rmse_speed", "rmse_flow_lane", "S_state", "S_LWR"]].mean())
