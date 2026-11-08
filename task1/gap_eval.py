"""RMSE of saved eval predictions inside / outside simulated Task 2 window gaps (ds4-style).

    python -I gap_eval.py <pred_tag> <ds_dir>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import common
from make_dataset import SEED

HERE = Path(__file__).resolve().parent
tag, dsn = sys.argv[1], sys.argv[2]
rows = []
for p in common.PANELS:
    ev = pd.read_parquet(HERE / dsn / f"eval_{p}.parquet", columns=["day", "t", "li", "y_v", "y_q", "win_cond"])
    pr = pd.read_parquet(HERE / "preds" / f"{tag}_{p}.parquet")
    ing = ev.win_cond.to_numpy() > 0
    xv = pr.pv.to_numpy() - ev.y_v.to_numpy()
    xq = pr.pq.to_numpy() - ev.y_q.to_numpy()
    rows.append((p, ing.mean(), np.sqrt((xv[~ing] ** 2).mean()), np.sqrt((xv[ing] ** 2).mean()) if ing.any() else np.nan,
                 np.sqrt((xq[~ing] ** 2).mean()), np.sqrt((xq[ing] ** 2).mean()) if ing.any() else np.nan,
                 np.sqrt((xv ** 2).mean())))
df = pd.DataFrame(rows, columns=["panel", "gap_frac", "rmse_v_out", "rmse_v_gap", "rmse_q_out", "rmse_q_gap", "rmse_v_all"])
print(df.round(3).to_string())
print(df.mean(numeric_only=True).round(4).to_dict())
