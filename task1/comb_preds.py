"""Replace eval predictions inside simulated Task 2 window gaps with a specialist's:
python comb_preds.py <out_tag> <base_tag> <gap_tag> <ds>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import common
HERE = Path(__file__).resolve().parent
out, base, spec, dsn = sys.argv[1:5]
w = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0  # specialist weight inside the gaps
for p in common.PANELS:
    ev = pd.read_parquet(HERE / dsn / f"eval_{p}.parquet", columns=["day", "t", "li", "win_cond"])
    a = pd.read_parquet(HERE / "preds" / f"{base}_{p}.parquet")
    b = pd.read_parquet(HERE / "preds" / f"{spec}_{p}.parquet")
    assert (a.day.to_numpy() == ev.day.to_numpy()).all() and (b.t.to_numpy() == ev.t.to_numpy()).all()
    g = ev.win_cond.to_numpy() > 0
    for c in ("pv", "pq", "pk"):
        a[c] = np.where(g, w * b[c] + (1 - w) * a[c], a[c])
    a.to_parquet(HERE / "preds" / f"{out}_{p}.parquet", index=False)
    print(p, int(g.sum()))
