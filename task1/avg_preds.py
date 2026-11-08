"""Average eval predictions of several tags into a new tag: python avg_preds.py <out_tag> <tag1,tag2,...>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import common
HERE = Path(__file__).resolve().parent
out, tags = sys.argv[1], sys.argv[2].split(",")
for p in common.PANELS:
    fr = [pd.read_parquet(HERE / "preds" / f"{t}_{p}.parquet") for t in tags]
    a = fr[0].copy()
    for c in ("pv", "pq", "pk"):
        a[c] = sum(f[c] for f in fr) / len(fr)
    a.to_parquet(HERE / "preds" / f"{out}_{p}.parquet", index=False)
print("ok")
