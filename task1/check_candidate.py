"""Check a candidate state_submission.csv against the validation and private templates.

Same key set as the templates (panel, timestamp, station_id, link_id, mask_regime),
no duplicates, finite non-negative values, and the S_FD empty-road floor.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import pyarrow.csv as pacsv
import common

KEYS = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
path = Path(sys.argv[1])
tb = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(
    column_types={**{k: "string" for k in KEYS}, "speed_kmh": "float64", "flow_vph": "float64"}))
sub = tb.to_pandas()
print("rows", len(sub), "columns", list(sub.columns))
ok = True
assert list(sub.columns) == KEYS + ["speed_kmh", "flow_vph"], sub.columns
for c in ("speed_kmh", "flow_vph"):
    bad = ~np.isfinite(sub[c]) | (sub[c] < 0)
    if bad.any():
        ok = False
        print("bad values in", c, int(bad.sum()))
dup = sub.duplicated(KEYS).sum()
if dup:
    ok = False
    print("duplicate keys", dup)
tot = 0
for split in ("validation", "private"):
    for p in common.PANELS:
        tpl = common.read_template(p, split)
        tot += len(tpl)
        m = tpl.merge(sub[KEYS], on=KEYS, how="left", indicator=True)
        miss = int((m._merge != "both").sum())
        if miss:
            ok = False
            print(split, p, "missing", miss)
print("template rows", tot, "submission rows", len(sub))
if tot != len(sub):
    ok = False
low = sub.groupby(["panel", "mask_regime"]).flow_vph.apply(lambda s: (s < 50).mean())
print("max share of flow < 50 vph per panel/regime", float(low.max()))
print(sub.groupby("panel")[["speed_kmh", "flow_vph"]].describe().round(1).iloc[:, [1, 3, 5, 7, 9, 11, 13, 15]].to_string())
print("OK" if ok else "FAILED")
