"""Low-confidence release onset windows (task2 safety check): predicted T+30 link positions must keep
N_val_003 37,39,104-105 / N_val_004 59,61-63,238 / S_val_005 39,41 (positions in upstream -> downstream order).

Usage: safety.py <candidate csv> [...]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T2 = Path(os.environ.get("T2_DIR", HERE.parent / "task2"))   # task2 code + cache (run_all: the fresh task2 build)
sys.path.insert(0, str(T2))
import pandas as pd

import features_v1 as F1

KEEP = {"Q2_D12_I5_N_validation_003": {37, 39, 104, 105}, "Q2_D12_I5_N_validation_004": {59, 61, 62, 63, 238},
        "Q2_D12_I5_S_validation_005": {39, 41}}


def positions(csv: str) -> dict:
    df = pd.read_csv(csv)
    out = {}
    for wid in KEEP:
        panel = wid[3:].rsplit("_validation", 1)[0]
        pos = {l: i for i, l in enumerate(F1.Static(panel).link_ids)}
        d = df[(df.window_id == wid) & (df.queue_pred == 1)]
        last = d.timestamp.max()
        out[wid] = sorted(pos[l] for l in d[d.timestamp == last].link_id)
    return out


def main() -> None:
    for csv in sys.argv[1:]:
        ps = positions(csv)
        ok = all(KEEP[w] <= set(p) for w, p in ps.items())
        print(("PASS" if ok else "FAIL"), csv)
        for w, p in ps.items():
            print("  ", w, p, "missing", sorted(KEEP[w] - set(p)))


if __name__ == "__main__":
    main()
