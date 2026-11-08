"""Compare a candidate with a reference submission: changed cells and windows, T+30 links of changed windows."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from diag_zero import spans
from features_v1 import Static

LOW = ["Q2_D12_I5_N_validation_003", "Q2_D12_I5_N_validation_004", "Q2_D12_I5_S_validation_005", "Q2_D12_I5_S_private_002",
       "Q2_D12_I5_S_private_004", "Q2_D7_I10_W_validation_001", "Q2_D7_I10_W_validation_002", "Q2_D7_I10_W_validation_005",
       "Q2_D7_I10_W_private_003"]
_order: dict = {}


def links(df: pd.DataFrame, wid: str) -> str:
    g = df[df.window_id == wid]
    g = g[g.timestamp == g.timestamp.max()]
    panel = wid[3:].split("_validation")[0].split("_private")[0]
    if panel not in _order:
        _order[panel] = {k: i for i, k in enumerate(Static(panel).link_ids)}
    m = np.zeros(len(_order[panel]), bool)
    m[g[g.queue_pred == 1].link_id.astype(str).map(_order[panel]).to_numpy().astype(int)] = True
    return spans(m)


def main(ref_path: str, new_path: str, show: str = "low") -> None:
    a = pd.read_csv(ref_path); b = pd.read_csv(new_path)
    assert len(b) == 174000 and b.queue_pred.notna().all() and b.queue_pred.isin([0, 1]).all()
    m = a.merge(b, on=["window_id", "timestamp", "link_id"]); d = m.queue_pred_x != m.queue_pred_y
    on = m.window_id.str[-3:].astype(int) <= 5
    print(f"rows {len(b)}, missing {int(b.queue_pred.isna().sum())}; changed cells {int(d.sum())} "
          f"(1->0 {int((d & (m.queue_pred_x == 1)).sum())}, 0->1 {int((d & (m.queue_pred_x == 0)).sum())}), "
          f"windows {m[d].window_id.nunique()} (onset {m[d & on].window_id.nunique()}, ongoing {m[d & ~on].window_id.nunique()}), "
          f"positives {int(a.queue_pred.sum())} -> {int(b.queue_pred.sum())}; "
          f"validation windows changed {m[d & m.window_id.str.contains('validation')].window_id.nunique()}")
    ws = LOW if show == "low" else sorted(m[d].window_id.unique())
    for w in ws:
        print(f"  {w[3:]:28s} {links(a, w):34s} -> {links(b, w)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
