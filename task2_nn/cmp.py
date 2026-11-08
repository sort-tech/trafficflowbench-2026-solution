"""Cell differences of candidates vs a reference file. Usage: cmp.py <reference csv> <candidate csv> [...]"""
import sys
import pandas as pd
ref = pd.read_csv(sys.argv[1])
for c in sys.argv[2:]:
    m = ref.merge(pd.read_csv(c), on=["window_id", "timestamp", "link_id"])
    d = m[m.queue_pred_x != m.queue_pred_y]
    val = d[d.window_id.str.contains("validation")].window_id.nunique()
    print(f"{c}: rows {len(m):,}, different {len(d)} (1->0 {int((d.queue_pred_x == 1).sum())}, 0->1 {int((d.queue_pred_y == 1).sum())}), "
          f"windows {d.window_id.nunique()} (validation {val}), positives {int(m.queue_pred_x.sum()):,} -> {int(m.queue_pred_y.sum()):,}")
