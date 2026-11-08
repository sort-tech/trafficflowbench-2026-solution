"""g6 vs v7 state candidates: mean / max |diff| of speed and flow, all cells and Task 2 gap
cells (T+5..T+90 after each window origin, every link of the panel) separately.

    python -I diff_g6.py [candidate dir, default gbm_v8s2rx3g_ss3f_ramp_g6]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import common

HERE = Path(__file__).resolve().parent
A = HERE / "candidates" / "gbm_v8s2rx3g_ss3f_ramp" / "state_submission.csv"
B = HERE / "candidates" / (sys.argv[1] if len(sys.argv) > 1 else "gbm_v8s2rx3g_ss3f_ramp_g6") / "state_submission.csv"


def main():
    use = ["panel", "timestamp", "link_id", "mask_regime", "speed_kmh", "flow_vph"]
    a = pd.read_csv(A, usecols=use)
    b = pd.read_csv(B, usecols=use)
    assert len(a) == len(b) and all((a[c].values == b[c].values).all() for c in ("panel", "timestamp", "link_id"))
    gap = np.zeros(len(a), bool)
    day = a.timestamp.str[:10].to_numpy()
    slot = (a.timestamp.str[11:13].astype(int) * 12 + a.timestamp.str[14:16].astype(int) // 5).to_numpy()
    for p in common.PANELS:
        for split in ("validation", "private"):
            dates = np.array(sorted(map(str, common.load(p, split)["dates"])))
            wins = common.split_windows(p, split, dates)
            if not wins:
                continue
            flat = np.zeros(len(dates) * common.T, bool)
            for di, s0, _ in wins:
                t0 = di * common.T + s0
                flat[max(0, t0 + 1):min(len(flat), t0 + 19)] = True
            sel = (a.panel.to_numpy() == p) & np.isin(day, dates)
            di = np.searchsorted(dates, day[sel])
            gap[sel] = flat[di * common.T + slot[sel]]
    print(f"cells {len(a):,}  gap cells {int(gap.sum()):,}  NaN {int(b[['speed_kmh', 'flow_vph']].isna().sum().sum())}")
    rows = []
    for name, m in (("all", np.ones(len(a), bool)), ("gap", gap), ("non-gap", ~gap)):
        for c in ("speed_kmh", "flow_vph"):
            d = (b[c] - a[c]).abs().to_numpy()[m]
            rows.append({"cells": name, "col": c, "n": int(m.sum()), "mean": d.mean(), "median": np.median(d),
                         "p99": np.quantile(d, 0.99), "max": d.max(), "changed": float((d > 1e-9).mean())})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    dq = (b.flow_vph - a.flow_vph).abs().to_numpy()
    for t in (500, 1000, 2000):
        print(f"|dq|>{t}: {int((dq > t).sum())} (gap {int(((dq > t) & gap).sum())})")


if __name__ == "__main__":
    main()
