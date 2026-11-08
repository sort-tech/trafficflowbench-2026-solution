"""Reproduce the official historical-mean baseline on train and score it."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import common, metrics


def profile_baseline(d):
    """weekday x tod mean per link from the unmasked train layer, pct>=75 (official)."""
    wd = common.slot_weekday(d["dates"])
    v, q = d["true_speed"], d["true_flow"]
    ok = (d["pct"] >= 75) & np.isfinite(v) & np.isfinite(q)
    nl = v.shape[2]
    ps = np.zeros((7, common.T, nl)); pqm = np.zeros((7, common.T, nl))
    for w in range(7):
        m = ok[wd == w]
        c = m.sum(0)
        ps[w] = np.where(m, v[wd == w], 0).sum(0) / np.maximum(c, 1)
        pqm[w] = np.where(m, q[wd == w], 0).sum(0) / np.maximum(c, 1)
        fb_s = np.where(ok, v, 0).sum((0, 1)) / np.maximum(ok.sum((0, 1)), 1)
        fb_q = np.where(ok, q, 0).sum((0, 1)) / np.maximum(ok.sum((0, 1)), 1)
        ps[w] = np.where(c > 0, ps[w], fb_s); pqm[w] = np.where(c > 0, pqm[w], fb_q)
    return ps[wd], pqm[wd]


if __name__ == "__main__":
    panels = sys.argv[1].split(",") if len(sys.argv) > 1 else common.PANELS
    rows = []
    for p in panels:
        t = time.time()
        d = common.load(p, "train")
        ps, pq = profile_baseline(d)
        sc = metrics.Scorer(p, d)
        r = sc.score(ps, pq)
        print(r.to_string(), f"{time.time()-t:.0f}s", flush=True)
        rows.append(r)
    df = pd.concat(rows)
    df.to_csv(Path(__file__).resolve().parent / "logs" / "baseline_train_full.csv", index=False)
    print(metrics.summarize(df))
