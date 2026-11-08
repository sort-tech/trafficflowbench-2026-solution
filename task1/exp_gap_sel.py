"""Gap cells: x3g05 everywhere, except a selected set where a v5 gap model takes weight w
(x3 the rest). Selected: inside the queue polygon / triangle (g6_in_any == 1) and/or
x3 speed < 85. Same scoring as exp_gap5.py.

    python -I exp_gap_sel.py <tag>[,<tag>...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import common
import metrics
import train_gbm as TG
from pipeline import reconcile

HERE = Path(__file__).resolve().parent
tags = sys.argv[1].split(",")
SELS = ("in or v<85", "in", "v<85")
arms = [("x3g05", None, None, 0.0)] + [(f"{t} | {s} | {w}", t, s, w) for t in tags for s in SELS for w in (0.5, 0.75, 1.0)]
rows = {a[0]: [] for a in arms}
err = {a[0]: [] for a in arms}
for p in common.PANELS:
    ev = pd.read_parquet(HERE / "ds4" / f"eval_{p}.parquet", columns=["day", "t", "li", "win_cond", "regime", "y_v"])
    base = pd.read_parquet(HERE / "preds" / f"gbm8s2_x3_{p}.parquet")
    old = pd.read_parquet(HERE / "preds" / f"gbm9gap_{p}.parquet")
    g = ev.win_cond.to_numpy() > 0
    x3g = {c: base[c].to_numpy().copy() for c in ("pv", "pq", "pk")}
    for c, a in x3g.items():
        a[g] = 0.5 * old[c].to_numpy()[g] + 0.5 * a[g]
    new, sel = {}, {}
    if g.any():
        X = pd.read_parquet(HERE / "ds10" / f"eval_{p}.parquet", columns=["g6_in_any"])
        inside = X.g6_in_any.to_numpy() == 1
        slow = base.pv.to_numpy()[g] < 85
        sel = {"in or v<85": inside | slow, "in": inside, "v<85": slow}
        new = {t: pd.read_parquet(HERE / "preds" / f"{t}gaponly_{p}.parquet") for t in tags}
    pe = TG.PanelEval(p, ev)
    yv = ev.y_v.to_numpy()[g]
    for name, t, s, w in arms:
        arr = {c: a.copy() for c, a in x3g.items()}
        if t is not None and g.any():
            m = sel[s]
            for c, a in arr.items():
                ag = a[g]
                ag[m] = w * new[t][c].to_numpy()[m] + (1 - w) * base[c].to_numpy()[g][m]
                a[g] = ag
        if g.any():
            err[name].append((arr["pv"][g] - yv, yv))
        v, q = reconcile(arr["pv"], arr["pq"], arr["pk"], 1.0, 0.2, 0.8, 80.0)
        rows[name].append(pe(v, q))
    print(p, flush=True)
out = []
for name, *_ in arms:
    e = np.concatenate([x[0] for x in err[name]])
    y = np.concatenate([x[1] for x in err[name]])
    c = y < 60
    r = lambda a: float(np.sqrt(np.mean(a * a)))
    out.append((name, metrics.contribution(metrics.summarize(pd.concat(rows[name]))), r(e), r(e[c]), r(e[~c])))
S = pd.DataFrame(out, columns=["arm", "contrib", "v", "v_cong", "v_free"]).set_index("arm")
S.insert(1, "d_vs_x3g05", S.contrib - S.loc["x3g05", "contrib"])
print(S.round(5).to_string())
S.to_csv(HERE / "logs" / f"exp_gap_sel_{'_'.join(tags)}.csv")
