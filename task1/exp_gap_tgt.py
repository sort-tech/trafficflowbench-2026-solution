"""Gap cells: per-target and per-state mixes of x3, gbm9gap and a new gap model.

Each arm gives, for speed / flow / density, the (gbm9gap, new) weights in predicted
free flow and in congestion (x3 speed >= / < 80); x3 takes the rest. Same scoring as
exp_gap5.py.

    python -I exp_gap_tgt.py <tag>
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
tag = sys.argv[1]
O, N, ON = (0.5, 0), (0, 0.5), (0.25, 0.25)
# arm: {target: (free (old, new), congested (old, new))}
ARMS = {
    "x3g05": {c: (O, O) for c in ("pv", "pq", "pk")},
    "new 0.5": {c: (N, N) for c in ("pv", "pq", "pk")},
    "new q only": {"pv": (O, O), "pq": (N, N), "pk": (O, O)},
    "new q,k": {"pv": (O, O), "pq": (N, N), "pk": (N, N)},
    "new q 1.0": {"pv": (O, O), "pq": ((0, 1), (0, 1)), "pk": (O, O)},
    "new v only": {"pv": (N, N), "pq": (O, O), "pk": (O, O)},
    "cong new / free old": {c: (O, N) for c in ("pv", "pq", "pk")},
    "cong new 0.75 / free old": {c: (O, (0, 0.75)) for c in ("pv", "pq", "pk")},
    "thirds": {c: ((1 / 3, 1 / 3), (1 / 3, 1 / 3)) for c in ("pv", "pq", "pk")},
    "free old / cong new, q new": {"pv": (O, N), "pq": (N, N), "pk": (O, N)},
    "0.25/0.25 all": {c: (ON, ON) for c in ("pv", "pq", "pk")},
    "new 0.75": {c: ((0, 0.75), (0, 0.75)) for c in ("pv", "pq", "pk")},
    "new 1.0": {c: ((0, 1), (0, 1)) for c in ("pv", "pq", "pk")},
    "v new 1.0, q,k new 0.5": {"pv": ((0, 1), (0, 1)), "pq": (N, N), "pk": (N, N)},
    "v new 1.0, q,k x3g05": {"pv": ((0, 1), (0, 1)), "pq": (O, O), "pk": (O, O)},
    "v new 0.75, q,k new 0.5": {"pv": ((0, 0.75), (0, 0.75)), "pq": (N, N), "pk": (N, N)},
}
rows = {k: [] for k in ARMS}
err = {k: [] for k in ARMS}
for p in common.PANELS:
    ev = pd.read_parquet(HERE / "ds4" / f"eval_{p}.parquet", columns=["day", "t", "li", "win_cond", "regime", "y_v", "y_q"])
    base = pd.read_parquet(HERE / "preds" / f"gbm8s2_x3_{p}.parquet")
    old = pd.read_parquet(HERE / "preds" / f"gbm9gap_{p}.parquet")
    g = ev.win_cond.to_numpy() > 0
    new = pd.read_parquet(HERE / "preds" / f"{tag}gaponly_{p}.parquet") if g.any() else None
    cong = base.pv.to_numpy()[g] < 80
    pe = TG.PanelEval(p, ev)
    for name, spec in ARMS.items():
        arr = {c: base[c].to_numpy().copy() for c in ("pv", "pq", "pk")}
        if g.any():
            for c, ((fo, fn), (co, cn)) in spec.items():
                wo = np.where(cong, co, fo)
                wn = np.where(cong, cn, fn)
                arr[c][g] = wo * old[c].to_numpy()[g] + wn * new[c].to_numpy() + (1 - wo - wn) * arr[c][g]
            err[name].append((arr["pv"][g] - ev.y_v.to_numpy()[g], arr["pq"][g] - ev.y_q.to_numpy()[g],
                              ev.y_v.to_numpy()[g]))
        v, q = reconcile(arr["pv"], arr["pq"], arr["pk"], 1.0, 0.2, 0.8, 80.0)
        rows[name].append(pe(v, q))
    print(p, flush=True)
r = lambda a: float(np.sqrt(np.mean(a * a)))
out = []
for name in ARMS:
    ev_ = np.concatenate([x[0] for x in err[name]])
    eq_ = np.concatenate([x[1] for x in err[name]])
    y = np.concatenate([x[2] for x in err[name]])
    c = y < 60
    out.append((name, metrics.contribution(metrics.summarize(pd.concat(rows[name]))), r(ev_), r(ev_[c]), r(ev_[~c]),
                r(eq_), r(eq_[c]), r(eq_[~c])))
S = pd.DataFrame(out, columns=["arm", "contrib", "v", "v_cong", "v_free", "q", "q_cong", "q_free"]).set_index("arm")
S.insert(1, "d_vs_x3g05", S.contrib - S.loc["x3g05", "contrib"])
print(S.round(5).to_string())
S.to_csv(HERE / "logs" / f"exp_gap_tgt_{tag}.csv")
