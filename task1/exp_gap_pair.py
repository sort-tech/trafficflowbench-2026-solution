"""Gap cells: x3 mixed with two new gap models a and b, separate weights in predicted
free flow and congestion (x3 speed >= / < 80). Same scoring as exp_gap5.py.

    python -I exp_gap_pair.py <tag_a[,seed variants]> <tag_b[,...]>   (variants are averaged)
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
ta, tb = sys.argv[1:3]
# arm: (free (w_a, w_b), congested (w_a, w_b)); gbm9gap only in the x3g05 reference
ARMS = {"x3g05": None, "a 0.75": ((0.75, 0), (0.75, 0)), "b 0.75": ((0, 0.75), (0, 0.75)),
        "avg 0.75": ((0.375, 0.375), (0.375, 0.375)), "avg 1.0": ((0.5, 0.5), (0.5, 0.5)),
        "free a / cong b, 0.75": ((0.75, 0), (0, 0.75)), "free a / cong b, 1.0": ((1, 0), (0, 1)),
        "free a 0.75 / cong avg 1.0": ((0.75, 0), (0.5, 0.5)), "free a 0.75 / cong avg 0.75": ((0.75, 0), (0.375, 0.375))}


def avg(tags, p):
    fr = [pd.read_parquet(HERE / "preds" / f"{t}gaponly_{p}.parquet") for t in tags.split(",")]
    return pd.DataFrame({c: sum(f[c].to_numpy() for f in fr) / len(fr) for c in ("pv", "pq", "pk")})


rows = {k: [] for k in ARMS}
err = {k: [] for k in ARMS}
for p in common.PANELS:
    ev = pd.read_parquet(HERE / "ds4" / f"eval_{p}.parquet", columns=["day", "t", "li", "win_cond", "regime", "y_v", "y_q"])
    base = pd.read_parquet(HERE / "preds" / f"gbm8s2_x3_{p}.parquet")
    old = pd.read_parquet(HERE / "preds" / f"gbm9gap_{p}.parquet")
    g = ev.win_cond.to_numpy() > 0
    A = avg(ta, p) if g.any() else None
    B = avg(tb, p) if g.any() else None
    cong = base.pv.to_numpy()[g] < 80
    pe = TG.PanelEval(p, ev)
    for name, spec in ARMS.items():
        arr = {c: base[c].to_numpy().copy() for c in ("pv", "pq", "pk")}
        if g.any():
            for c, a in arr.items():
                if spec is None:
                    a[g] = 0.5 * old[c].to_numpy()[g] + 0.5 * a[g]
                    continue
                (fa, fb), (ca, cb) = spec
                wa, wb = np.where(cong, ca, fa), np.where(cong, cb, fb)
                a[g] = wa * A[c].to_numpy() + wb * B[c].to_numpy() + (1 - wa - wb) * a[g]
            err[name].append((arr["pv"][g] - ev.y_v.to_numpy()[g], arr["pq"][g] - ev.y_q.to_numpy()[g], ev.y_v.to_numpy()[g]))
        v, q = reconcile(arr["pv"], arr["pq"], arr["pk"], 1.0, 0.2, 0.8, 80.0)
        rows[name].append(pe(v, q))
    print(p, flush=True)
r = lambda a: float(np.sqrt(np.mean(a * a)))
out = []
for name in ARMS:
    ev_ = np.concatenate([x[0] for x in err[name]]); eq_ = np.concatenate([x[1] for x in err[name]])
    y = np.concatenate([x[2] for x in err[name]]); c = y < 60
    out.append((name, metrics.contribution(metrics.summarize(pd.concat(rows[name]))), r(ev_), r(ev_[c]), r(ev_[~c]),
                r(eq_), r(eq_[c]), r(eq_[~c])))
S = pd.DataFrame(out, columns=["arm", "contrib", "v", "v_cong", "v_free", "q", "q_cong", "q_free"]).set_index("arm")
S.insert(1, "d_vs_x3g05", S.contrib - S.loc["x3g05", "contrib"])
print(S.round(5).to_string())
S.to_csv(HERE / "logs" / f"exp_gap_pair_{ta.replace(',', '+')}_{tb.replace(',', '+')}.csv")
