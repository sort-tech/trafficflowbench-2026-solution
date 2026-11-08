"""Step 1: do on-ramp flow surges before T say where an onset queue appears at T+30?

Train only. On-ramp flows (ramp_states, rows with timestamp <= T only) are mapped to their nearest mainline link
(ramp_attachment_map.csv). For each onset window of the official-style monthly set and each link l:
  surge(l)  = max over on-ramps attached to links l..l+3 (at or just downstream of l) of
              mean flow(T-10..T) / mean flow(T-40..T-30)
  rflow(l)  = same ramps, mean flow(T-10..T) / time-of-day climatology of that ramp
Then, with the out-of-fold lgb_v1 onset probability p (cache/cand_eval2.pkl), a logistic model of the T+30 truth
on [logit p] vs [logit p, ramp features] is fit on two folds and scored on the third (log-loss and the window
IoU of top-k on the refined probability).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pyarrow.dataset as pds

import features_v1 as F1
from common import CACHE, H, PANELS, REL, STEP, T0, aggregate
from model import best_k_set, iou


def ramp_flows(panel: str) -> tuple[np.ndarray, list[str]]:
    """Dense [time, ramp] on-ramp flow on train (NaN where missing)."""
    root = REL / "corridors" / panel / "train" / "ramp_states"
    d = pds.dataset(str(root), format="parquet", partitioning="hive")
    t = d.to_table(columns=["timestamp", "ramp_link_id", "ramp_type", "flow_vph"], filter=pds.field("ramp_type") == "OR").to_pandas()
    ramps = sorted(t.ramp_link_id.unique())
    pos = {r: i for i, r in enumerate(ramps)}
    n_t = int((pd.Timestamp("2031-03-01T00:00:00Z") - T0) / STEP)
    out = np.full((n_t, len(ramps)), np.nan, np.float32)
    ti = ((pd.to_datetime(t.timestamp, utc=True) - T0) // STEP).to_numpy()
    out[ti, t.ramp_link_id.map(pos).to_numpy()] = t.flow_vph.to_numpy(np.float32)
    return out, ramps


def main() -> None:
    c2 = pickle.load(open(CACHE / "cand_eval2.pkl", "rb"))["on"]
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    rows = []
    for panel in PANELS:
        st = F1.Static(panel)
        lpos = {k: i for i, k in enumerate(st.link_ids)}
        amap = pd.read_csv(REL / "corridors" / panel / "network" / "ramp_attachment_map.csv")
        fl, ramps = ramp_flows(panel)
        att = amap.set_index("ramp_link_id").nearest_mainline_link_id.astype(str).map(lpos)
        r_link = np.array([att.get(r, np.nan) for r in ramps], float)
        tod = np.arange(fl.shape[0]) % 288
        dt = (F1.dow_of(np.arange(fl.shape[0])) >= 5).astype(int)
        for (p, t), r in c2.items():
            if p != panel:
                continue
            assert t + 1 <= fl.shape[0]
            k = m2f[r["month"]]
            # climatology of each ramp at this time of day from the other folds' months, rows of those months only
            now = np.nanmean(fl[t - 2:t + 1], axis=0)              # T-10 .. T
            before = np.nanmean(fl[t - 8:t - 5], axis=0)           # T-40 .. T-30
            surge = now / np.maximum(before, 50.0)
            L = st.L
            s_l = np.full(L, np.nan); f_l = np.full(L, np.nan)
            for j in range(len(ramps)):
                if np.isnan(r_link[j]) or np.isnan(surge[j]):
                    continue
                a = int(r_link[j])
                for l in range(max(a - 3, 0), a + 1):
                    s_l[l] = np.nanmax([s_l[l], surge[j]])
                    f_l[l] = np.nanmax([f_l[l], now[j]])
            rows.append({"panel": panel, "t": t, "month": r["month"], "fold": k, "p": r["v1_0"],
                         "y": (r["y"] & r["e"])[H - 1], "e": r["e"][H - 1], "surge": np.clip(np.nan_to_num(s_l, nan=1.0), 0.05, 20.0),
                         "rflow": np.nan_to_num(f_l, nan=0.0), "Y": r["y"], "E": r["e"]})
        print(panel, flush=True)
    pickle.dump(rows, open(CACHE / "ramp_check.pkl", "wb"))
    evaluate(rows)


def evaluate(rows) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    def X(r, use):
        lp = np.log(np.clip(r["p"], 1e-4, 1 - 1e-4) / (1 - np.clip(r["p"], 1e-4, 1 - 1e-4)))
        cols = [lp]
        if use:
            cols += [np.log(r["surge"]), np.log1p(r["rflow"] / 1000.0)]
        return np.column_stack(cols)
    allY = np.concatenate([r["y"] for r in rows]).astype(int)
    print("link rows", len(allY), "positives", int(allY.sum()))
    s = np.concatenate([np.log(r["surge"]) for r in rows])
    print("AUC surge alone %.3f, rflow alone %.3f, v1 p %.3f" % (roc_auc_score(allY, s),
          roc_auc_score(allY, np.concatenate([r["rflow"] for r in rows])), roc_auc_score(allY, np.concatenate([r["p"] for r in rows]))))
    meta, base, new, ll0, ll1 = [], [], [], [], []
    for k in "ABC":
        tr = [r for r in rows if r["fold"] != k]; te = [r for r in rows if r["fold"] == k]
        for use, store in ((False, ll0), (True, ll1)):
            m = LogisticRegression(C=1.0, max_iter=500).fit(np.vstack([X(r, use) for r in tr]), np.concatenate([r["y"] for r in tr]).astype(int))
            q = m.predict_proba(np.vstack([X(r, use) for r in te]))[:, 1]
            store.append(log_loss(np.concatenate([r["y"] for r in te]).astype(int), q, labels=[0, 1]) * len(q))
            if use:
                print(f"fold {k} coef {np.round(m.coef_[0], 3)}")
                i = 0
                for r in te:
                    n = len(r["p"]); qq = q[i:i + n]; i += n
                    pr = np.zeros(r["Y"].shape, bool); pr[H - 1] = best_k_set(qq)
                    p0 = np.zeros(r["Y"].shape, bool); p0[H - 1] = best_k_set(r["p"])
                    new.append(iou(pr, r["Y"], r["E"])); base.append(iou(p0, r["Y"], r["E"]))
                    meta.append({"panel": r["panel"], "month": r["month"], "condition": "queue_onset", "t": r["t"]})
    n = len(allY)
    print("OOF log-loss per row: v1 only %.5f, + ramp %.5f" % (sum(ll0) / n, sum(ll1) / n))
    meta = pd.DataFrame(meta); b = np.array(base); a = np.array(new); d = a - b
    print("onset mean IoU v1 %.4f -> ramp-refined %.4f (diff %+.4f +- %.4f, better %d worse %d)" %
          (b.mean(), a.mean(), d.mean(), d.std() / np.sqrt(len(d)), (d > 0).sum(), (d < 0).sum()))


if __name__ == "__main__":
    main()
