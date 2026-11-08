"""Month-fold cross-validation of the Task 2 models on train pseudo windows.

Folds are 3-month blocks (A: Jun-Aug, B: Sep-Nov, C: Dec-Feb). Each fold's greedy windows
(the imitated official selection) are scored with models fit on the other two folds.
Labels: observed speed <= v_cut on eligible cells (proxy for the noise-free truth).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE, PANELS, aggregate
from features import FOLDS
from model import SEED, TOD_FEATURES, best_k_set, feature_cols, fit, iou

DROP: set[str] = set()
DROP_SETS = {"tod": TOD_FEATURES}


def load(cond: int, cols: list[str] | None = None) -> pd.DataFrame:
    parts = []
    for p in PANELS:
        d = pd.read_parquet(CACHE / f"ds_{p}.parquet", columns=cols)
        d = d[d.cond == cond]
        d["panel"] = p
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def decide(g: pd.DataFrame, rule: str) -> np.ndarray:
    p = g["p"].to_numpy()
    if rule == "k":
        return best_k_set(p)
    if rule.startswith("thr"):
        return p >= float(rule[3:])
    raise ValueError(rule)


def score(pred: pd.DataFrame, rules: list[str], cond_name: str) -> pd.DataFrame:
    rows = []
    for (panel, t), g in pred.groupby(["panel", "t"], sort=False):
        r = {"panel": panel, "t": t, "condition": cond_name}
        for rule in rules:
            r[rule] = iou(decide(g, rule), g.y.to_numpy(), g.elig.to_numpy())
        if "pers" in g:
            r["pers_T"] = iou(g.pers.to_numpy(), g.y.to_numpy(), g.elig.to_numpy())
        rows.append(r)
    return pd.DataFrame(rows)


def run(cond: int, rounds: int, rules: list[str], params: dict | None = None, tag: str = "", n_seeds: int = 1) -> pd.DataFrame:
    df = load(cond)
    feats = [f for f in feature_cols(df) if f not in DROP]
    cond_name = "queue_onset" if cond == 1 else "queue_ongoing"
    res = []
    imps = []
    preds = []
    for k in FOLDS:
        tr = df[(df.fold != k) & (df.vis == 1) & ((df.keep == 1) | (cond == 1))]
        te = df[(df.fold == k) & (df.greedy == 1)].copy()
        ps = []
        for i in range(n_seeds):
            m = fit(tr, feats, rounds, dict(params or {}, seed=SEED + 1000 * i))
            ps.append(m.predict(te[feats], num_threads=3))
        te["p"] = np.mean(ps, axis=0)
        te["p_s0"] = ps[0]
        if cond == 0:
            te.loc[te.keep == 0, ["p", "p_s0"]] = 0.0
            te["pers"] = (te.r_0 <= 1.0).astype(int)
        res.append(score(te, rules, cond_name))
        keepcols = ["panel", "t", "link", "h", "p", "p_s0", "y", "elig", "vis", "r_0", "dd_0", "du_0", "tail_rel", "to_tail", "to_head", "keep", "tod"]
        preds.append(te[[c for c in keepcols if c in te.columns]])
        imps.append(pd.Series(m.feature_importance("gain"), index=feats))
        print(tag, cond_name, k, "train rows", len(tr), "test windows", te.groupby(["panel", "t"]).ngroups, flush=True)
    out = pd.concat(res, ignore_index=True)
    pd.concat(preds, ignore_index=True).to_parquet(CACHE / f"cvpred_{cond_name}{tag}.parquet", index=False)
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    print(imp.head(25).round(0).to_string())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["onset", "ongoing", "both"], default="both")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--drop", default="", help="feature set name in DROP_SETS or comma list")
    args = ap.parse_args()
    for item in filter(None, args.drop.split(",")):
        DROP.update(DROP_SETS.get(item, {item}))
    rules = ["k", "thr0.3", "thr0.4", "thr0.5"]
    outs = []
    if args.cond in ("onset", "both"):
        outs.append(run(1, args.rounds, rules, tag=args.tag, n_seeds=args.seeds))
    if args.cond in ("ongoing", "both"):
        outs.append(run(0, args.rounds, rules, tag=args.tag, n_seeds=args.seeds))
    res = pd.concat(outs, ignore_index=True)
    res.to_csv(CACHE / f"cv_{args.cond}.csv", index=False)
    if "pers_T" in res:
        res["pers_T"] = res["pers_T"].fillna(0.0)
    for rule in rules + (["pers_T"] if "pers_T" in res else []):
        by = res.groupby(["condition", "panel"])[rule].mean().groupby("condition").mean().round(4).to_dict()
        line = f"{rule:8s} " + " ".join(f"{c}={v:.4f}" for c, v in by.items())
        if res.condition.nunique() == 2:
            line += f"  S={aggregate(res, rule)['S_queue']:.4f}"
        print(line)


if __name__ == "__main__":
    main()
