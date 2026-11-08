"""Onset: LambdaRank link ranking vs binary classifier, official-style monthly out-of-fold evaluation.

Per fold, a binary model gives p (calibrated-ish) and a lambdarank model gives a within-window score.
Decisions compared on the same windows:
  bin   : expected-IoU top-k on p
  rank  : order by rank score, k = size of the bin decision
  blend : order by mean of within-window ranks of p and rank score, k from bin
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import CACHE, H, PANELS, STEP, T0, Panel, aggregate, select_windows
from features import FOLDS, Clim, Static, link_features, online_features, onset_rows, train_window_arrays
from model import PARAMS, VARIANTS, best_k_set, feature_cols, iou

VAR = sys.argv[1] if len(sys.argv) > 1 else "onphys"
MD = Path(__file__).resolve().parent / "models" / "cv"


def train_models(k: str):
    pb, pr = MD / f"onset_{VAR}_{k}.txt", MD / f"onsetrank_{VAR}_{k}.txt"
    parts = []
    for p in PANELS:
        d = pd.read_parquet(CACHE / f"ds_{p}.parquet", filters=[("cond", "=", 1)])
        d = d[d.fold != k]
        d["panel"] = p
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    feats = [f for f in feature_cols(df) if f not in VARIANTS[VAR]]
    if not pb.exists():
        tr = df[df.vis == 1]
        ds = lgb.Dataset(tr[feats], label=tr.y.to_numpy(), categorical_feature=["pid"])
        lgb.train(PARAMS, ds, 300).save_model(str(pb))
    if not pr.exists():
        df = df.sort_values(["panel", "t", "link"])
        lab = np.where(df.vis.to_numpy() == 1, df.y.to_numpy(), 0)
        grp = df.groupby(["panel", "t"], sort=False).size().to_numpy()
        prm = dict(PARAMS, objective="lambdarank", metric="ndcg", eval_at=[3, 5], lambdarank_truncation_level=20)
        ds = lgb.Dataset(df[feats], label=lab, group=grp, categorical_feature=["pid"])
        lgb.train(prm, ds, 300).save_model(str(pr))
        print("fit rank", k, flush=True)
    return lgb.Booster(model_file=str(pb)), lgb.Booster(model_file=str(pr))


def main() -> None:
    MD.mkdir(parents=True, exist_ok=True)
    models = {k: train_models(k) for k in FOLDS}
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in FOLDS.items() for m in ms}
    months = [m for ms in FOLDS.values() for m in ms]
    rows = []
    for panel in PANELS:
        st = Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        clims = {k: Clim(P, qc, [m for f, ms in FOLDS.items() if f != k for m in ms], onset_t) for k in FOLDS}
        for mon in months:
            start = int((pd.Timestamp(f"{mon}-01T00:00:00Z") - T0) / STEP)
            end = int((pd.Timestamp(f"{mon}-01T00:00:00Z") + pd.offsets.MonthBegin(1) - T0) / STEP)
            sel = select_windows(P, qc, max_per_condition=5, start=max(start, 12), end=end)
            k = m2f[mon]; clim = clims[k]; mb, mr = models[k]
            for w in sel[sel.condition == "queue_onset"].itertuples():
                rng = np.random.default_rng([PANELS.index(panel), int(w.t)])
                arr = train_window_arrays(P, w.t, rng, [0.2, 0.3, 0.5][rng.integers(3)])
                df = onset_rows(link_features(arr, st, clim, w.t, online_features(P.speed, P.vcut, int(w.t), start)), clim, w.t)
                p = mb.predict(df[mb.feature_name()], num_threads=3)
                s = mr.predict(df[mr.feature_name()], num_threads=3)
                sel_b = best_k_set(p)
                kk = int(sel_b.sum())
                y, e = P.q[w.t + H], P.elig[w.t + H]
                def topk(score):
                    m = np.zeros(st.L, bool)
                    if kk:
                        m[np.argsort(-score, kind="stable")[:kk]] = True
                    return m
                rp = pd.Series(p).rank().to_numpy(); rs = pd.Series(s).rank().to_numpy()
                rows.append({"panel": panel, "month": mon, "condition": "queue_onset", "t": w.t,
                             "bin": iou(sel_b, y, e), "rank": iou(topk(s), y, e), "blend": iou(topk(rp + rs), y, e)})
        print(panel, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / f"onset_rank_{VAR}.csv", index=False)
    for c in ("bin", "rank", "blend"):
        print(VAR, c, round(df.groupby("panel")[c].mean().mean(), 4))


if __name__ == "__main__":
    main()
