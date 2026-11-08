"""Task 2 gap specialist with v5 window features (queue polygon + moving frame).

Same fit rows as ds9 (ds10 = ds9 rows, FEAT=5, eval rows inside the gaps only), same
params and fixed rounds as gbm9gap; only the g5_* columns are new. Gap predictions
replace gbm9gap inside the gaps of the x3 stage-2 average, scored like x3g05.

    DS_DIR=ds10 GAPS=win FEAT=5 N_PER_REGIME=200000 GAP_OS=8 EXTRA_WIN=1 GAP_ONLY=1 \
        EVAL_GAP_ONLY=1 python -I make_dataset.py <8 panels>
    python -I exp_gap5.py <tag> [drop prefixes, comma separated]
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("FIXED_ROUNDS", "y_v:3000,y_q:2996,y_k:3000")
import lightgbm as lgb
import numpy as np
import pandas as pd

import common
import metrics
import train_gbm as TG
from pipeline import reconcile

HERE = Path(__file__).resolve().parent
DS = HERE / os.environ.get("GAP_DS", "ds10")  # ds11: + ramp features (FEAT=6)
REF = HERE / "ds4"  # full eval rows, identical to ds7/ds9
BASE = "gbm8s2_x3"
OLD = "gbm9gap"


def train(tag, drop):
    TG.DS = DS
    panels = [p for p in common.PANELS if (DS / f"fit_{p}.parquet").exists()]
    fit = TG.load_fit(panels)
    cols = [c for c in TG.feat_cols(fit) if not any(c.startswith(d) for d in drop)]
    print("fit rows", len(fit), "features", len(cols), "new", sum(c.startswith("g5_") for c in cols), flush=True)
    fr = TG.fixed_rounds()
    models = {}
    for tgt in TG.TARGETS:
        f = HERE / "models" / f"{tag}_{tgt}.txt"
        if f.exists():
            models[tgt] = lgb.Booster(model_file=str(f))
            continue
        w = None
        if tgt == "y_k":
            y = TG.k_true(fit) - TG.k_base(fit)
            w = (fit.lanes * fit.len).to_numpy(np.float32)
        else:
            y = (fit[tgt] - fit["base_" + tgt[-1]]).to_numpy(np.float32)
        t0 = time.time()
        m = lgb.train(TG.params(), lgb.Dataset(fit[cols], y, weight=w), fr[tgt])
        m.save_model(str(f))
        print(tgt, fr[tgt], "rounds", f"{time.time()-t0:.0f}s", flush=True)
        models[tgt] = m
    return models, cols


def rmse(e):
    return float(np.sqrt(np.mean(e * e)))


def main():
    tag = sys.argv[1]
    drop = [d for d in (sys.argv[2].split(",") if len(sys.argv) > 2 else []) if d]
    models, cols = train(tag, drop)
    rep, rows = [], {}
    arms = {"x3g05": (OLD, 0.5), f"{tag} 0.5": (tag, 0.5), f"{tag} 1.0": (tag, 1.0), "x3 only": (None, 0.0)}
    for p in common.PANELS:
        ev = pd.read_parquet(REF / f"eval_{p}.parquet", columns=["day", "t", "li", "win_cond", "y_v", "y_q", "regime"])
        base = pd.read_parquet(HERE / "preds" / f"{BASE}_{p}.parquet")
        old = pd.read_parquet(HERE / "preds" / f"{OLD}_{p}.parquet")
        g = ev.win_cond.to_numpy() > 0
        new = None
        if g.any():
            X = pd.read_parquet(DS / f"eval_{p}.parquet")
            e = ev[g]
            assert (X.day.to_numpy() == e.day.to_numpy()).all() and (X.t.to_numpy() == e.t.to_numpy()).all() \
                and (X.li.to_numpy() == e.li.to_numpy()).all()
            X["rate"] = X.regime.map(TG.RATE).astype(np.float32)
            Xc = X[cols]
            new = {"pv": X.base_v.to_numpy() + models["y_v"].predict(Xc, num_threads=common.THREADS),
                   "pq": X.base_q.to_numpy() + models["y_q"].predict(Xc, num_threads=common.THREADS),
                   "pk": TG.k_base(X) + models["y_k"].predict(Xc, num_threads=common.THREADS)}
            pd.DataFrame({k: v.astype(np.float32) for k, v in new.items()}).assign(
                day=X.day, t=X.t, li=X.li).to_parquet(HERE / "preds" / f"{tag}gaponly_{p}.parquet", index=False)
            yv, yq = e.y_v.to_numpy(), e.y_q.to_numpy()
            for name, src in (("x3", base), (OLD, old)):
                rep.append((p, name, src.pv.to_numpy()[g] - yv, src.pq.to_numpy()[g] - yq, yv))
            rep.append((p, tag, new["pv"] - yv, new["pq"] - yq, yv))
            for name, w in (("x3g05", None), (f"{tag} 0.5", 0.5)):
                b = old if w is None else None
                mv = 0.5 * (b.pv.to_numpy()[g] if b is not None else new["pv"]) + 0.5 * base.pv.to_numpy()[g]
                mq = 0.5 * (b.pq.to_numpy()[g] if b is not None else new["pq"]) + 0.5 * base.pq.to_numpy()[g]
                rep.append((p, name, mv - yv, mq - yq, yv))
        pe = TG.PanelEval(p, ev)
        for arm, (src, w) in arms.items():
            pv, pq, pk = (base[c].to_numpy().copy() for c in ("pv", "pq", "pk"))
            if src is not None and g.any():
                s = old if src == OLD else None
                for c, a in (("pv", pv), ("pq", pq), ("pk", pk)):
                    sv = s[c].to_numpy()[g] if s is not None else new[c]
                    a[g] = w * sv + (1 - w) * a[g]
            v, q = reconcile(pv, pq, pk, 1.0, 0.2, 0.8, 80.0)
            rows.setdefault(arm, []).append(pe(v, q))
        print(p, flush=True)
    out = []
    for arm, r in rows.items():
        df = pd.concat(r)
        s = metrics.summarize(df)
        out.append((arm, metrics.contribution(s), s["S_state"], s["S_LWR"] if "S_LWR" in s else np.nan))
    S = pd.DataFrame(out, columns=["arm", "contrib", "S_state", "S_LWR"]).set_index("arm")
    S["d_vs_x3g05"] = S.contrib - S.loc["x3g05", "contrib"]
    print(S.round(5).to_string())
    R = []
    for name in dict.fromkeys(r[1] for r in rep):
        ev_ = np.concatenate([r[2] for r in rep if r[1] == name])
        eq_ = np.concatenate([r[3] for r in rep if r[1] == name])
        yv = np.concatenate([r[4] for r in rep if r[1] == name])
        c = yv < 60
        R.append((name, rmse(ev_), rmse(ev_[c]), rmse(ev_[~c]), rmse(eq_), rmse(eq_[c]), rmse(eq_[~c])))
    R = pd.DataFrame(R, columns=["pred", "v", "v_cong", "v_free", "q", "q_cong", "q_free"])
    print("gap cells", len(yv), "congested", round(float(c.mean()), 3))
    print(R.round(3).to_string())
    imp = pd.Series(models["y_v"].feature_importance("gain"), index=cols)
    print("y_v gain share of g5_*", round(float(imp[imp.index.str.startswith("g5_")].sum() / imp.sum()), 4))
    print(imp[imp.index.str.startswith("g5_")].sort_values(ascending=False).div(imp.sum()).round(4).to_string())
    (HERE / "logs").mkdir(parents=True, exist_ok=True)
    S.to_csv(HERE / "logs" / f"exp_gap5_{tag}.csv")
    R.to_csv(HERE / "logs" / f"exp_gap5_{tag}_rmse.csv", index=False)


if __name__ == "__main__":
    main()
