"""Fit LightGBM residual models on the fit fold, score on the eval fold.

y_v - base_v and y_q - base_q (per lane), base = own-link temporal interpolation.
"""
import sys, time, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import lightgbm as lgb
import common, metrics

HERE = Path(__file__).resolve().parent
DS = HERE / "ds"
META = ["panel", "regime", "day", "t", "li", "y_v", "y_q"]
RATE = {"R1": 0.2, "R2": 0.3, "R3": 0.5}
SEED = int(os.environ.get("TFB_SEED", 20261025))


def load_fit(panels):
    fit = pd.concat([pd.read_parquet(DS / f"fit_{p}.parquet") for p in panels
                     if (DS / f"fit_{p}.parquet").exists()], ignore_index=True)
    fit["rate"] = fit.regime.map(RATE).astype(np.float32)
    return fit


# DROP="g5_,g7_": leave out feature columns with these prefixes (gap specialists on ds11)
DROP = [x for x in os.environ.get("DROP", "").split(",") if x]


def feat_cols(df):
    return [c for c in df.columns if c not in META and not any(c.startswith(x) for x in DROP)]


def params(obj="l2"):
    return dict(objective=obj, learning_rate=LR, num_leaves=int(os.environ.get("NUM_LEAVES", 255)),
                min_data_in_leaf=int(os.environ.get("MIN_DATA", 200)),
                feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                num_threads=common.THREADS, seed=SEED, deterministic=True, force_row_wise=True, verbose=-1)


ROUNDS = int(os.environ.get("ROUNDS", 2000))
LR = float(os.environ.get("LR", 0.05))
TARGETS = os.environ.get("TARGETS", "y_v,y_q,y_k").split(",")


def fixed_rounds():
    """FIXED_ROUNDS="y_v:3983,y_q:3999,y_k:4000": train every row for exactly that many
    rounds, no early stopping. Early stopping lands at different rounds with different
    thread counts; a fixed count keeps reruns on other machines within float noise."""
    s = os.environ.get("FIXED_ROUNDS", "")
    return {k: int(v) for k, v in (x.split(":") for x in s.split(",") if x)}


def train_one(prm, X, y, w, va, rounds, tgt):
    fr = fixed_rounds()
    if tgt in fr:
        m = lgb.train(prm, lgb.Dataset(X, y, weight=w), fr[tgt])
        print(tgt, "fixed rounds", fr[tgt], flush=True)
        return m
    dtr = lgb.Dataset(X.loc[~va], y[~va], weight=None if w is None else w[~va])
    dva = lgb.Dataset(X.loc[va], y[va], weight=None if w is None else w[va], reference=dtr)
    m = lgb.train(prm, dtr, rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    print(tgt, "best", m.best_iteration, flush=True)
    return m


def k_true(df):
    return (df.y_q / df.y_v.clip(lower=1.0)).to_numpy(np.float32)


def k_base(df):
    return (df.base_q / df.base_v.clip(lower=1.0)).to_numpy(np.float32)


def fit_models(fit, rounds=None, valid_frac=0.05, obj="l2"):
    rounds = rounds or ROUNDS
    cols = feat_cols(fit)
    rng = np.random.default_rng(SEED)
    va = rng.random(len(fit)) < valid_frac
    models = {}
    for tgt in TARGETS:
        w = None
        if tgt == "y_k":
            y = k_true(fit) - k_base(fit)
            # S_LWR sums |dN| in vehicles: weight per-lane density errors by lanes * length
            w = (fit.lanes * fit.len).to_numpy(np.float32)
        else:
            y = (fit[tgt] - fit["base_" + tgt[-1]]).to_numpy(np.float32)
        t0 = time.time()
        m = train_one(params(obj), fit[cols], y, w, va, rounds, tgt)
        print(tgt, "best_iter", m.best_iteration, f"{time.time()-t0:.0f}s", flush=True)
        models[tgt] = m
    return models, cols


def predict(models, cols, X):
    X = X.copy()
    if "rate" not in X:
        X["rate"] = X.regime.map(RATE).astype(np.float32)
    pv = X.base_v.to_numpy() + models["y_v"].predict(X[cols], num_iteration=models["y_v"].best_iteration)
    pq = X.base_q.to_numpy() + models["y_q"].predict(X[cols], num_iteration=models["y_q"].best_iteration)
    pk = None
    if "y_k" in models:
        pk = k_base(X) + models["y_k"].predict(X[cols], num_iteration=models["y_k"].best_iteration)
    return pv, pq, pk


class PanelEval:
    """Scorer over the eval-fold days of one panel, reusable across predictions."""

    def __init__(self, p, ev):
        d = common.load(p, "train")
        self.lanes = common.network(p).lanes.to_numpy(np.float32)
        self.days = np.unique(ev.day.to_numpy())
        pos = {x: i for i, x in enumerate(self.days)}
        self.idx = (ev.day.map(pos).to_numpy(), ev.t.to_numpy(), ev.li.to_numpy())
        self.sc = metrics.Scorer(p, d, self.days)
        self.shape = (len(self.days), common.T, len(self.lanes))

    def __call__(self, v, q):
        ps = np.zeros(self.shape); pq = np.zeros(self.shape)
        ps[self.idx] = np.maximum(v, 1.0)
        pq[self.idx] = np.maximum(q, 0.0) * self.lanes[self.idx[2]]
        return self.sc.score(ps, pq)


def eval_panel(p, pred_v=None, pred_q=None, ev=None):
    """Score eval-fold predictions (per-lane flow) of one panel; None -> base."""
    d = common.load(p, "train")
    net = common.network(p)
    lanes = net.lanes.to_numpy(np.float32)
    if ev is None:
        ev = pd.read_parquet(DS / f"eval_{p}.parquet")
    days = np.unique(ev.day.to_numpy())
    pos = {x: i for i, x in enumerate(days)}
    nd, nl = len(days), len(lanes)
    ps = np.zeros((nd, common.T, nl)); pq = np.zeros((nd, common.T, nl))
    di = ev.day.map(pos).to_numpy()
    v = ev.base_v.to_numpy() if pred_v is None else pred_v
    q = ev.base_q.to_numpy() if pred_q is None else pred_q
    ps[di, ev.t.to_numpy(), ev.li.to_numpy()] = np.maximum(v, 1.0)
    pq[di, ev.t.to_numpy(), ev.li.to_numpy()] = np.maximum(q, 0.0) * lanes[ev.li.to_numpy()]
    sc = metrics.Scorer(p, d, days)
    return sc.score(ps, pq)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "gbm"
    if len(sys.argv) > 2:
        DS = HERE / sys.argv[2]
    panels = [p for p in common.PANELS if (DS / f"eval_{p}.parquet").exists()]
    fit = load_fit(panels)
    print("fit rows", len(fit), "features", len(feat_cols(fit)), flush=True)
    models, cols = fit_models(fit)
    del fit
    for sub in ("models", "preds", "logs"):
        (HERE / sub).mkdir(parents=True, exist_ok=True)
    for k, m in models.items():
        m.save_model(str(HERE / "models" / f"{tag}_{k}.txt"))
    rows_b, rows_m = [], []
    for p in panels:
        ev = pd.read_parquet(DS / f"eval_{p}.parquet")
        rows_b.append(eval_panel(p, ev=ev))
        pv, pq, pk = predict(models, cols, ev)
        out = pd.DataFrame({"day": ev.day, "t": ev.t, "li": ev.li, "pv": pv.astype(np.float32),
                            "pq": pq.astype(np.float32)})
        if pk is not None:
            out["pk"] = pk.astype(np.float32)
        out.to_parquet(HERE / "preds" / f"{tag}_{p}.parquet", index=False)
        rows_m.append(eval_panel(p, pv, pq, ev=ev))
        print(p, "base", metrics.summarize(rows_b[-1]), "\n   gbm", metrics.summarize(rows_m[-1]), flush=True)
    B = pd.concat(rows_b); M = pd.concat(rows_m)
    B.to_csv(HERE / "logs" / f"eval_base.csv", index=False)
    M.to_csv(HERE / "logs" / f"eval_{tag}.csv", index=False)
    sb, sm = metrics.summarize(B), metrics.summarize(M)
    print("BASE", sb, metrics.contribution(sb))
    print("GBM ", sm, metrics.contribution(sm))
