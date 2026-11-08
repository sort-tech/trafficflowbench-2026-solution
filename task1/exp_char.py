"""Guide 3-2 check: characteristic-line baseline inside the simulated Task 2 gaps.

A gap cell (t, link i) lies between the last observed slot T and the next one
T2 = T + 19. Free flow carries values downstream at the link free speed, so the
cell sees the field at T a distance vf*(t-T) upstream and the field at T2 a
distance vf*(T2-t) downstream. Congestion carries values upstream at
w = -18 km/h, so the cell sees the field at T a distance 18*(t-T) downstream
and the field at T2 18*(T2-t) upstream. The two ends are weighted by time
distance; the state is chosen from the own-link temporal interpolation
(> 80 free, <= 60 congested, linear mix in between). The field at T (T2) is
the last (first) observation of each link at or before T (at or after T2)
within 6 slots, interpolated in milepost position; off-corridor -> own-link
interpolation.

Compared with the current final eval predictions (x3g05) on the ds4 eval gap
cells: RMSE, blends, and a 2-fold-by-day stacking check of information gain.

    python -I exp_char.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightgbm as lgb
import numpy as np
import pandas as pd

import common
import train_gbm as TG
from pipeline import reconcile, SPEED_CAP

HERE = Path(__file__).resolve().parent
DS = HERE / "ds4"
FINAL = "x3g05"
W_CONG = 18.0
LOOK = 6


def side_field(x, gaps, slot, forward):
    """(n, nl) value of each link at slot (last obs at/before, or first at/after)."""
    nT, nl = x.shape
    out = np.full((len(slot), nl), np.nan, np.float32)
    for k in range(LOOK + 1):
        s = slot - k if not forward else slot + k
        ok = (s >= 0) & (s < nT)
        sc = np.clip(s, 0, nT - 1)
        val = np.where(ok[:, None] & ~gaps[sc][:, None], x[sc], np.nan)
        fill = np.isnan(out) & np.isfinite(val)
        out[fill] = val[fill]
    return out


def sample(field, xs, xq):
    """Linear interpolation of each row of field (links at xs) at positions xq; NaN outside."""
    out = np.full(len(xq), np.nan, np.float32)
    for r in range(len(xq)):
        f = field[r]
        m = np.isfinite(f)
        if m.sum() < 1 or xq[r] < xs[m][0] - 0.5 or xq[r] > xs[m][-1] + 0.5:
            continue
        out[r] = np.interp(xq[r], xs[m], f[m])
    return out


def char_baseline(p, ev):
    d = common.load(p, "train")
    net = common.network(p)
    lanes = net.lanes.to_numpy(np.float32)
    nd, nl = len(d["dates"]), len(lanes)
    xs = (net.length_km.cumsum() - net.length_km / 2).to_numpy(np.float64)
    vf = net.free_speed_kmh.to_numpy(np.float64)
    v = d["obs_speed_kmh"].reshape(-1, nl)
    q = (d["obs_flow_vph"] / lanes).reshape(-1, nl)
    g = ev.win_cond.to_numpy() > 0
    e = ev[g]
    tt = e.day.to_numpy().astype(np.int64) * common.T + e.t.to_numpy()
    li = e.li.to_numpy()
    wp = e.win_pos.to_numpy().astype(np.int64)
    T0 = tt - wp
    T2 = T0 + 19
    # ds4 windows come from the eval rows themselves (win_pos = slot - T)
    gflat = np.zeros(nd * common.T, bool)
    for a in np.unique(T0):
        gflat[a + 1:a + 19] = True
    dt0 = wp * 5 / 60.0
    dt2 = (19 - wp) * 5 / 60.0
    res = {}
    for name, x in (("v", v), ("q", q)):
        f0 = side_field(x, gflat, T0, False)
        f2 = side_field(x, gflat, T2, True)
        free0 = sample(f0, xs, xs[li] - vf[li] * dt0)
        free2 = sample(f2, xs, xs[li] + vf[li] * dt2)
        cong0 = sample(f0, xs, xs[li] + W_CONG * dt0)
        cong2 = sample(f2, xs, xs[li] - W_CONG * dt2)
        w0 = (19 - wp) / 19.0

        def two(a, b):
            out = w0 * a + (1 - w0) * b
            out = np.where(np.isnan(a), b, out)
            return np.where(np.isnan(b), a, out)

        res[name + "_free"] = two(free0, free2)
        res[name + "_cong"] = two(cong0, cong2)
    bv = e.base_v.to_numpy()
    lam = np.clip((80.0 - bv) / 20.0, 0, 1)  # 0 free .. 1 congested
    for name, base in (("v", bv), ("q", e.base_q.to_numpy())):
        fr = np.where(np.isnan(res[name + "_free"]), base, res[name + "_free"])
        cg = np.where(np.isnan(res[name + "_cong"]), base, res[name + "_cong"])
        res[name] = (1 - lam) * fr + lam * cg
    return g, pd.DataFrame({"cv": res["v"], "cq": res["q"], "cv_free": res["v_free"], "cv_cong": res["v_cong"],
                            "cq_free": res["q_free"], "cq_cong": res["q_cong"]})


def rmse(a):
    return float(np.sqrt(np.nanmean(a * a)))


def main():
    parts = []
    for p in common.PANELS:
        if p.startswith("D12_I405"):
            continue
        cols = ["day", "t", "li", "y_v", "y_q", "base_v", "base_q", "win_cond", "win_pos", "regime", "tod", "pos",
                "lanes", "vf", "cap_l"]
        ev = pd.read_parquet(DS / f"eval_{p}.parquet", columns=cols)
        g, C = char_baseline(p, ev)
        pr = pd.read_parquet(HERE / "preds" / f"{FINAL}_{p}.parquet")
        sp = pd.read_parquet(HERE / "preds" / f"gbm9gap_{p}.parquet")
        e = ev[g].reset_index(drop=True)
        e = pd.concat([e, C], axis=1)
        e["fv"], e["fq"], e["fk"] = pr.pv.to_numpy()[g], pr.pq.to_numpy()[g], pr.pk.to_numpy()[g]
        e["gv"], e["gq"] = sp.pv.to_numpy()[g], sp.pq.to_numpy()[g]
        e["panel"] = p
        e["row"] = np.nonzero(g)[0]
        parts.append(e)
        print(p, int(g.sum()), flush=True)
    E = pd.concat(parts, ignore_index=True)
    cong = E.y_v < 60
    print("gap cells", len(E), "congested share", round(float(cong.mean()), 3))
    rep = []
    for name, cv, cq in (("own interp", E.base_v, E.base_q), ("characteristic", E.cv, E.cq),
                         ("gbm9gap", E.gv, E.gq), ("final x3g05", E.fv, E.fq)):
        ev_, eq_ = cv - E.y_v, cq - E.y_q
        rep.append((name, rmse(ev_), rmse(ev_[cong]), rmse(ev_[~cong]), rmse(eq_), rmse(eq_[cong]), rmse(eq_[~cong])))
    for a in (0.1, 0.2, 0.3):
        bv, bq = E.fv + a * (E.cv - E.fv), E.fq + a * (E.cq - E.fq)
        ev_, eq_ = bv - E.y_v, bq - E.y_q
        rep.append((f"final + {a} char", rmse(ev_), rmse(ev_[cong]), rmse(ev_[~cong]), rmse(eq_), rmse(eq_[cong]),
                    rmse(eq_[~cong])))
    # information gain: residual model on the final prediction, 2 folds by day, with / without char columns
    base_cols = ["fv", "fq", "base_v", "base_q", "win_pos", "win_cond", "tod", "pos", "lanes", "vf", "cap_l"]
    char_cols = ["cv", "cq", "cv_free", "cv_cong", "cq_free", "cq_cong"]
    prm = dict(objective="l2", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=common.THREADS, seed=1,
               deterministic=True, force_row_wise=True, verbose=-1)
    fold = (E.day.to_numpy() // 9) % 2
    stk = {}
    for tag, cols in (("stack no char", base_cols), ("stack + char", base_cols + char_cols)):
        out = {"v": np.zeros(len(E)), "q": np.zeros(len(E))}
        for f in (0, 1):
            tr, te = fold != f, fold == f
            for k, y, m in (("v", E.y_v - E.fv, E.fv), ("q", E.y_q - E.fq, E.fq)):
                mdl = lgb.train(prm, lgb.Dataset(E.loc[tr, cols], y[tr].to_numpy(np.float32)), 300)
                out[k][te] = m[te] + mdl.predict(E.loc[te, cols])
        stk[tag] = out
        ev_, eq_ = out["v"] - E.y_v, out["q"] - E.y_q
        rep.append((tag, rmse(ev_), rmse(ev_[cong]), rmse(ev_[~cong]), rmse(eq_), rmse(eq_[cong]), rmse(eq_[~cong])))
    R = pd.DataFrame(rep, columns=["method", "v", "v_cong", "v_free", "q", "q_cong", "q_free"])
    print(R.round(3).to_string())
    # proxy contribution: replace gap cells of the final prediction
    variants = {"final": (E.fv, E.fq), "final + 0.2 char": (E.fv + 0.2 * (E.cv - E.fv), E.fq + 0.2 * (E.cq - E.fq)),
                "stack no char": (stk["stack no char"]["v"], stk["stack no char"]["q"]),
                "stack + char": (stk["stack + char"]["v"], stk["stack + char"]["q"])}
    rows = []
    for p in common.PANELS:
        ev = pd.read_parquet(DS / f"eval_{p}.parquet", columns=["day", "t", "li"])
        pr = pd.read_parquet(HERE / "preds" / f"{FINAL}_{p}.parquet")
        pe = TG.PanelEval(p, ev)
        d = common.load(p, "train")
        li = ev.li.to_numpy()
        vmin = np.nanmin(d["true_speed"], axis=(0, 1))
        qmax = np.nanmax(d["true_flow"], axis=(0, 1)) / pe.lanes
        sel = (E.panel == p).to_numpy()
        for name, (vv, qq) in variants.items():
            pv, pq = pr.pv.to_numpy().copy(), pr.pq.to_numpy().copy()
            if sel.any():
                pv[E.row.to_numpy()[sel]] = np.asarray(vv)[sel]
                pq[E.row.to_numpy()[sel]] = np.asarray(qq)[sel]
            pv = np.clip(pv, vmin[li], SPEED_CAP)
            v, q = reconcile(pv, pq, pr.pk.to_numpy(), 1.0, 0.2, 0.8, 80.0)
            r = pe(v, np.clip(q, 1.0, 1.1 * qmax[li]))
            r["variant"] = name
            rows.append(r)
            if name == "final":
                # guide 3-1: arithmetic instead of geometric blend of q and k*v (reconcile = geometric)
                b = np.where(pv < 80, 0.8, 0.2)
                qa = (1 - b) * np.maximum(pq, 1) + b * np.maximum(pr.pk.to_numpy(), 1e-3) * pv
                r = pe(pv, np.clip(qa, 1.0, 1.1 * qmax[li]))
                r["variant"] = "final arith blend"
                rows.append(r)
    df = pd.concat(rows)
    df["family"] = df.panel.map(common.FAMILY)
    df["contrib"] = 0.35 * df.S_state + 0.15 * df.S_physics
    agg = df.groupby(["variant", "regime", "family"]).contrib.mean().groupby(["variant", "regime"]).mean().unstack()
    agg["all"] = agg.mean(1)
    print(agg.round(5).to_string())
    print((agg - agg.loc["final"]).round(5).to_string())
    (HERE / "logs").mkdir(parents=True, exist_ok=True)
    R.to_csv(HERE / "logs" / "exp_char_rmse.csv", index=False)
    df.to_csv(HERE / "logs" / "exp_char.csv", index=False)


if __name__ == "__main__":
    main()
