"""Final training and prediction used by build.py.

Training rows come from every train day (actual release masks, simulated Task 2
gaps), N_PER_REGIME target cells per regime and panel. Three LightGBM residual
models are fitted with fixed round counts: speed, per-lane flow and per-lane
density (weighted by lanes * length, the vehicle scale S_LWR sums over). The
density model then moves the implied density q/v toward its own prediction.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import common
import features as F

HERE = Path(__file__).resolve().parent
CONFIG = {
    "name": "gbm_v4",
    "feature_version": 4,
    "seed": 20261025,
    "n_per_regime": 80000,
    "gap_oversample": 4.0,
    "block_days": 21,
    "windows_per_day": 10 / 31,
    "lgb": dict(objective="l2", learning_rate=0.1, num_leaves=255, min_data_in_leaf=200,
                feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                num_threads=common.THREADS, deterministic=True, force_row_wise=True, verbose=-1),
    "rounds": {"y_v": 600, "y_q": 1000, "y_k": 800},
    "reconcile": {"alpha": 1.0, "beta": 0.3, "beta_c": 0.8, "vthr": 80.0},
}
# Same model at half the learning rate, twice the rounds and 1.5x the rows.
CONFIG_V5 = dict(CONFIG, name="gbm_v5", n_per_regime=120000,
                 lgb=dict(CONFIG["lgb"], learning_rate=0.05),
                 rounds={"y_v": 1200, "y_q": 2400, "y_k": 1600})
# gbm7 on train eval: 200k rows per regime, gap cells x4, lr 0.05 (all three models
# still improving at 4000 rounds there), density reconcile beta 0.2 / 0.8.
CONFIG_V7 = dict(CONFIG, name="gbm_v7", n_per_regime=200000,
                 lgb=dict(CONFIG["lgb"], learning_rate=0.05),
                 rounds={"y_v": 5000, "y_q": 5000, "y_k": 5000},
                 reconcile={"alpha": 1.0, "beta": 0.2, "beta_c": 0.8, "vthr": 80.0})
# gbm_v7 + the v7 ramp flow and mainline conservation features (10/30); models pick
# their own columns, so the window columns left out at training are simply unused.
CONFIG_V7R = dict(CONFIG_V7, name="gbm_v7r", feature_version=7)
CONFIGS = {c["name"]: c for c in (CONFIG, CONFIG_V5, CONFIG_V7, CONFIG_V7R)}
TARGETS = ("y_v", "y_q", "y_k")
SPEED_CAP = 120.75
RATE = {"R1": 0.2, "R2": 0.3, "R3": 0.5}
META = ["panel", "regime", "day", "t", "li", "y_v", "y_q"]


def k_base(df):
    return (df.base_q / df.base_v.clip(lower=1.0)).to_numpy(np.float32)


def training_rows(p: str, cfg=CONFIG) -> pd.DataFrame:
    """Sampled target cells of every train day, with simulated Task 2 window gaps."""
    d = common.load(p, "train")
    net = common.network(p)
    lanes = net.lanes.to_numpy(np.float32)
    prof = F.profiles([d], lanes)
    nd, nl = len(d["dates"]), len(lanes)
    idx = common.PANELS.index(p)
    rng = np.random.default_rng(cfg["seed"] + idx)
    wins = common.train_windows(p, d, np.random.default_rng(cfg["seed"] + 200 + idx), cfg["windows_per_day"])
    gaps = np.zeros(nd * common.T, bool)
    for di0, s0, _ in wins:
        a = di0 * common.T + s0
        gaps[a + 1:a + 19] = True
    gaps = gaps.reshape(nd, common.T)
    counts = {r: d["target"][d["regime"] == r].sum() for r in common.REGIMES}
    prob = {r: min(1.0, cfg["n_per_regime"] / max(c, 1)) for r, c in counts.items()}
    parts = []
    B = cfg["block_days"]
    for s in range(0, nd, B):
        lo, hi = max(0, s - 1), min(nd, s + B + 1)
        days = np.arange(lo, hi)
        eb = np.broadcast_to(gaps[days][:, :, None], (len(days), common.T, nl))
        bw = [((di0 - lo) * common.T + s0, c) for di0, s0, c in wins if lo <= di0 < hi]
        blk = F.Block(d, days, net, prof, extra_blank=eb, version=cfg["feature_version"], windows=bw)
        tg = d["target"][days].reshape(len(days) * common.T, nl)
        ti, li = np.nonzero(tg)
        dd = days[ti // common.T]
        reg = d["regime"][dd]
        pr = np.vectorize(prob.get)(reg) * np.where(gaps[dd, ti % common.T], cfg["gap_oversample"], 1.0)
        keep = (dd >= s) & (dd < min(nd, s + B)) & (rng.random(len(ti)) < pr)
        X = blk.features(ti[keep], li[keep], net)
        X["regime"] = reg[keep]
        tt = ti[keep] % common.T
        X["y_v"] = d["true_speed"][dd[keep], tt, li[keep]].astype(np.float32)
        X["y_q"] = (d["true_flow"][dd[keep], tt, li[keep]] / lanes[li[keep]]).astype(np.float32)
        parts.append(X)
    out = pd.concat(parts, ignore_index=True)
    print(f"  {p}: {len(out):,} rows, {len(wins)} simulated windows", flush=True)
    return out


def models_exist(mdir: Path) -> bool:
    return all((mdir / f"{t}.txt").exists() for t in TARGETS)


def train_final(mdir: Path, cfg=CONFIG) -> None:
    mdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fit = pd.concat([training_rows(p, cfg) for p in common.PANELS], ignore_index=True)
    fit["rate"] = fit.regime.map(RATE).astype(np.float32)
    cols = [c for c in fit.columns if c not in META]
    print(f"training rows {len(fit):,}, features {len(cols)}, {time.time()-t0:.0f}s", flush=True)
    for tgt in TARGETS:
        w = None
        if tgt == "y_k":
            y = (fit.y_q / fit.y_v.clip(lower=1.0)).to_numpy(np.float32) - k_base(fit)
            w = (fit.lanes * fit.len).to_numpy(np.float32)
        else:
            y = (fit[tgt] - fit["base_" + tgt[-1]]).to_numpy(np.float32)
        ds = lgb.Dataset(fit[cols], y, weight=w)
        prm = dict(cfg["lgb"], seed=cfg["seed"])
        t = time.time()
        m = lgb.train(prm, ds, cfg["rounds"][tgt])
        m.save_model(str(mdir / f"{tgt}.txt"))
        print(f"{tgt}: {cfg['rounds'][tgt]} rounds {time.time()-t:.0f}s", flush=True)
    (mdir / "config.json").write_text(json.dumps(dict(cfg, columns=cols), indent=1))


def load_models(mdir: Path) -> dict:
    return {t: lgb.Booster(model_file=str(mdir / f"{t}.txt")) for t in TARGETS}


def reconcile(pv, pq, pk, alpha, beta, beta_c=None, vthr=80.0):
    """Move the implied density q/v toward the density model: a fraction beta of the
    log gap (beta_c where predicted speed < vthr), share alpha of it carried by flow."""
    pv = np.maximum(pv, 1.0); pq = np.maximum(pq, 1.0); pk = np.maximum(pk, 1e-3)
    if beta_c is not None:
        beta = np.where(pv < vthr, beta_c, beta)
    dlt = np.log(pk) - np.log(pq / pv)
    return pv * np.exp(-(1 - alpha) * beta * dlt), pq * np.exp(alpha * beta * dlt)


def predict_split(p: str, split: str, models: dict, cfg=CONFIG) -> pd.DataFrame:
    """Template rows of one panel/split with speed_kmh and flow_vph filled."""
    d = common.load(p, split)
    tr = common.load(p, "train")
    net = common.network(p)
    lanes = net.lanes.to_numpy(np.float32)
    prof = F.profiles([tr, d], lanes)
    nd, nl = len(d["dates"]), len(lanes)
    wins = [(di0 * common.T + s0, c) for di0, s0, c in common.split_windows(p, split, d["dates"])]
    blk = F.Block(d, np.arange(nd), net, prof, version=cfg["feature_version"], windows=wins)
    tpl = common.read_template(p, split)
    di, ti, li = common.template_index(tpl, d["dates"], d["links"])
    assert not (np.isnan(di.astype(float)).any() or (li < 0).any())
    X = blk.features(di * common.T + ti, li, net)
    X["rate"] = tpl.mask_regime.map(RATE).to_numpy(np.float32)
    cols = models["y_v"].feature_name()
    Xc = X[cols]
    pv = X.base_v.to_numpy() + models["y_v"].predict(Xc, num_threads=common.THREADS)
    pq = X.base_q.to_numpy() + models["y_q"].predict(Xc, num_threads=common.THREADS)
    pk = k_base(X) + models["y_k"].predict(Xc, num_threads=common.THREADS)
    # The generator caps speed at 120.75 km/h; keep speed inside the link's train
    # range before reconciling so the density step sees the final speed.
    vmin = np.nanmin(tr["true_speed"], axis=(0, 1))
    qmax = np.nanmax(tr["true_flow"], axis=(0, 1)) / lanes
    pv = np.clip(pv, vmin[li], SPEED_CAP)
    rc = cfg["reconcile"]
    v, q = reconcile(pv, pq, pk, rc["alpha"], rc["beta"], rc.get("beta_c"), rc.get("vthr", 80.0))
    q = np.clip(q, 1.0, 1.1 * qmax[li]) * lanes[li]
    out = tpl.copy()
    out["speed_kmh"] = v
    out["flow_vph"] = q
    assert np.isfinite(out.speed_kmh).all() and np.isfinite(out.flow_vph).all()
    return out


def stage2_raw(p: str, split: str, s1: dict, s2, cfg, tt, li, rate, extra_blank=None):
    """Raw two-stage predictions (speed, per-lane flow, per-lane density) at flat cells
    (tt = day*T + slot, li). extra_blank hides further cells before anything is computed."""
    import stage2 as S2
    d = common.load(p, split)
    tr = common.load(p, "train")
    net = common.network(p)
    lanes = net.lanes.to_numpy(np.float32)
    prof = F.profiles([tr, d], lanes)
    nd, nl = len(d["dates"]), len(lanes)
    wins = [(di0 * common.T + s0, c) for di0, s0, c in common.split_windows(p, split, d["dates"])]
    rmp = split_ramps(p, split, d, tr) if cfg["feature_version"] >= 6 else None
    blk = F.Block(d, np.arange(nd), net, prof, ramps=rmp, version=cfg["feature_version"], windows=wins,
                  extra_blank=extra_blank)
    reg = np.repeat(d["regime"], common.T)
    miss = ~np.isfinite(blk.v)
    mi, ml = np.nonzero(miss)
    mrate = np.vectorize(RATE.get)(reg[mi]).astype(np.float32)
    preds = [S2.predict_cells(blk, s1[par], mi, ml, net, mrate) for par in (0, 1)]
    fv = blk.v.copy(); fq = blk.q.copy(); fk = blk.k.copy()
    fv[mi, ml] = np.clip((preds[0]["y_v"] + preds[1]["y_v"]) / 2, 1, SPEED_CAP)
    fq[mi, ml] = np.maximum((preds[0]["y_q"] + preds[1]["y_q"]) / 2, 1)
    fk[mi, ml] = np.maximum((preds[0]["y_k"] + preds[1]["y_k"]) / 2, 1e-3)
    dfill = {"obs_speed_kmh": fv.reshape(nd, common.T, nl), "obs_flow_vph": (fq * lanes).reshape(nd, common.T, nl),
             "obs_density_occ_linear_vehpkm": d["obs_density_occ_linear_vehpkm"], "dates": d["dates"], "pct": d["pct"]}
    fb = F.Block(dfill, np.arange(nd), net, prof, version=2)
    X = blk.features(tt, li, net)
    X["rate"] = rate
    G = S2.filled_features(fb, tt, li, net, fv[tt, li], fq[tt, li], fk[tt, li], blk.v)
    X = pd.concat([X, G], axis=1)
    # s2 may be one model set or a list of seed variants, whose residuals are averaged
    sets = s2 if isinstance(s2, list) else [s2]
    cols = sets[0]["y_v"].feature_name()
    Xc = X[cols]
    res = {t: np.mean([m[t].predict(Xc, num_threads=common.THREADS) for m in sets], axis=0) for t in TARGETS}
    pv = X.p1_v.to_numpy() + res["y_v"]
    pq = X.p1_q.to_numpy() + res["y_q"]
    pk = X.p1_k.to_numpy() + res["y_k"]
    return X, pv, pq, pk


def finalize(p, tpl, li, pv, pq, pk, cfg):
    """Clip to the train range, reconcile density, write total flow into template rows."""
    tr = common.load(p, "train")
    lanes = common.network(p).lanes.to_numpy(np.float32)
    vmin = np.nanmin(tr["true_speed"], axis=(0, 1))
    qmax = np.nanmax(tr["true_flow"], axis=(0, 1)) / lanes
    pv = np.clip(pv, vmin[li], SPEED_CAP)
    rc = cfg["reconcile"]
    v, q = reconcile(pv, pq, pk, rc["alpha"], rc["beta"], rc.get("beta_c"), rc.get("vthr", 80.0))
    out = tpl.copy()
    out["speed_kmh"] = v
    out["flow_vph"] = np.clip(q, 1.0, 1.1 * qmax[li]) * lanes[li]
    assert np.isfinite(out.speed_kmh).all() and np.isfinite(out.flow_vph).all()
    return out


def template_cells(p, split):
    d = common.load(p, split)
    tpl = common.read_template(p, split)
    di, ti, li = common.template_index(tpl, d["dates"], d["links"])
    return tpl, di * common.T + ti, li, tpl.mask_regime.map(RATE).to_numpy(np.float32)


def predict_split_stage2(p: str, split: str, s1: dict, s2, cfg=CONFIG, gap=None, gap_w=0.5) -> pd.DataFrame:
    """Two-stage prediction: stage 1 (both day-parity models, averaged) fills every
    missing cell, stage 2 corrects the template cells using the filled field.
    gap: optional Task 2 gap specialist (one model set or a list averaged), blended with
    weight gap_w into cells inside the published window gaps."""
    tpl, tt, li, rate = template_cells(p, split)
    X, pv, pq, pk = stage2_raw(p, split, s1, s2, cfg, tt, li, rate)
    pv, pq, pk = apply_gap(X, pv, pq, pk, gap, gap_w, cells=(p, split, tt, li))
    return finalize(p, tpl, li, pv, pq, pk, cfg)


GAP_FEATURE_VERSION = 6


def split_ramps(p, split, d, tr):
    """Ramp flows of a split with their weekday x slot profile (train + the split)."""
    r = common.load_ramps(p, split, d["dates"], d["links"])
    parts = [(r["flow"], d["dates"])]
    if split != "train":
        rt = common.load_ramps(p, "train", tr["dates"], tr["links"])
        parts.insert(0, (rt["flow"], tr["dates"]))
    r["prof"] = F.ramp_profile(parts)
    return r


def gap_features(p, split, tt, li, wins=None, extra_blank=None, days=None):
    """Stage-1 features with the Task 2 window (v5) and ramp flow (v6) columns at cells
    (tt = day*T + slot, li). wins defaults to the split's published windows. days: build
    on these consecutive days only (tt, wins and extra_blank then relative to days[0])."""
    d = common.load(p, split)
    tr = common.load(p, "train")
    net = common.network(p)
    lanes = net.lanes.to_numpy(np.float32)
    prof = F.profiles([tr, d], lanes)
    r = split_ramps(p, split, d, tr)
    if wins is None:
        wins = [(di0 * common.T + s0, c) for di0, s0, c in common.split_windows(p, split, d["dates"])]
    days = np.arange(len(d["dates"])) if days is None else days
    r["flow"] = r["flow"][days]
    blk = F.Block(d, days, net, prof, ramps=r, extra_blank=extra_blank, version=GAP_FEATURE_VERSION, windows=wins)
    return blk.features(tt, li, net)


def apply_gap(X, pv, pq, pk, gap, gap_w, cells=None):
    """Blend the Task 2 gap specialist into cells inside the published window gaps.

    gap: one model set, or a list of sets whose predictions are averaged. A set may
    need columns X lacks (v5/v6 window and ramp features); those are built for the
    gap cells only, from cells = (panel, split, tt, li) of the rows of X."""
    if gap is None:
        return pv, pq, pk
    sets = gap if isinstance(gap, list) else [gap]
    g = X.win_cond.to_numpy() > 0
    if g.any():
        need = set().union(*(m["y_v"].feature_name() for m in sets)) - set(X.columns)
        if need:
            p, split, tt, li = cells
            Xs = gap_features(p, split, np.asarray(tt)[g], np.asarray(li)[g])
            Xs["rate"] = X.rate.to_numpy()[g]
            assert np.array_equal(Xs.win_cond.to_numpy(), X.win_cond.to_numpy()[g])
        else:
            Xs = X.loc[g]
        out = []
        for m in sets:
            Xg = Xs[m["y_v"].feature_name()]
            out.append((Xs.base_v.to_numpy() + m["y_v"].predict(Xg, num_threads=common.THREADS),
                        Xs.base_q.to_numpy() + m["y_q"].predict(Xg, num_threads=common.THREADS),
                        k_base(Xs) + m["y_k"].predict(Xg, num_threads=common.THREADS)))
        if len(out) == 1:
            gv, gq, gk = out[0]
        else:
            gv, gq, gk = (sum(o[i] for o in out) / len(out) for i in range(3))
        pv, pq, pk = pv.copy(), pq.copy(), pk.copy()
        pv[g] = gap_w * gv + (1 - gap_w) * pv[g]
        pq[g] = gap_w * gq + (1 - gap_w) * pq[g]
        pk[g] = gap_w * gk + (1 - gap_w) * pk[g]
    return pv, pq, pk


SS_EXTRA = 0.15
# SS_FOLDS > 1 hides SS_FOLDS disjoint SS_EXTRA shares of the visible cells one at a
# time, so R sees SS_FOLDS times more pseudo cells with the same missing-neighbour
# context. 1 reproduces the single-mask candidates exactly.
SS_FOLDS = int(os.environ.get("SS_FOLDS", 1))
SS_RCOLS = ["m_v", "m_q", "m_k", "tod", "wd", "pos", "lanes", "vf", "cap_l", "len", "li_id", "panel_id",
            "gp", "gn", "rate", "cov_t", "pv", "pq", "rv_all", "rq_s2.0", "rv_s2.0"]
# SS_RCOLS_EXTRA="g8_on_r,g9_est,...": more columns of the stage-2 frame for R (they must be
# computed there, e.g. the v7 ramp / conservation columns with gbm_v7r); empty = as before
SS_RCOLS = SS_RCOLS + [c for c in os.environ.get("SS_RCOLS_EXTRA", "").split(",") if c]


def selfsup_split(split: str, s1: dict, s2, cfg, panels, seed=20261026, gap=None, gap_w=0.5):
    """gbm_v8s2 + a correction learned from the split's own visible cells.

    For each panel, a further SS_EXTRA share of the visible eligible cells
    (outside the Task 2 gaps) is hidden; the two-stage model predicts them and a
    small model R learns the residual against their published observations.
    R is fitted on all panels of the split and applied to the template cells.
    Only published observations of the split are used as labels.
    """
    import lightgbm as lgb
    rng = np.random.default_rng(seed + (0 if split == "validation" else 1))
    pseudo, real = [], []
    for p in panels:
        d = common.load(p, split)
        lanes = common.network(p).lanes.to_numpy(np.float32)
        nd, nl = len(d["dates"]), len(lanes)
        vis = d["elig"] & np.isfinite(d["obs_speed_kmh"]) & np.isfinite(d["obs_flow_vph"]) & ~d["target"]
        u = rng.random(vis.shape)
        for f in range(SS_FOLDS):
            extra = vis & (u >= f * SS_EXTRA) & (u < (f + 1) * SS_EXTRA)
            di, ti, li = np.nonzero(extra)
            rate = np.vectorize(RATE.get)(d["regime"][di]).astype(np.float32)
            Xp, pv, pq, pk = stage2_raw(p, split, s1, s2, cfg, di * common.T + ti, li, rate, extra_blank=extra)
            pv, pq, pk = apply_gap(Xp, pv, pq, pk, gap, gap_w, cells=(p, split, di * common.T + ti, li))
            P = Xp[[c for c in SS_RCOLS if c in Xp.columns]].copy()
            P["m_v"], P["m_q"], P["m_k"] = pv, pq, pk
            P["li_id"], P["panel_id"] = li, common.PANELS.index(p)
            P["y_v"] = d["obs_speed_kmh"][di, ti, li]
            P["y_q"] = d["obs_flow_vph"][di, ti, li] / lanes[li]
            pseudo.append(P)
        tpl, tt, li2, rate2 = template_cells(p, split)
        Xr, rv, rq, rk = stage2_raw(p, split, s1, s2, cfg, tt, li2, rate2)
        rv, rq, rk = apply_gap(Xr, rv, rq, rk, gap, gap_w, cells=(p, split, tt, li2))
        R = Xr[[c for c in SS_RCOLS if c in Xr.columns]].copy()
        R["m_v"], R["m_q"], R["m_k"] = rv, rq, rk
        R["li_id"], R["panel_id"] = li2, common.PANELS.index(p)
        real.append((p, tpl, li2, R))
        print(f"  {split} {p}: pseudo {sum(len(x) for x in pseudo[-SS_FOLDS:]):,} real {len(R):,}", flush=True)
    P = pd.concat(pseudo, ignore_index=True)
    rprm = dict(objective="l2", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=common.THREADS, seed=seed,
                deterministic=True, force_row_wise=True, verbose=-1)
    ys = {"v": P.y_v - P.m_v, "q": P.y_q - P.m_q, "k": P.y_q / P.y_v.clip(lower=1) - P.m_k}
    models = {t: lgb.train(rprm, lgb.Dataset(P[SS_RCOLS], y.to_numpy(np.float32),
                                              categorical_feature=["li_id", "panel_id"]), 300) for t, y in ys.items()}
    if os.environ.get("SS_SAVE_R"):
        # saved for later swaps of gap-cell predictions without refitting R; output unchanged
        rdir = Path(os.environ["SS_SAVE_R"])
        rdir.mkdir(parents=True, exist_ok=True)
        for t, m in models.items():
            m.save_model(str(rdir / f"ssR_{split}_f{SS_FOLDS}_{t}.txt"))
    outs = []
    for p, tpl, li2, R in real:
        cv, cq, ck = (models[t].predict(R[SS_RCOLS], num_threads=common.THREADS) for t in ("v", "q", "k"))
        outs.append(finalize(p, tpl, li2, R.m_v.to_numpy() + cv, R.m_q.to_numpy() + cq, R.m_k.to_numpy() + ck, cfg))
    return outs
