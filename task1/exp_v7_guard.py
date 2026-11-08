"""Does the v7-like prediction lose where the review guide expects the conservation estimate
to blow up? Train eval days, paired arms from the Mac mini check:
    old: s2b13 (FEAT=4 stage 2) + gbm9gap 0.5 in the gaps        (ss3f-like)
    new: s2r13 (FEAT=7 stage 2) + ramp gap models 0.75 in the gaps (v7-like)
Cells grouped by: u..d span > 10 links, g9_est / base_q outside +-50%, and the size of the
total-flow change new - old (> 2000 vph, as in the 401 test cells). For each group: count,
RMSE of both, and the proxy if those cells take the old values (or the mean of both).
Features from ds14 (= ds4 eval rows, FEAT=7).

    python -I exp_v7_guard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import common
import train_gbm as TG
from pipeline import reconcile, SPEED_CAP

HERE = Path(__file__).resolve().parent
TAGS = ["gap8", "gap8_s11", "gap8b", "gap8b_s11"]


def final(a, vmin, qmax, li):
    v, q = reconcile(np.clip(a["pv"], vmin[li], SPEED_CAP), a["pq"], a["pk"], 1.0, 0.2, 0.8, 80.0)
    return v, np.clip(q, 1.0, 1.1 * qmax[li])


def main():
    conds = ["span>10", "est/base out 50%", "|dQ|>2000", "|dQ|>2000 gap", "|dQ|>2000 non-gap", "|dQ|>1000"]
    arms = ["new"] + [f"{c} -> old" for c in conds] + [f"{c} -> mean" for c in conds]
    rows = {a: [] for a in arms}
    rep = {c: [] for c in conds}
    for p in common.PANELS:
        ev = pd.read_parquet(HERE / "ds4" / f"eval_{p}.parquet", columns=["day", "t", "li", "win_cond", "regime", "y_v", "y_q", "lanes"])
        X = pd.read_parquet(HERE / "ds14" / f"eval_{p}.parquet", columns=["day", "t", "li", "g9_lu", "g9_ld", "g9_est", "base_q"])
        assert (X.day.to_numpy() == ev.day.to_numpy()).all() and (X.li.to_numpy() == ev.li.to_numpy()).all()
        g = ev.win_cond.to_numpy() > 0
        P = {}
        for arm, tag in (("old", "s2b13"), ("new", "s2r13")):
            pr = pd.read_parquet(HERE / "preds" / "mini_s2pair" / f"{tag}_{p}.parquet")
            P[arm] = {c: pr[c].to_numpy().astype(np.float64) for c in ("pv", "pq", "pk")}
        if g.any():
            old = pd.read_parquet(HERE / "preds" / f"gbm9gap_{p}.parquet")
            fr = [pd.read_parquet(HERE / "preds" / f"{t}gaponly_{p}.parquet") for t in TAGS]
            for c in ("pv", "pq", "pk"):
                P["old"][c][g] = 0.5 * old[c].to_numpy()[g] + 0.5 * P["old"][c][g]
                P["new"][c][g] = 0.75 * (sum(f[c].to_numpy() for f in fr) / 4) + 0.25 * P["new"][c][g]
        d = common.load(p, "train")
        pe = TG.PanelEval(p, ev)
        li = ev.li.to_numpy()
        vmin = np.nanmin(d["true_speed"], axis=(0, 1))
        qmax = np.nanmax(d["true_flow"], axis=(0, 1)) / pe.lanes
        vo, qo = final(P["old"], vmin, qmax, li)
        vn, qn = final(P["new"], vmin, qmax, li)
        lanes = ev.lanes.to_numpy()
        dQ = np.abs(qn - qo) * lanes
        span = (X.g9_lu + X.g9_ld).to_numpy()
        ratio = (X.g9_est / X.base_q.clip(lower=1)).to_numpy()
        M = {"span>10": span > 10, "est/base out 50%": (ratio < 0.5) | (ratio > 1.5), "|dQ|>2000": dQ > 2000,
             "|dQ|>2000 gap": (dQ > 2000) & g, "|dQ|>2000 non-gap": (dQ > 2000) & ~g, "|dQ|>1000": dQ > 1000}
        yv, yq = ev.y_v.to_numpy(), ev.y_q.to_numpy()
        for c, m in M.items():
            rep[c].append((int(m.sum()), ((vo - yv)[m] ** 2).sum(), ((vn - yv)[m] ** 2).sum(),
                           (((qo - yq) * lanes)[m] ** 2).sum(), (((qn - yq) * lanes)[m] ** 2).sum()))
        rows["new"].append(pe(vn, qn))
        for c, m in M.items():
            for how in ("old", "mean"):
                v, q = vn.copy(), qn.copy()
                v[m] = vo[m] if how == "old" else 0.5 * (vo[m] + vn[m])
                q[m] = qo[m] if how == "old" else 0.5 * (qo[m] + qn[m])
                rows[f"{c} -> {how}"].append(pe(v, q))
        print(p, flush=True)
    print("cells | RMSE speed old / new | RMSE total flow old / new")
    for c in conds:
        n, so, sn, qo_, qn_ = (sum(x[i] for x in rep[c]) for i in range(5))
        print(f"  {c}: n={n}", (f"v {np.sqrt(so / n):.2f} / {np.sqrt(sn / n):.2f}  Q {np.sqrt(qo_ / n):.0f} / {np.sqrt(qn_ / n):.0f}"
                                if n else ""))
    out = []
    for arm, r in rows.items():
        df = pd.concat(r)
        df["family"] = df.panel.map(common.FAMILY)
        df["contrib"] = 0.35 * df.S_state + 0.15 * df.S_physics
        t = df.groupby(["regime", "family"]).contrib.mean().groupby("regime").mean()
        out.append({"arm": arm, "all": float(t.mean())})
    S = pd.DataFrame(out).set_index("arm")
    S["d_vs_new"] = S["all"] - S.loc["new", "all"]
    print(S.round(6).to_string())
    S.to_csv(HERE / "logs" / "exp_v7_guard.csv")


if __name__ == "__main__":
    main()
