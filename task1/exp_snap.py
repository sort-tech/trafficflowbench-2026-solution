"""Soft snap of congested speeds to the link's typical congested speed (review guide 10/31).

Per link, from the train fit days only (eval days index % 9 == 4 left out): median and std
of the true speed where it is congested (v <= 0.6 free speed). On the train eval days,
after the usual clip + density reconcile + flow cap, cells with predicted v < 0.55 free
speed on links whose congested std <= STD_MAX move v <- (1 - a) v + a v_cong; flow stays,
so the implied density changes. Base prediction: the v7-like paired stage 2 (s2r13) with
the four ramp gap models at 0.75 in the gaps.

    python -I exp_snap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import common
import train_gbm as TG
from make_dataset import eval_days
from pipeline import reconcile, SPEED_CAP

HERE = Path(__file__).resolve().parent
TAGS = ["gap8", "gap8_s11", "gap8b", "gap8b_s11"]
ALPHAS = (0.15, 0.25, 0.35)
STD_MAX = 3.0


def cong_stats(d, vf):
    v = d["true_speed"]
    fit = ~eval_days(len(d["dates"]))
    x = v[fit].reshape(-1, v.shape[2])
    c = x <= 0.6 * vf[None, :]
    med = np.array([np.median(x[c[:, l], l]) if c[:, l].sum() >= 20 else np.nan for l in range(x.shape[1])])
    sd = np.array([np.std(x[c[:, l], l]) if c[:, l].sum() >= 20 else np.nan for l in range(x.shape[1])])
    return med, sd


def main():
    arms = ["base"] + [f"a{a}" for a in ALPHAS]
    rows = {a: [] for a in arms}
    stats = []
    for p in common.PANELS:
        ev = pd.read_parquet(HERE / "ds4" / f"eval_{p}.parquet", columns=["day", "t", "li", "win_cond", "regime", "y_v", "y_q"])
        pr = pd.read_parquet(HERE / "preds" / "mini_s2pair" / f"s2r13_{p}.parquet")
        assert (pr.day.to_numpy() == ev.day.to_numpy()).all()
        g = ev.win_cond.to_numpy() > 0
        a = {c: pr[c].to_numpy().astype(np.float64) for c in ("pv", "pq", "pk")}
        if g.any():
            fr = [pd.read_parquet(HERE / "preds" / f"{t}gaponly_{p}.parquet") for t in TAGS]
            for c in a:
                a[c][g] = 0.75 * (sum(f[c].to_numpy() for f in fr) / 4) + 0.25 * a[c][g]
        d = common.load(p, "train")
        net = common.network(p)
        vf = net.free_speed_kmh.to_numpy(np.float64)
        med, sd = cong_stats(d, vf)
        pe = TG.PanelEval(p, ev)
        li = ev.li.to_numpy()
        vmin = np.nanmin(d["true_speed"], axis=(0, 1))
        qmax = np.nanmax(d["true_flow"], axis=(0, 1)) / pe.lanes
        v, q = reconcile(np.clip(a["pv"], vmin[li], SPEED_CAP), a["pq"], a["pk"], 1.0, 0.2, 0.8, 80.0)
        q = np.clip(q, 1.0, 1.1 * qmax[li])
        m = (v < 0.55 * vf[li]) & (sd[li] <= STD_MAX) & np.isfinite(med[li])
        yv = ev.y_v.to_numpy()
        stats.append((p, int(m.sum()), int(np.isfinite(sd).sum()), int((sd <= STD_MAX).sum()), len(sd),
                      float(np.sqrt(np.mean((v[m] - yv[m]) ** 2))) if m.any() else np.nan))
        for arm in arms:
            vv = v.copy()
            if arm != "base":
                al = float(arm[1:])
                vv[m] = (1 - al) * v[m] + al * med[li][m]
            r = pe(vv, q)
            r["arm"] = arm
            rows[arm].append(r)
            if arm != "base" and m.any():
                stats.append((p, arm, "", "", "", float(np.sqrt(np.mean((vv[m] - yv[m]) ** 2)))))
        print(p, flush=True)
    for s in stats:
        print(s)
    out = []
    for arm, r in rows.items():
        df = pd.concat(r)
        df["family"] = df.panel.map(common.FAMILY)
        df["contrib"] = 0.35 * df.S_state + 0.15 * df.S_physics
        t = df.groupby(["regime", "family"]).contrib.mean().groupby("regime").mean()
        out.append({"arm": arm, **t.round(5).to_dict(), "all": round(float(t.mean()), 5)})
    S = pd.DataFrame(out).set_index("arm")
    S["d"] = (S["all"] - S.loc["base", "all"]).round(5)
    print(S.to_string())
    S.to_csv(HERE / "logs" / "exp_snap.csv")


if __name__ == "__main__":
    main()
