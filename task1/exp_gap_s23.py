"""Six ramp gap specialists instead of four (seed 23 added), train eval days.

Base: the v7-like paired stage 2 (s2r13, Mac mini check) with the gap models averaged and
blended at 0.75 inside the Task 2 gaps, then the usual clip + density reconcile + flow cap.
Arms: the v7 set (gap8, gap8_s11, gap8b, gap8b_s11), the six-model set, and the seed-23 pair
alone; also gap-cell speed / flow RMSE.

    python -I exp_gap_s23.py
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
V7 = ["gap8", "gap8_s11", "gap8b", "gap8b_s11"]
ARMS = {"x3 only": [], "v7 4 models": V7, "6 models": V7 + ["gap8_s23", "gap8b_s23"],
        "s23 pair only": ["gap8_s23", "gap8b_s23"]}


def main():
    rows = {a: [] for a in ARMS}
    err = {a: [0, 0.0, 0.0] for a in ARMS}
    for p in common.PANELS:
        ev = pd.read_parquet(HERE / "ds4" / f"eval_{p}.parquet", columns=["day", "t", "li", "win_cond", "regime", "y_v", "y_q", "lanes"])
        pr = pd.read_parquet(HERE / "preds" / "mini_s2pair" / f"s2r13_{p}.parquet")
        assert (pr.day.to_numpy() == ev.day.to_numpy()).all() and (pr.li.to_numpy() == ev.li.to_numpy()).all()
        g = ev.win_cond.to_numpy() > 0
        d = common.load(p, "train")
        pe = TG.PanelEval(p, ev)
        li = ev.li.to_numpy()
        vmin = np.nanmin(d["true_speed"], axis=(0, 1))
        qmax = np.nanmax(d["true_flow"], axis=(0, 1)) / pe.lanes
        fr = {t: pd.read_parquet(HERE / "preds" / f"{t}gaponly_{p}.parquet") for t in ARMS["6 models"]} if g.any() else {}
        for arm, tags in ARMS.items():
            a = {c: pr[c].to_numpy().astype(np.float64) for c in ("pv", "pq", "pk")}
            if g.any() and tags:
                for c in a:
                    a[c][g] = 0.75 * (sum(fr[t][c].to_numpy() for t in tags) / len(tags)) + 0.25 * a[c][g]
            v, q = reconcile(np.clip(a["pv"], vmin[li], SPEED_CAP), a["pq"], a["pk"], 1.0, 0.2, 0.8, 80.0)
            q = np.clip(q, 1.0, 1.1 * qmax[li])
            rows[arm].append(pe(v, q))
            if g.any():
                e = err[arm]
                e[0] += int(g.sum())
                e[1] += float(((v - ev.y_v.to_numpy())[g] ** 2).sum())
                e[2] += float(((q - ev.y_q.to_numpy())[g] ** 2).sum())
        print(p, flush=True)
    out = []
    for arm, r in rows.items():
        df = pd.concat(r)
        df["family"] = df.panel.map(common.FAMILY)
        df["contrib"] = 0.35 * df.S_state + 0.15 * df.S_physics
        t = df.groupby(["regime", "family"]).contrib.mean().groupby("regime").mean()
        n, sv, sq = err[arm]
        out.append({"arm": arm, **t.round(6).to_dict(), "all": float(t.mean()),
                    "gap v RMSE": np.sqrt(sv / n), "gap q/lane RMSE": np.sqrt(sq / n)})
    S = pd.DataFrame(out).set_index("arm")
    S["d_vs_v7"] = S["all"] - S.loc["v7 4 models", "all"]
    print(S.round(6).to_string())
    S.to_csv(HERE / "logs" / "exp_gap_s23.csv")


if __name__ == "__main__":
    main()
