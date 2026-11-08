"""B18 evidence on validation itself: the production prediction path (build.predict_split_v1, i5 settings) run on
March pseudo windows with --split-clim-all lam, i.e. every climatology input blended with the complete March days
before the window's day (the rule the validation windows get).

Pseudo windows: organizer-style selector (no cap) on the masked March observations. Window inputs: masked rows
T-60..T. Truth: observed (not hidden) eligible cells of the horizon (onset: T+30 only). Release window ids are not
used; a temporary template folder holds the pseudo windows' rows.
"""
from __future__ import annotations

import shutil
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd

import build
import features_v1 as F1
from common import FAMILY, H, NHIST, PANELS, STEP, T0, select_windows
from split_clim import START, SplitPanel

LAMS = (1.0, 0.75, 0.5, 0.25)
if os.environ.get("TFB_LAMS"):
    LAMS = tuple(float(x) for x in os.environ["TFB_LAMS"].split(","))
TMP = Path("/tmp/sclim_tmpl")
I5 = dict(on_mix={"v1": .5, "v1_thr": .5}, go_mix={"v1": .15, "phys": .15, "aug": .35, "onphys": .35})


def main() -> None:
    a0 = int((START["validation"] - T0) / STEP)
    wins, truth = {}, {}
    if TMP.exists():
        shutil.rmtree(TMP)
    for panel in PANELS:
        st = F1.Static(panel)
        P = SplitPanel(panel, st, {"validation": 31})
        qc = P.clean_queue()
        sel = select_windows(P, qc, start=a0 + 12, end=a0 + 288 * 31)
        rows, tm = [], []
        for i, w in enumerate(sel.itertuples()):
            t = int(w.t)
            wid = f"PS_{panel}_{i:03d}"
            T = T0 + t * STEP
            sl = slice(t - NHIST, t + 1)
            arr = {"speed": P.speed[sl].copy(), "flow": P.flow[sl].copy(), "occ": P.occ[sl].copy(), "pct": P.pct[sl].copy()}
            y = P.q[t + 1:t + 1 + H] & P.elig[t + 1:t + 1 + H] & P.visible[t + 1:t + 1 + H]
            e = P.elig[t + 1:t + 1 + H] & P.visible[t + 1:t + 1 + H]
            if w.condition == "queue_onset":
                y = y.copy(); e = e.copy(); y[:H - 1] = False; e[:H - 1] = False
            if not (y & e).any():
                continue
            rows.append((pd.Series({"window_id": wid, "condition": w.condition, "forecast_origin": T.strftime("%Y-%m-%dT%H:%M:%SZ")}), arr, t))
            truth[wid] = (panel, w.condition, y, e)
            for h in range(1, H + 1):
                ts = (T + h * STEP).strftime("%Y-%m-%dT%H:%M:%SZ")
                tm += [(wid, ts, lid) for lid in st.link_ids]
        wins[panel] = rows
        d = TMP / "task2" / panel / "validation"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(tm, columns=["window_id", "timestamp", "link_id"]).to_csv(d / "sample_submission_queue.csv", index=False)
        print(panel, len(rows), flush=True)

    def fake_release(panel, split, st):
        for w, arr, t in wins[panel]:
            yield w, arr, t
    F1.release_windows = fake_release
    build.REL = TMP
    models = build.get_models_v1(False, False)
    res = {}
    for lam in LAMS:
        sub = build.predict_split_v1("validation", models, False, 0.37, go_mix=I5["go_mix"], on_mix=I5["on_mix"],
                                     small_thr=0.31, split_clim_all=None if lam == 1.0 else lam)
        res[lam] = sub
        print("lam", lam, "positives", int(sub.queue_pred.sum()), flush=True)
    rows = []
    pos = {p: {k: i for i, k in enumerate(F1.Static(p).link_ids)} for p in PANELS}
    for lam, sub in res.items():
        for wid, g in sub.groupby("window_id"):
            panel, cond, y, e = truth[wid]
            pr = np.zeros_like(y)
            g = g[g.queue_pred == 1]
            T = pd.Timestamp([w for w, _, _ in wins[panel] if w.window_id == wid][0].forecast_origin)
            hh = ((pd.to_datetime(g.timestamp, utc=True) - T) // STEP).to_numpy() - 1
            pr[hh, g.link_id.astype(str).map(pos[panel]).to_numpy()] = True
            p, t_ = pr & e, y & e
            u = (p | t_).sum()
            rows.append({"lam": lam, "window_id": wid, "panel": panel, "condition": cond, "iou": 1.0 if u == 0 else (p & t_).sum() / u})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "cache" / "sclim_all_eval.csv", index=False)
    piv = df.pivot_table(index=["window_id", "panel", "condition"], columns="lam", values="iou").reset_index()
    out = []
    for lam in LAMS:
        pc = piv.groupby(["panel", "condition"])[lam].mean().reset_index()
        pm = pc.groupby("panel")[lam].mean()
        fam = pm.groupby(pm.index.map(FAMILY)).mean()
        oc = pc.groupby("condition")[lam].mean()
        d = piv[lam] - piv[1.0]
        out.append({"lam": lam, "S_q": fam.mean(), "onset": oc["queue_onset"], "ongoing": oc["queue_ongoing"],
                    "win_diff": d.mean(), "se": d.std() / np.sqrt(len(d)), "better": int((d > 0).sum()), "worse": int((d < 0).sum())})
    pd.set_option("display.width", 200)
    print(f"March pseudo windows {len(piv)} (onset {int((piv.condition == 'queue_onset').sum())}, ongoing {int((piv.condition == 'queue_ongoing').sum())})")
    print(pd.DataFrame(out).round(4).to_string(index=False))
    fam = piv.assign(family=piv.panel.map(FAMILY)).groupby(["family", "condition"])[list(LAMS)].mean()
    print(fam.round(3).to_string())


if __name__ == "__main__":
    main()
