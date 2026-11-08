"""Second batch of decision checks on lgb_v1 out-of-fold probabilities (cache/pp_windows.pkl, no refit).

2A-1  ongoing threshold by horizon: 0.40 at T+5/T+10, higher at T+15..T+30.
2A-2  base-rate logit shift: logit(p') = logit(p) + a * log(pi_split / pi_train). pi_split is the panel's
      eligible-cell queue rate in its own split before T (train: the window's own month from day 1 to T),
      shrunk toward pi_train with one day of pseudo-counts; pi_train is the rate over the months the fold
      model was trained on.
2B    low-confidence onset windows (bottom 20% of max p): where is the T+30 truth relative to the corridor's
      habitual onset bottlenecks (from the other folds' onset windows), and a "also switch on the second
      probability peak" rule.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import features_v1 as F1
from common import CACHE, H, PANELS, STEP, T0, Panel, aggregate
from model import best_k_set, iou
from pp_eval import collect, onset_pred, v1_pred

EXTRA = CACHE / "pp_extra.pkl"


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != p + 1:
            out.append((s, p)); s = i
        p = i
    out.append((s, p))
    return out


def extras(recs: list[dict]) -> dict:
    if EXTRA.exists():
        return pickle.load(open(EXTRA, "rb"))
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    out = {"pi": {}, "habit": {}, "month_rate": {}}
    for panel in PANELS:
        P = Panel(panel)
        qc = np.load(CACHE / f"{panel}_qclean.npy")
        mon = pd.DatetimeIndex(T0 + np.arange(P.n_t) * STEP).strftime("%Y-%m").to_numpy()
        vis = P.visible & P.elig
        q = P.q & vis
        cv = np.concatenate([[0], np.cumsum(vis.sum(1))])
        cq = np.concatenate([[0], np.cumsum(q.sum(1))])
        rate_m = {m: q[mon == m].sum() / max(vis[mon == m].sum(), 1) for m in np.unique(mon)}
        out["month_rate"].update({(panel, m): float(r) for m, r in rate_m.items()})
        pi_tr = {k: float(sum(q[mon == m].sum() for f, ms in F1.FOLDS.items() if f != k for m in ms)
                          / max(sum(vis[mon == m].sum() for f, ms in F1.FOLDS.items() if f != k for m in ms), 1))
                 for k in F1.FOLDS}
        day_cells = 288 * P.n_l * 0.5
        ot = W[(W.panel == panel) & (W.condition == "queue_onset")]
        for k in F1.FOLDS:
            o = ot[ot.month.map(m2f) != k].t.to_numpy()
            out["habit"][(panel, k)] = qc[o + H].mean(0) if len(o) else np.zeros(P.n_l)
        for r in recs:
            if r["panel"] != panel:
                continue
            t = r["t"]
            ms = int((pd.Timestamp(f"{r['month']}-01T00:00:00Z") - T0) / STEP)
            n, nq = cv[t] - cv[ms], cq[t] - cq[ms]              # rows ms .. t-1 only
            k = m2f[r["month"]]
            out["pi"][(panel, t)] = ((nq + day_cells * pi_tr[k]) / (n + day_cells), pi_tr[k])
        print(panel, flush=True)
    pickle.dump(out, open(EXTRA, "wb"))
    return out


def logit_shift(p: np.ndarray, d: float) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) + d
    return 1 / (1 + np.exp(-z))


def score(recs, meta, base, preds_fn, cond):
    sc = base.copy()
    for i, r in enumerate(recs):
        if cond in ("both", r["condition"]):
            sc[i] = iou(preds_fn(r), r["y"], r["e"])
    return sc


def summarize(name, cond, sc, base, meta, sub=None):
    S0 = aggregate(meta.assign(iou=base), "iou")["S_queue"]
    S = aggregate(meta.assign(iou=sc), "iou")["S_queue"]
    m = (meta.condition == cond) if cond != "both" else np.ones(len(meta), bool)
    d = (sc - base)[m]
    mo = [aggregate(meta[meta.month == x].assign(iou=sc[meta.month == x]), "iou")["S_queue"]
          - aggregate(meta[meta.month == x].assign(iou=base[meta.month == x]), "iou")["S_queue"] for x in sorted(meta.month.unique())]
    row = {"rule": name, "cond": cond, "dS": S - S0, "win_mean": d.mean(), "win_se": d.std() / np.sqrt(len(d)),
           "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
           "months_up": int(sum(x > 1e-9 for x in mo)), "months_down": int(sum(x < -1e-9 for x in mo))}
    if sub is not None:
        ds = (sc - base)[m & sub]
        row.update({"low_n": int((m & sub).sum()), "low_mean": ds.mean() if len(ds) else np.nan})
    return row


def main() -> None:
    recs = collect()
    ex = extras(recs)
    meta = pd.DataFrame([{k: r[k] for k in ("panel", "month", "condition", "t")} for r in recs])
    m2f = {m: f for f, ms in F1.FOLDS.items() for m in ms}
    base = np.array([iou(v1_pred(r), r["y"], r["e"]) for r in recs])
    # low-demand subset (reported only): the 3 lowest-rate months of each panel, ongoing windows with <= 25 queued links at T
    lowm = {(p, m) for p in PANELS for m in sorted([m for (pp, m) in ex["month_rate"] if pp == p and m in m2f],
                                                     key=lambda m: ex["month_rate"][(p, m)])[:3]}
    sub = np.array([(r["panel"], r["month"]) in lowm and (r["condition"] == "queue_onset" or r["q_t5"].sum() <= 25)
                    for r in recs])
    rows = []
    # 2A-1 horizon thresholds
    for thr in (0.42, 0.45, 0.48, 0.50, 0.52):
        th = np.array([0.40, 0.40, thr, thr, thr, thr])[:, None]
        rows.append(summarize(f"2A1_h3-6_{thr}", "queue_ongoing",
                              score(recs, meta, base, lambda r, th=th: r["p"] >= th, "queue_ongoing"), base, meta, sub))
    for thr in (0.45, 0.50):
        th = np.array([0.40, 0.40, 0.40, 0.40, thr, thr])[:, None]
        rows.append(summarize(f"2A1_h5-6_{thr}", "queue_ongoing",
                              score(recs, meta, base, lambda r, th=th: r["p"] >= th, "queue_ongoing"), base, meta, sub))
    # 2A-2 base-rate logit shift
    def shift(r, a):
        pi_s, pi_t = ex["pi"][(r["panel"], r["t"])]
        return a * np.log(max(pi_s, 1e-5) / max(pi_t, 1e-5))
    for a in (0.3, 0.5, 0.7, 1.0):
        on = lambda r, a=a: onset_pred(best_k_set(logit_shift(r["p"], shift(r, a))), len(r["p"]))
        go = lambda r, a=a: logit_shift(r["p"], shift(r, a)) >= 0.4
        rows.append(summarize(f"2A2_onset_a{a}", "queue_onset", score(recs, meta, base, on, "queue_onset"), base, meta, sub))
        rows.append(summarize(f"2A2_ongoing_a{a}", "queue_ongoing", score(recs, meta, base, go, "queue_ongoing"), base, meta, sub))
        both = lambda r, on=on, go=go: on(r) if r["condition"] == "queue_onset" else go(r)
        rows.append(summarize(f"2A2_both_a{a}", "both", score(recs, meta, base, both, "both"), base, meta, sub))
    sh = np.array([shift(r, 1.0) for r in recs])
    print("log(pi_split/pi_train): mean %.3f sd %.3f, 5-95%% %.3f..%.3f" % (sh.mean(), sh.std(), *np.percentile(sh, [5, 95])))
    # 2B low-confidence onset windows
    on_idx = [i for i, r in enumerate(recs) if r["condition"] == "queue_onset"]
    pmax = np.array([recs[i]["p"].max() for i in on_idx])
    cut = np.quantile(pmax, 0.2)
    cats = []
    for i, pm in zip(on_idx, pmax):
        if pm > cut:
            continue
        r = recs[i]
        hab = ex["habit"][(r["panel"], m2f[r["month"]])]
        hr = sorted(runs(hab >= 0.05), key=lambda s: -hab[s[0]:s[1] + 1].max())
        truth = (r["y"] & r["e"])[H - 1]
        pred = best_k_set(r["p"])
        rank = None
        for j, (a, b) in enumerate(hr):
            if truth[max(a - 1, 0):b + 2].any():
                rank = j + 1
                break
        cats.append({"panel": r["panel"], "pmax": pm, "iou": base[i], "hit_pred": bool((truth & pred).any()),
                     "truth_rank": rank if rank is not None else 0, "n_habit": len(hr)})
    c = pd.DataFrame(cats)
    c["where"] = np.select([c.hit_pred, c.truth_rank == 1, c.truth_rank == 2, c.truth_rank >= 3],
                           ["overlaps prediction", "habit #1", "habit #2", "habit #3+"], "not habitual")
    print(f"2B: bottom 20% max p (<= {cut:.3f}): {len(c)} windows, mean IoU {c.iou.mean():.3f}")
    print(c.groupby("where").agg(n=("iou", "size"), iou=("iou", "mean")).round(3))
    # second-peak rule: also switch on the second probability run (links with p >= half of that run's max)
    def second_peak(p, conf, rel=0.5, tau=0.03):
        pred = best_k_set(p)
        if p.max() >= conf:
            return pred
        rr = [(a, b) for a, b in runs(p >= tau) if not pred[a:b + 1].any()]
        if not rr:
            return pred
        a, b = max(rr, key=lambda s: p[s[0]:s[1] + 1].max())
        seg = p[a:b + 1]
        pred[a:b + 1] |= seg >= rel * seg.max()
        return pred
    for conf in (cut, 0.6, 0.8):
        for tau in (0.01, 0.03):
            f = lambda r, conf=conf, tau=tau: onset_pred(second_peak(r["p"], conf, tau=tau), len(r["p"]))
            rows.append(summarize(f"2B_peak2_conf{conf:.2f}_tau{tau}", "queue_onset",
                                  score(recs, meta, base, f, "queue_onset"), base, meta, sub))
    def habit2(r, conf):
        pred = best_k_set(r["p"])
        if r["p"].max() >= conf:
            return onset_pred(pred, len(pred))
        hab = ex["habit"][(r["panel"], m2f[r["month"]])]
        hr = [s for s in sorted(runs(hab >= 0.05), key=lambda s: -hab[s[0]:s[1] + 1].max()) if not pred[s[0]:s[1] + 1].any()]
        if hr:
            a, b = hr[0]
            seg = hab[a:b + 1]
            pred[a:b + 1] |= seg >= 0.5 * seg.max()
        return onset_pred(pred, len(pred))
    for conf in (cut, 0.6):
        rows.append(summarize(f"2B_habit_next_conf{conf:.2f}", "queue_onset",
                              score(recs, meta, base, lambda r, conf=conf: habit2(r, conf), "queue_onset"), base, meta, sub))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"v1 S_q {aggregate(meta.assign(iou=base), 'iou')['S_queue']:.4f}; low-demand subset {int(sub.sum())} windows")
    print(res.round(4).to_string(index=False))
    res.to_csv(CACHE / "pp_eval2.csv", index=False)
    c.to_csv(CACHE / "pp_eval2_lowconf_onset.csv", index=False)


if __name__ == "__main__":
    main()
