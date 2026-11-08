"""Explain the private onset windows that lgb_i5_sclim75p changes relative to i5 (lgb_onthr_sq31_pao1535).

Per changed window: T, the T+30 links of both files, the onset location prior (features_v1 onset_prior =
Clim.onset[daytype, tod + 30 min]) at those links from the train climatology, the split climatology (validation +
private days before T's day) and the 0.75 / 0.25 blend, and March evidence: among validation onset pseudo windows
(organizer-style selector on the masked observations) of the same daytype within +-60 min of T's time of day, the
share whose observed T+30 queue includes the link.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import features_v1 as F1
from cmp_cand import links
from common import CACHE, H, PANELS, STEP, T0, Panel, select_windows
from diag_zero import spans
from split_clim import START, Blend, SplitPanel, onset_split_clim

A = "candidates/lgb_onthr_sq31_pao1535/queue_submission.csv"
B = "candidates/lgb_i5_sclim75p/queue_submission.csv"


def main() -> None:
    a = pd.read_csv(A); b = pd.read_csv(B)
    m = a.merge(b, on=["window_id", "timestamp", "link_id"])
    changed = sorted(m[m.queue_pred_x != m.queue_pred_y].window_id.unique())
    wi = pd.concat([pd.read_csv(Path(F1.REL) / "task2" / p / "private" / "window_index.csv") for p in PANELS])
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    rows = []
    for panel in PANELS:
        ws = [w for w in changed if f"_{panel}_" in w]
        if not ws:
            continue
        st = F1.Static(panel); P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        ot = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        ct = F1.Clim(P, qc, sorted(W.month.unique()), ot)
        del P, qc
        mar = SplitPanel(panel, st, {"validation": 31}); mq = mar.clean_queue()
        a0 = int((START["validation"] - T0) / STEP)
        ev = select_windows(mar, mq, start=a0 + 12, end=a0 + 288 * 31)
        ev = ev[ev.condition == "queue_onset"].t.to_numpy()
        for wid in ws:
            T = pd.Timestamp(wi.loc[wi.window_id == wid, "forecast_origin"].iloc[0])
            t = int((T - T0) / STEP)
            tod = t % 288; dt = int(F1.dow_of(t) >= 5)
            sc = onset_split_clim(panel, "private", st, t)
            bl = Blend(ct, sc, 0.75)
            la, lb = links(a, wid), links(b, wid)
            idx = sorted(set(_expand(la)) | set(_expand(lb)))
            near = [e for e in ev if int(F1.dow_of(e) >= 5) == dt and abs(e % 288 - tod) <= 12]
            share = {l: np.mean([mar.q[e + H, l] and mar.elig[e + H, l] for e in near]) if near else np.nan for l in idx}
            pr = lambda c, l: c.onset[dt, (tod + H) % 288, l]
            top = lambda c: spans(np.isin(np.arange(st.L), np.argsort(-c.onset[dt, (tod + H) % 288])[:5]))
            rows.append({"window": wid[3:], "T (UTC)": T.strftime("%m-%d %H:%M"), "daytype": "weekend" if dt else "weekday",
                         "i5 T+30": la, "sclim75p T+30": lb,
                         "prior train/split/blend": "; ".join(f"{l}: {pr(ct, l):.2f}/{pr(sc, l):.2f}/{pr(bl, l):.2f}" for l in idx),
                         "top-5 prior train": top(ct), "top-5 prior split": top(sc),
                         "split onsets": sc.n_onset, "March onsets near T": len(near),
                         "March share queued at T+30": "; ".join(f"{l}: {share[l]:.2f}" for l in idx)})
        print(panel, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "sclim_explain.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 200)
    print(df.to_string(index=False))


def _expand(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if part == "-":
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


if __name__ == "__main__":
    main()
