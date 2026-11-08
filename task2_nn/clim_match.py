"""How well the train time-of-day climatology matches observed queues, per split (out of fold / validation / private).

Cells: every observed cell (speed not NaN) in the 13 history rows T-60 .. T of every ongoing window
  oof         the 719-window out-of-fold set (prep.py ev_*: official-style monthly selection, 5 per condition and
              month, T-row masks as task2/monthly_v1.py)
  validation  release windows (prep.py rel_validation_*, the released arrays)
  private     release windows (rel_private_*)
Queue: speed <= v_cut of the link. Climatology: prep.py clim_<k>_q = features_v1.Clim(...).q, the share of time the
clean queue state (task2 common.clean_queue: observed speed, gaps forward/back filled, <= v_cut) is on, by daytype
(weekday / weekend), 5-min time of day (smoothed +-15 min) and link, over train months. Out of fold uses the table
without the window's own 3-month fold (6 months), release uses all 9 train months. The table is read at each history
row's own time of day and daytype.
Prints P(observed queue | clim >= 0.5) and P(observed queue | clim < 0.1), overall and per panel.

Usage: clim_match.py            (needs task2_nn/data/<panel>.npz from prep.py)
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FK = ["A", "B", "C", "all"]
T0_DOW = 5                     # 2030-06-01 (t = 0) is a Saturday; weekend = day of week >= 5 (Mon = 0)
HI, LO = 0.5, 0.1


def main() -> None:
    rows = []
    for f in sorted(glob.glob(str(HERE / "data" / "D*.npz"))):
        panel = Path(f).stem
        z = np.load(f)
        vcut = z["vcut"]; clim = {k: z[f"clim_{k}_q"] for k in FK}
        for split, pre in (("oof", "ev"), ("validation", "rel_validation"), ("private", "rel_private")):
            T, SP, CO = z[f"{pre}_t"], z[f"{pre}_speed"], z[f"{pre}_cond"]
            KEY = z["ev_fold"] if split == "oof" else np.full(len(T), 3)
            a = np.zeros(4)
            for t, sp, key, cond in zip(T, SP, KEY, CO):
                if cond != 0:          # ongoing windows only
                    continue
                for r in range(13):
                    tt = int(t) - 12 + r
                    cl = clim[FK[int(key)]][int((tt // 288 + T0_DOW) % 7 >= 5), tt % 288]
                    ob = ~np.isnan(sp[r]); q = sp[r] <= vcut
                    hi = (cl >= HI) & ob; lo = (cl < LO) & ob
                    a += [hi.sum(), (hi & q).sum(), lo.sum(), (lo & q).sum()]
            rows.append({"panel": panel, "split": split, "n_hi": a[0], "q_hi": a[1], "n_lo": a[2], "q_lo": a[3]})
    d = pd.DataFrame(rows)
    show = lambda g: pd.Series({"cells clim>=0.5": int(g.n_hi.sum()), "P(queue|clim>=0.5)": g.q_hi.sum() / g.n_hi.sum(),
                                "cells clim<0.1": int(g.n_lo.sum()), "P(queue|clim<0.1)": g.q_lo.sum() / g.n_lo.sum()})
    pd.set_option("display.width", 200)
    print(d.groupby("split").apply(show).round(4).to_string())
    print(d.groupby(["panel", "split"]).apply(show)["P(queue|clim>=0.5)"].unstack("split").round(3).to_string())


if __name__ == "__main__":
    main()
