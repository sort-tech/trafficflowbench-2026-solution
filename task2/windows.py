"""Build pseudo windows on train with the imitated organizer selector.

greedy : the selector run from the start of train (evaluation set, same rules as official).
cand   : every origin that satisfies a condition (no spacing), ongoing thinned to every 3rd step,
         used as extra training windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from common import CACHE, H, NHIST, PANELS, Panel, T0, STEP, clean_queue, select_windows


def candidates(P: Panel, qc: np.ndarray, cap: float = 0.9, min_steps: int = 2, stride: int = 3) -> pd.DataFrame:
    qv = P.q & P.elig
    anyq = qv.any(1)
    cs = np.concatenate([[0], np.cumsum(anyq)])
    rows = []
    for t in range(NHIST, P.n_t - H - 1):
        hor = qc[t + 1:t + 1 + H]
        if not hor.any():
            continue
        nsteps = cs[t] - cs[t - NHIST]
        if nsteps == 0 and not anyq[t]:
            if not qc[t + 1:t + H].any():
                rows.append((t, "queue_onset"))
        elif nsteps >= min_steps and t % stride == 0:
            pers = qc[t]
            union = (hor | pers).sum()
            if (hor & pers).sum() / union <= cap:
                rows.append((t, "queue_ongoing"))
    df = pd.DataFrame(rows, columns=["t", "condition"])
    df["panel"] = P.panel
    return df


def main() -> None:
    out = []
    for p in PANELS:
        P = Panel(p)
        qc = clean_queue(P)
        np.save(CACHE / f"{p}_qclean.npy", qc)
        g = select_windows(P, qc)
        g["greedy"] = True
        c = candidates(P, qc)
        c = c.merge(g[["t", "condition"]].assign(greedy=True), on=["t", "condition"], how="left")
        c["greedy"] = c.greedy.fillna(False).astype(bool)
        # greedy windows that the thinned ongoing scan skipped
        miss = g[~g.t.isin(c.t)]
        c = pd.concat([c, miss[["t", "condition", "panel", "greedy"]]], ignore_index=True)
        out.append(c)
        print(p, len(g), len(c), flush=True)
    df = pd.concat(out, ignore_index=True)
    df["month"] = pd.to_datetime(T0 + df.t.to_numpy() * STEP).strftime("%Y-%m")
    df = df.sort_values(["panel", "t"]).reset_index(drop=True)
    df.to_parquet(CACHE / "pseudo_windows.parquet", index=False)
    print(df.groupby(["greedy", "condition"]).size().unstack())


if __name__ == "__main__":
    main()
