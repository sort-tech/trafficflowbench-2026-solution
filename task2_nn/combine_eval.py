"""Out-of-fold check of network + LightGBM probability mixes on the 719 official-style train windows.

LightGBM probabilities are the cached task2 out-of-fold ones (task2/mix_eval5.load). Bases:
  h5  onset v1 .5 + v1_thr .5 (top-k), ongoing v1 .2 / phys .2 / aug .6, threshold 0.37, small-queue windows 0.31
  i5  h5 with ongoing v1 .15 / phys .15 / aug .35 / onphys .35
Mix: p = (1 - w) p_lgb + w p_nn, then the same decision rule. dS0 = no-shift dS_q vs the base; shift = mean over
+-45 / +-60 min (both onset and ongoing time-of-day inputs moved) of dS_q vs the base at the same shift.

Usage: combine_eval.py <run dir with oof_A.pkl oof_B.pkl oof_C.pkl> [more run dirs, probabilities averaged]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task2"))
import numpy as np
import pandas as pd

from common import H, aggregate
from mix_eval import combine
from mix_eval5 import H5_STEPS, load, thr_for
from model import best_k_set, iou

SH = (9, -9, 12, -12)
BASES = {"h5": {"v1": .2, "phys": .2, "aug": .6}, "i5": {"v1": .15, "phys": .15, "aug": .35, "onphys": .35}}


def load_nn(dirs: list[Path]) -> dict:
    runs = []
    for d in dirs:
        r = {}
        for k in "ABC":
            r.update(pickle.load(open(d / f"oof_{k}.pkl", "rb")))
        runs.append(r)
    return {key: {f: np.mean([r[key][f] for r in runs], 0) for f in runs[0][key]} for key in runs[0]}


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]]
    G, kt = load()
    NN = load_nn(dirs)
    meta = G.meta

    def on_scores(w, sh):
        out = []
        for k in G.ko:
            r = G.on[k]
            p = (1 - w) * (0.5 * r[f"v1_{sh}"] + 0.5 * r[f"v1_thr_{sh}"]) + w * NN[k][f"nn_{sh}"]
            pr = np.zeros(r["y"].shape, bool); pr[H - 1] = best_k_set(p)
            out.append(iou(pr, r["y"], r["e"]))
        return np.array(out)

    def go_scores(bname, w, sh, thr_shift=0.0):
        base = BASES[bname]
        out = []
        for i, k in enumerate(G.kg):
            r = G.go[k]
            p = (1 - w) * combine({v: r[f"{v}_{sh}"] for v in base}, base) + w * NN[k][f"nn_{sh}"]
            out.append(iou(p >= thr_for(int(kt[i]), H5_STEPS) + thr_shift, r["y"], r["e"]))
        return np.array(out)

    S = lambda a, b: aggregate(meta.assign(iou=np.concatenate([a, b])), "iou")
    rows = []
    cache = {}

    def get(kind, *a):
        if (kind, a) not in cache:
            cache[(kind, a)] = on_scores(*a) if kind == "on" else go_scores(*a)
        return cache[(kind, a)]

    for bname, base in BASES.items():
        ref = {s: S(get("on", 0.0, s), get("go", bname, 0.0, s)) for s in (0,) + SH}
        for won in (0.0, 0.2, 0.3, 0.5):
            for wgo in (0.0, 0.3, 0.5, 0.7):
                if won == wgo == 0.0:
                    continue
                for ts in ((0.0,) if wgo != 0.5 else (0.0, 0.03)):
                    a = {s: S(get("on", won, s), go_scores(bname, wgo, s, ts) if ts else get("go", bname, wgo, s)) for s in (0,) + SH}
                    d = np.concatenate([get("on", won, 0), go_scores(bname, wgo, 0, ts) if ts else get("go", bname, wgo, 0)]) \
                        - np.concatenate([get("on", 0.0, 0), get("go", bname, 0.0, 0)])
                    rows.append({"base": bname, "w_onset": won, "w_ongoing": wgo, "thr_shift": ts,
                                 "S_q": a[0]["S_queue"], "onset": a[0]["onset"], "ongoing": a[0]["ongoing"],
                                 "dS0": a[0]["S_queue"] - ref[0]["S_queue"], "se_win": d.std() / np.sqrt(len(d)),
                                 "better": int((d > 1e-9).sum()), "worse": int((d < -1e-9).sum()),
                                 "shift": np.mean([a[s]["S_queue"] - ref[s]["S_queue"] for s in SH])})
        print(bname, "base S_q", round(ref[0]["S_queue"], 4), "onset", round(ref[0]["onset"], 4), "ongoing", round(ref[0]["ongoing"], 4),
              "| shifted", {s * 5: round(ref[s]["S_queue"], 4) for s in SH})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
    print(res.round(4).to_string(index=False))
    tag = "+".join(d.name for d in dirs)
    res.to_csv(HERE / "logs" / f"combine_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
