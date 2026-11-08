"""Agreement of two release candidates under the queue metric: the reference file is scored as if it were the
labels (window IoU, empty-empty = 1, then window -> (panel, condition) -> panel -> family). Also the mean absolute
difference of the averaged network probabilities.

Usage: agree.py <release.pkl list A, comma separated> <list B> <reference csv> <candidate csv> [<ref> <cand> ...]
"""
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/../task2")
from common import FAMILY


def load(paths):
    rs = [pickle.load(open(p, "rb")) for p in paths.split(",")]
    return rs[0], {w: np.mean([r[w]["p"] for r in rs], axis=0) for w in rs[0]}


meta, pa = load(sys.argv[1])
_, pb = load(sys.argv[2])
d = np.concatenate([np.abs(pa[w] - pb[w]).ravel() for w in pa])
print(f"network probability |A-B|: mean {d.mean():.4f}, p99 {np.quantile(d, .99):.4f}, max {d.max():.4f}")
for ref, cand in zip(sys.argv[3::2], sys.argv[4::2]):
    m = pd.read_csv(ref).merge(pd.read_csv(cand), on=["window_id", "timestamp", "link_id"])
    m["i"] = m.queue_pred_x & m.queue_pred_y
    m["u"] = m.queue_pred_x | m.queue_pred_y
    g = m.groupby("window_id")[["i", "u"]].sum()
    iou = np.where(g.u > 0, g.i / g.u.clip(lower=1), 1.0)
    rows = pd.DataFrame({"panel": [meta[w]["panel"] for w in g.index], "cond": [meta[w]["cond"] for w in g.index],
                         "split": [meta[w]["split"] for w in g.index], "iou": iou})
    for split, r in [("all", rows)] + list(rows.groupby("split")):
        pc = r.groupby(["panel", "cond"]).iou.mean().reset_index()
        pm = pc.groupby("panel").iou.mean()
        s = pm.groupby(pm.index.map(FAMILY)).mean().mean()
        print(f"{cand} vs {ref} [{split}]: metric {s:.4f} (1 - metric {1 - s:.2%}), "
              f"windows changed {(r.iou < 1).sum()}/{len(r)}, min window IoU {r.iou.min():.3f}")
