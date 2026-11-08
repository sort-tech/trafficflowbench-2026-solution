"""Release-window probabilities of one run under other model/table choices (diagnosis of the LB gap):
  all_k3   full-train model, tables from all 9 train months (what release.pkl holds)
  all_kf   full-train model, average over the three 6-month tables (key A, B, C), the tables seen in training
  folds    average of the three fold models, fold model k with key k tables (the out-of-fold set-up)
Writes <run>/rel_variants.pkl {variant: {wid: p[6, L]}}.

Usage: rel_folds.py <run dir> [--device cuda]
"""
import argparse
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch

from train import FK, PANELS, Net, PanelData, predict

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--device", default="cuda")
ap.add_argument("--data", type=Path, default=HERE / "data")
a = ap.parse_args()
torch.set_num_threads(2)
dev = torch.device(a.device)
Ds, off = [], 0
for p in PANELS:
    Ds.append(PanelData(a.data / f"{p}.npz", dev, off)); off += Ds[-1].L


def load(k):

    net = Net(off, 32, 96, drop=0.1).to(dev)
    net.load_state_dict(torch.load(a.run / f"net_{k}.pt", map_location=dev))
    return net


nets = {k: load(k) for k in FK}
out = {"all_k3": {}, "all_kf": {}, "folds": {}}
with torch.no_grad():
    for D in Ds:
        for s, r in D.rel.items():
            arrs = {x: r[x] for x in ("speed", "flow", "occ", "pct")}
            run = lambda net, key: predict(net, D, arrs, r["t"], r["cond"], key, 0)
            p3 = run(nets["all"], 3)
            pf = np.mean([run(nets["all"], i) for i in range(3)], 0)
            pk = np.mean([run(nets[FK[i]], i) for i in range(3)], 0)
            for i, w in enumerate(r["wid"]):
                out["all_k3"][str(w)] = p3[i]; out["all_kf"][str(w)] = pf[i]; out["folds"][str(w)] = pk[i]
pickle.dump(out, open(a.run / "rel_variants.pkl", "wb"))
ref = pickle.load(open(a.run / "release.pkl", "rb"))
print("max |all_k3 - release.pkl|", max(float(np.abs(out["all_k3"][w] - ref[w]["p"]).max()) for w in ref))
