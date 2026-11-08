"""Release-window probabilities from a saved full-train model (net_all.pt), the same call train.py makes after
training (tables from all train months, no time shift). Lets run_all reproduce release.pkl from shipped weights.

Usage: rel_predict.py <run dir with net_all.pt and meta_all.json> --out <release.pkl> [--device cpu --threads 2]
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch

from train import PANELS, Net, PanelData, predict

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--data", type=Path, default=Path(os.environ.get("NN_DATA", HERE / "data")))
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.use_deterministic_algorithms(True)
m = json.load(open(a.run / "meta_all.json"))["args"]
dev = torch.device(a.device)
Ds, off = [], 0
for p in m["panels"].split(","):
    Ds.append(PanelData(a.data / f"{p}.npz", dev, off)); off += Ds[-1].L
net = Net(off, int(m["width"]), int(m["ch"]), drop=float(m["drop"])).to(dev)
net.load_state_dict(torch.load(a.run / "net_all.pt", map_location=dev))
rel = {}
with torch.no_grad():
    for D in Ds:
        for s, r in D.rel.items():
            p = predict(net, D, {x: r[x] for x in ("speed", "flow", "occ", "pct")}, r["t"], r["cond"], 3, 0)
            for i, w in enumerate(r["wid"]):
                rel[str(w)] = {"panel": D.name, "split": s, "cond": int(r["cond"][i]), "p": p[i]}
pickle.dump(rel, open(a.out, "wb"))
print("wrote", a.out, len(rel), "windows")
