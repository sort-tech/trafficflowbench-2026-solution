"""Extra ongoing training origins without the persistence cap (persistence IoU > 0.9), every 5-min origin.
Writes data/extra_<panel>.npz: cand2_t, cand2_fold. Same Panel / clean-queue inputs as prep.py."""
from __future__ import annotations
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
T2 = Path(os.environ.get("T2_DIR", HERE.parent / "task2"))   # task2 code + cache (run_all: the fresh task2 build)
sys.path.insert(0, str(T2))
DATA = Path(os.environ.get("NN_DATA", HERE / "data"))         # prep.py output
sys.path.insert(0, str(HERE))
import numpy as np
import features_v1 as F1
from common import CACHE, PANELS, Panel
from windows import candidates
from prep import month_of

def main():
    m2f = {m: i for i, (f, ms) in enumerate(F1.FOLDS.items()) for m in ms}
    for panel in PANELS:
        P = Panel(panel); qc = np.load(CACHE / f"{panel}_qclean.npy")
        a = candidates(P, qc, cap=1.0, stride=1); b = candidates(P, qc, stride=1)
        a = a[a.condition == "queue_ongoing"]; a = a[~a.t.isin(b.t)]
        t = a.t.to_numpy(np.int64)
        np.savez(DATA / f"extra_{panel}.npz", cand2_t=t, cand2_fold=np.array([m2f[m] for m in month_of(t)], np.int8))
        print(panel, len(t), flush=True)

if __name__ == "__main__":
    main()
