"""Build the dense-array cache for every panel and split (run once)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

panels = sys.argv[1].split(",") if len(sys.argv) > 1 else common.PANELS
splits = sys.argv[2].split(",") if len(sys.argv) > 2 else ["train", "validation", "private"]
for p in panels:
    for s in splits:
        t = time.time()
        d = common.load(p, s)
        print(p, s, d["target"].shape, int(d["target"].sum()), f"{time.time()-t:.1f}s", flush=True)
