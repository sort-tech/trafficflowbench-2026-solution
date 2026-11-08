import sys, time
sys.path.insert(0, '.')
import numpy as np, pandas as pd
import common, metrics, features as F

p = sys.argv[1] if len(sys.argv) > 1 else "D12_I5_N"
d = common.load(p, "train")
net = common.network(p)
lanes = net.lanes.to_numpy(np.float32)
t0 = time.time()
prof = F.profiles([d], lanes)
print("prof", time.time() - t0)
days = np.arange(60, 102)
blk = F.Block(d, days, net, prof)
print("block", time.time() - t0)
sc = metrics.Scorer(p, d, days)
nd = len(days); nl = len(lanes)
def sh(x): return x.reshape(nd, common.T, nl)
res = {}
cands = {
    "profile": (blk.pv, blk.pq),
    "interp": (blk.f["v_I"], blk.f["q_I"]),
    "W2": (blk.f["v_W2"], blk.f["q_W2"]),
    "W6": (blk.f["v_W6"], blk.f["q_W6"]),
    "lv4": (blk.f["v_lv4"], blk.f["q_lv4"]),
}
for name, (v, q) in cands.items():
    v = np.where(np.isfinite(v), v, blk.pv); q = np.where(np.isfinite(q), q, blk.pq)
    r = sc.score(sh(v), sh(q * lanes))
    s = metrics.summarize(r)
    print(f"{name:10s}", " ".join(f"{k}={v:.4f}" for k, v in s.items()), "|", " ".join(f"{a}:{b:.3f}/{c:.3f}" for a, b, c in zip(r.regime, r.S_state, r.S_LWR)))
