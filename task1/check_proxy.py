"""Check metrics.Scorer's S_LWR proxy against the official score_task3.py.

Builds an observation-matched boundary flux for one panel's train split
(inflow chosen so that dt*(in + on - out - off) equals the observed dN), writes
a state submission from temporal interpolation, and runs the official scorer
with --boundary-flux. The two S_FD/S_LWR columns should agree.
"""
import sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import common, metrics, features as F
from task3.score_task3 import load_ramp_flows

HERE = Path(__file__).resolve().parent
p = sys.argv[1] if len(sys.argv) > 1 else "D12_I405_S"
out = HERE / "logs" / "proxy_check"
out.mkdir(parents=True, exist_ok=True)
d = common.load(p, "train")
net = common.network(p)
lanes = net.lanes.to_numpy(np.float32)
nd, nl = len(d["dates"]), len(lanes)
prof = F.profiles([d], lanes)
blk = F.Block(d, np.arange(nd), net, prof)
v = np.where(np.isfinite(blk.f["v_I"]), blk.f["v_I"], blk.pv).reshape(nd, common.T, nl)
q = (np.where(np.isfinite(blk.f["q_I"]), blk.f["q_I"], blk.pq) * lanes).reshape(nd, common.T, nl)

# proxy
sc = metrics.Scorer(p, d)
r = sc.score(v, q)
print(r.to_string())

# submission
tpl = common.read_template(p, "train")
di, ti, li = common.template_index(tpl, d["dates"], d["links"])
tpl["speed_kmh"] = v[di, ti, li]
tpl["flow_vph"] = q[di, ti, li]
tpl.to_csv(out / f"state_{p}.csv", index=False)

# observation-matched flux
P = metrics.panel_params(p, d["links"])
N = d["true_flow"] / np.clip(d["true_speed"], 1, None) * P["length"]
N = N.reshape(nd * common.T, nl)
dN = np.vstack([N[1:] - N[:-1], np.zeros((1, nl))])
ts = pd.date_range(d["dates"][0], periods=nd * common.T, freq="5min", tz="UTC")
flux = pd.DataFrame({"timestamp": np.repeat(ts, nl), "link_id": np.tile(d["links"], nd * common.T),
                     "dN": dN.ravel()})
rf = load_ramp_flows(common.REL / "corridors" / p, "train")
flux = flux.merge(rf[["timestamp", "link_id", "on_ramp_flow_vph", "off_ramp_flow_vph"]],
                  on=["timestamp", "link_id"], how="left").fillna({"on_ramp_flow_vph": 0, "off_ramp_flow_vph": 0})
flux["inflow_vph"] = flux.dN / common.DT_H - flux.on_ramp_flow_vph + flux.off_ramp_flow_vph
flux["outflow_vph"] = 0.0
flux["inflow_vph"] = flux.inflow_vph.fillna(0.0)
flux["timestamp"] = flux.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
flux[["timestamp", "link_id", "inflow_vph", "outflow_vph"]].to_parquet(out / f"flux_{p}.parquet", index=False)

cmd = [sys.executable, "-I", str(common.SRC / "task3" / "score_task3.py"), "--state-submission",
       str(out / f"state_{p}.csv"), "--split", "train", "--boundary-flux", str(out / f"flux_{p}.parquet"),
       "--release-root", str(common.REL), "--panel", p, "--output", str(out / f"official_{p}.csv"),
       "--modes-config", str(common.KAGGLE / "trafficflowbench-public" / "config" / "task3_lwr_modes.json")]
subprocess.run(cmd, check=True, cwd=str(common.KAGGLE / "trafficflowbench-public"))
print(pd.read_csv(out / f"official_{p}.csv").to_string())
