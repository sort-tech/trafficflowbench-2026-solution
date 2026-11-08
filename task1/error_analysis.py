"""Where does the Task 1 + Task 3 loss come from?

Loss of one panel/regime against a perfect answer (proxy):
    L = 0.35*(0.54*RMSE_v/25 + 0.46*RMSE_q/600) + 0.15*(2/3)*(1 - S_LWR) + 0.15*(1/3)*(1 - S_FD)
Each target cell gets a share: the speed and flow terms by its share of the squared
error, the S_LWR term by the conservation errors |dN_sub - dN_obs| of the transitions
it sits on (split between the target ends). S_FD is left out (its loss is ~0.002).
Panel/regime losses are weighted as in the official mean (regime 1/3, panel within
family, family 1/5), so the cell shares add up to the overall loss.

    python -I error_analysis.py <ds> <pred_tag> [beta beta_c vthr]
writes logs/error_breakdown.csv (one row per cell group) and prints summaries.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import common
import metrics
from pipeline import reconcile, SPEED_CAP

HERE = Path(__file__).resolve().parent
dsn, tag = sys.argv[1], sys.argv[2]
rc = tuple(float(x) for x in sys.argv[3:6]) if len(sys.argv) > 5 else (0.2, 0.8, 80.0)
PEAK = set(range(13, 21))  # hours with >6% congested cells in train


def run_class(miss: np.ndarray):
    """Length of the run of consecutive missing cells (time axis) each cell sits in."""
    n, m = miss.shape
    out = np.zeros((n, m), np.int32)
    for j in range(m):
        col = miss[:, j]
        i = 0
        while i < n:
            if col[i]:
                k = i
                while k < n and col[k]:
                    k += 1
                out[i:k, j] = k - i
                i = k
            else:
                i += 1
    return out


fam_n = pd.Series(common.FAMILY).value_counts()
groups = []
for p in common.PANELS:
    ev = pd.read_parquet(HERE / dsn / f"eval_{p}.parquet", columns=["day", "t", "li", "regime", "win_cond", "y_v", "y_q"])
    pr = pd.read_parquet(HERE / "preds" / f"{tag}_{p}.parquet")
    d = common.load(p, "train")
    net = common.network(p)
    lanes = net.lanes.to_numpy(np.float32)
    P = metrics.panel_params(p, d["links"])
    li = ev.li.to_numpy()
    vmin = np.nanmin(d["true_speed"], axis=(0, 1))
    qmax = np.nanmax(d["true_flow"], axis=(0, 1)) / lanes
    v, q = reconcile(np.clip(pr.pv.to_numpy(), vmin[li], SPEED_CAP), pr.pq.to_numpy(), pr.pk.to_numpy(), 1.0, *rc)
    q = np.clip(q, 1.0, 1.1 * qmax[li])
    days = np.unique(ev.day.to_numpy())
    pos = {x: i for i, x in enumerate(days)}
    di = ev.day.map(pos).to_numpy(); ti = ev.t.to_numpy()
    nd, nl = len(days), len(lanes)
    # run length of the visible gap around each target (masked layer + simulated window gaps)
    obs = np.isfinite(d["obs_speed_kmh"][days])
    gap = np.zeros((nd, common.T, nl), bool)
    gap[di, ti, li] |= ev.win_cond.to_numpy() > 0
    miss = (~obs | gap).reshape(nd * common.T, nl)
    runs = run_class(miss).reshape(nd, common.T, nl)[di, ti, li]
    # squared errors (per-lane flow as scored)
    sl = P["score_lanes"][li]
    se_v = (v - ev.y_v.to_numpy()) ** 2
    se_q = ((q - ev.y_q.to_numpy()) * lanes[li] / sl) ** 2
    # conservation errors per transition, attributed to target ends
    tv = d["true_speed"][days].astype(float); tq = d["true_flow"][days].astype(float)
    el = d["elig"][days]
    sv = tv.copy(); sq = tq.copy()
    tgt = np.zeros((nd, common.T, nl), bool); tgt[di, ti, li] = True
    sv[di, ti, li] = v; sq[di, ti, li] = q * lanes[li]
    N = sq / np.clip(sv, 1, None) * P["length"]
    N0 = tq / np.clip(tv, 1, None) * P["length"]
    on, off = metrics.ramp_valid(p, "train", d["dates"], d["links"])
    ro = (on & off)[days]
    reg_day = d["regime"][days]
    lwr_cell = np.zeros((nd, common.T, nl))
    D = {}
    for r in common.REGIMES:
        sel = np.where(reg_day == r)[0]
        if len(sel) == 0:
            continue
        f = lambda a: a[sel].reshape(len(sel) * common.T, nl)
        Nf, N0f, ef, rof, tf = f(N), f(N0), f(el), f(ro), f(tgt)
        cont = np.ones(len(sel) * common.T - 1, bool)
        dd = pd.to_datetime(pd.Series(d["dates"][days][sel])).to_numpy()
        gapd = np.diff(dd) != np.timedelta64(1, "D")
        cont[(np.arange(1, len(sel)) * common.T - 1)[gapd]] = False
        ok = ef[:-1] & ef[1:] & rof[:-1] & cont[:, None]
        err = np.where(ok, np.abs((Nf[1:] - Nf[:-1]) - (N0f[1:] - N0f[:-1])), 0)
        D[r] = np.abs(N0f[1:] - N0f[:-1])[ok].sum()
        a, b = tf[:-1], tf[1:]
        both = a & b
        share = np.zeros_like(Nf)
        share[:-1] += np.where(both, err / 2, np.where(a, err, 0))
        share[1:] += np.where(both, err / 2, np.where(b, err, 0))
        lwr_cell[sel] = share.reshape(len(sel), common.T, nl)
    lw = lwr_cell[di, ti, li]
    df = pd.DataFrame({"panel": p, "regime": ev.regime.to_numpy(), "se_v": se_v, "se_q": se_q, "lwr": lw,
                       "y_v": ev.y_v.to_numpy(), "runs": runs, "wgap": ev.win_cond.to_numpy() > 0,
                       "hour": ti // 12, "len": P["length"][li]})
    # per panel-regime loss attribution
    out = []
    for r, g in df.groupby("regime"):
        w = (1 / 3) * (1 / fam_n[common.FAMILY[p]]) * (1 / 5)
        n = len(g)
        rm_v = np.sqrt(g.se_v.mean()); rm_q = np.sqrt(g.se_q.mean())
        g = g.copy()
        g["loss_v"] = w * 0.35 * 0.54 * (rm_v / 25) * g.se_v / g.se_v.sum()
        g["loss_q"] = w * 0.35 * 0.46 * (rm_q / 600) * g.se_q / g.se_q.sum()
        g["loss_lwr"] = w * 0.15 * (2 / 3) * g.lwr / D[r]
        out.append(g)
    df = pd.concat(out)
    df["state"] = np.select([df.y_v < 60, df.y_v < 95], ["cong", "mid"], "free")
    df["gapcls"] = np.select([df.wgap, df.runs <= 1, df.runs <= 3, df.runs <= 11], ["task2gap", "1", "2-3", "4-11"], "12+")
    df["peak"] = np.where(df.hour.isin(PEAK), "peak", "offpeak")
    keys = ["panel", "regime", "state", "gapcls", "peak"]
    groups.append(df.groupby(keys).agg(n=("se_v", "size"), sse_v=("se_v", "sum"), sse_q=("se_q", "sum"),
                                         lwr_err=("lwr", "sum"), loss_v=("loss_v", "sum"), loss_q=("loss_q", "sum"),
                                         loss_lwr=("loss_lwr", "sum")).reset_index())
    print(p, flush=True)
G = pd.concat(groups, ignore_index=True)
G["loss"] = G.loss_v + G.loss_q + G.loss_lwr
G.to_csv(HERE / "logs" / "error_breakdown.csv", index=False)
tot = G.loss.sum()
print(f"total attributed loss {tot:.4f} (speed {G.loss_v.sum():.4f}, flow {G.loss_q.sum():.4f}, LWR {G.loss_lwr.sum():.4f})")


def view(by):
    t = G.groupby(by).agg(n=("n", "sum"), sse_v=("sse_v", "sum"), sse_q=("sse_q", "sum"), loss_v=("loss_v", "sum"),
                          loss_q=("loss_q", "sum"), loss_lwr=("loss_lwr", "sum"), loss=("loss", "sum"))
    t["cells%"] = 100 * t.n / t.n.sum()
    t["rmse_v"] = np.sqrt(t.sse_v / t.n)
    t["rmse_q"] = np.sqrt(t.sse_q / t.n)
    t["loss%"] = 100 * t.loss / tot
    t["lwr_loss%"] = 100 * t.loss_lwr / tot
    print(t[["cells%", "rmse_v", "rmse_q", "loss", "loss%", "lwr_loss%"]].round(4).to_string())
    print()


for by in (["state"], ["gapcls"], ["peak"], ["regime"], ["panel"], ["state", "gapcls"], ["regime", "gapcls"]):
    view(by)
top = G.groupby(["panel", "state", "gapcls"]).loss.sum().sort_values(ascending=False).head(15)
print((100 * top / tot).round(2).to_string())
