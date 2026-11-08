"""Corridor-wide spatio-temporal network for Task 2 (queue forecast), trained on train pseudo windows.

Input per window (rows T-60..T, links upstream -> downstream):
  dynamic [7, 13, L]  speed / v_cut (time-filled), observed flag, flow / capacity, occupancy, pct_observed / 100,
                      eligible observed queue, filled queue
  per link [26, L]    static link attributes, climatology lookups (features_v1.Clim) at T+0/+3/+6/-6, onset prior,
                      time of day, weekend, condition, bottleneck-threshold features (task2/thr_feats.py logic)
  learned per-link embedding
Output [6, L] queue probabilities for T+5..T+30. Onset windows: only T+30 is predicted (queues appear only there).
Loss: masked BCE + per-window soft IoU over eligible cells.

Training windows: every train origin meeting a condition (prep.py), T-row masks drawn per sample with the release
rates (20/30/50 % of eligible cells), time-of-day lookups shifted by U{-12..12} steps with probability --shift-p.
Climatology and thresholds for a training window come from the months outside its own fold (as in task2 LightGBM).

Modes
  --fold A|B|C   fit without that fold, write out-of-fold probabilities for its evaluation windows (shifts 0, +-9, +-12)
  --fold all     fit on all folds, write release (validation + private) probabilities
Every input of a window comes from its own 13 rows (T-60..T) plus train-only tables; nothing after T is read.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

PANELS = ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]
FAMILY = {p: p.rsplit("_", 1)[0] for p in PANELS}
FK = ["A", "B", "C", "all"]
T0_DOW = dt.date(2030, 6, 1).weekday()        # pandas day_of_week convention (Mon = 0)
H, NROW = 6, 13
SHIFTS = (0, 9, -9, 12, -12)
CD, CV = 7, 26


# ------------------------------------------------------------------ data

class PanelData:
    def __init__(self, path: Path, dev: torch.device, link_off: int):
        z = np.load(path, allow_pickle=False)
        f = lambda a, dt_=torch.float32: torch.as_tensor(np.ascontiguousarray(a)).to(dev, dt_)
        self.name = path.stem
        self.L = int(z["vcut"].shape[0])
        self.sp, self.fl, self.oc = f(z["speed"]), f(z["flow"]), f(z["occ"])
        self.pct = f(z["pct"], torch.uint8)
        self.q, self.elig = f(z["q"], torch.bool), f(z["elig"], torch.bool)
        self.vcut, self.cap, self.lanes = f(z["vcut"]), f(z["cap"]), f(z["lanes"])
        st = np.stack([z["frac"], z["lanes"] / 5, np.clip(z["length"], 0, 3), z["vcut"] / 100, z["cap"] / 10000])
        self.static = f(st)
        self.clim_q = f(np.stack([z[f"clim_{k}_q"] for k in FK]))            # [4, 2, 288, L]
        self.clim_any = f(np.stack([z[f"clim_{k}_any"] for k in FK]))        # [4, 2, 288]
        self.clim_on = f(np.stack([z[f"clim_{k}_onset"] for k in FK]))
        self.clim_rate = f(np.stack([z[f"clim_{k}_rate"] for k in FK]))
        self.thr = [self._thr_maps(z[f"thr_{k}"], dev) for k in FK]
        self.link_idx = torch.arange(self.L, device=dev) + link_off
        self.cand_t, self.cand_cond, self.cand_fold = z["cand_t"], z["cand_cond"], z["cand_fold"]
        ex = path.with_name(f"extra_{path.stem}.npz")
        if ex.exists():
            e = np.load(ex)
            self.cand2_t, self.cand2_fold = e["cand2_t"], e["cand2_fold"]
        else:
            self.cand2_t, self.cand2_fold = np.zeros(0, np.int64), np.zeros(0, np.int8)
        self.ev = {k: z[f"ev_{k}"] for k in ("t", "cond", "fold", "speed", "flow", "occ", "pct")}
        self.rel = {s: {k: z[f"rel_{s}_{k}"] for k in ("wid", "t", "cond", "speed", "flow", "occ", "pct")}
                    for s in ("validation", "private")}

    def _thr_maps(self, thr: np.ndarray, dev):
        """Static link -> group maps of thr_feats.window_features (nearest group at/downstream; nearest within 3)."""
        L = self.L
        G = len(thr)
        down = np.full(L, -1); ddist = np.full(L, 99.0); near = np.full(L, -1)
        for l in range(L):
            cand = [(max(int(a) - l, 0), k) for k, (a, b, h, c) in enumerate(thr) if b >= l]
            if cand:
                ddist[l], down[l] = min(cand)
            if G:
                nb = [((min(abs(l - a), abs(l - b)) if not a <= l <= b else 0), k) for k, (a, b, h, c) in enumerate(thr)]
                dn, kn = min(nb)
                if dn <= 3:
                    near[l] = kn
        t = lambda a, d=torch.long: torch.as_tensor(a).to(dev, d)
        return {"G": G, "ref": t(thr[:, 2].astype(int) if G else np.zeros(0, int)), "c": t(thr[:, 3] if G else np.zeros(0), torch.float32),
                "down": t(down), "ddist": t(ddist, torch.float32), "near": t(near)}

    def gather(self, t: np.ndarray, mask_rand: np.ndarray | None, rates: np.ndarray | None):
        dev = self.sp.device
        tt = torch.as_tensor(t, device=dev)
        idx = tt[:, None] + torch.arange(-NROW + 1, 1, device=dev)[None]
        sp, fl, oc, pct = self.sp[idx], self.fl[idx], self.oc[idx], self.pct[idx]
        if mask_rand is not None:
            m = (pct[:, -1] >= 75) & (torch.as_tensor(mask_rand, device=dev) < torch.as_tensor(rates, device=dev)[:, None])
            for a in (sp, fl, oc):
                a[:, -1][m] = float("nan")
        hidx = tt[:, None] + torch.arange(1, H + 1, device=dev)[None]
        return sp, fl, oc, pct, self.q[hidx], self.elig[hidx]


def ffill_time(a: torch.Tensor, default: float) -> torch.Tensor:
    """[B, 13, L]: forward fill along time, then back fill, then default."""
    a = a.clone()
    for i in range(1, a.shape[1]):
        m = torch.isnan(a[:, i]); a[:, i][m] = a[:, i - 1][m]
    for i in range(a.shape[1] - 2, -1, -1):
        m = torch.isnan(a[:, i]); a[:, i][m] = a[:, i + 1][m]
    return torch.nan_to_num(a, nan=default)


def causal_q(flow: torch.Tensor, lanes: torch.Tensor) -> torch.Tensor:
    """thr_feats.causal_q on [B, 13, L]: mean of observed values in the last 3 slots, else the last observed."""
    q = flow / torch.clamp(lanes, min=1)
    ok = torch.isfinite(q)
    q0 = torch.where(ok, q, torch.zeros_like(q))
    out = torch.full_like(q, float("nan"))
    last = torch.full_like(q[:, 0], float("nan"))
    for i in range(q.shape[1]):
        a = max(0, i - 2)
        n = ok[:, a:i + 1].sum(1); s = q0[:, a:i + 1].sum(1)
        last = torch.where(ok[:, i], q[:, i], last)
        out[:, i] = torch.where(n > 0, s / torch.clamp(n, min=1), last)
    return out


def thr_features(D: PanelData, key: torch.Tensor, fl: torch.Tensor) -> torch.Tensor:
    """[B, 10, L] threshold features (thr_feats.THR_COLS, NaN -> 0, plus availability flags), per-sample table key."""
    B, L = fl.shape[0], D.L
    out = torch.zeros(B, 10, L, device=fl.device)
    q = causal_q(fl, D.lanes)
    for k in torch.unique(key).tolist():
        sel = (key == k).nonzero().squeeze(1)
        M = D.thr[k]
        out[sel, 0] = torch.clamp(M["ddist"], max=20)[None] / 20
        if M["G"] == 0:
            continue
        qs = q[sel]
        qr = qs[:, :, M["ref"]]                                         # [b, 13, G]
        g_r = qr[:, -1] / M["c"]; g_tr = (qr[:, -1] - qr[:, -4]) / M["c"]
        win = qr[:, -7:]
        g_mx = torch.where(torch.isnan(win), torch.full_like(win, -1e9), win).max(1).values
        g_mx = torch.where(g_mx < -1e8, torch.full_like(g_mx, float("nan")), g_mx) / M["c"]
        order = torch.argsort(-torch.nan_to_num(g_r, nan=-9.0), dim=1, stable=True)
        rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(M["G"], device=fl.device).expand_as(order))
        has = M["down"] >= 0
        di = torch.clamp(M["down"], min=0)
        vals = [g_r[:, di], g_tr[:, di], g_mx[:, di], 1 - g_r[:, di], rank[:, di].float() / 8]
        for j, v in enumerate(vals):
            out[sel, 2 + j] = torch.where(has[None], torch.nan_to_num(v, nan=0.0), torch.zeros_like(v)).clamp(-3, 3)
        out[sel, 1] = has.float()[None].expand(len(sel), L)
        hn = M["near"] >= 0
        rl = qs[:, -1] / M["c"][torch.clamp(M["near"], min=0)][None]
        rl = torch.where(hn[None], rl, torch.full_like(rl, float("nan")))
        rlp = Fn.pad(torch.where(torch.isnan(rl), torch.full_like(rl, -1e9), rl)[:, None], (3, 3), value=-1e9)
        nb = Fn.max_pool1d(rlp, 7, 1)[:, 0]
        nb = torch.where(nb < -1e8, torch.zeros_like(nb), nb)
        out[sel, 7] = torch.nan_to_num(rl, nan=0.0).clamp(-3, 3)
        out[sel, 8] = torch.isfinite(rl).float()
        out[sel, 9] = nb.clamp(-3, 3)
    return out


def make_inputs(D: PanelData, sp, fl, oc, pct, t_idx: np.ndarray, cond: np.ndarray, key: np.ndarray, shift: np.ndarray):
    dev = sp.device
    ratio = sp / D.vcut
    obs = (~torch.isnan(sp)).float()
    r = ffill_time(ratio, 1.67).clamp(0, 2)
    f = ffill_time(fl / D.cap, 0.5).clamp(0, 2)
    o = (ffill_time(oc, 0.05) * 4).clamp(0, 2)
    p = pct.float() / 100
    qe = (torch.nan_to_num(ratio, nan=9.0) <= 1.0) & (pct >= 75)
    dyn = torch.stack([r, obs, f, o, p, qe.float(), (r <= 1.0).float()], 1)          # [B, 7, 13, L]
    ts = torch.as_tensor(t_idx + shift, device=dev)
    tod = ts % 288
    dw = ((ts // 288 + T0_DOW) % 7 >= 5).long()
    kk = torch.as_tensor(key, device=dev)
    B, L = sp.shape[0], D.L
    cq = lambda h: D.clim_q[kk, dw, (tod + h) % 288]                                 # [B, L]
    on = D.clim_on[kk, dw, (tod + H) % 288]
    rate = (D.clim_rate[kk, dw, (tod + H) % 288] * 50).clamp(0, 3)
    anyq = D.clim_any[kk, dw, tod]
    ang = tod.float() * (2 * np.pi / 288)
    bc = lambda v: v[:, None].expand(B, L)
    vec = torch.stack([*D.static[:, None].expand(5, B, L), cq(0), cq(3), cq(6), cq(-6), bc(anyq), on, bc(rate),
                       bc(torch.sin(ang)), bc(torch.cos(ang)), bc(dw.float()),
                       bc(torch.as_tensor(cond, device=dev).float())], 1)               # [B, 16, L]
    vec = torch.cat([vec, thr_features(D, kk, fl)], 1)                                # [B, 26, L]
    return dyn, vec


# ------------------------------------------------------------------ model

class Block(nn.Module):
    def __init__(self, c: int, dil: int, drop: float):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(1, c), nn.GroupNorm(1, c)
        self.c1 = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
        self.c2 = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.c1(Fn.gelu(self.n1(x)))
        h = self.c2(self.drop(Fn.gelu(self.n2(h))))
        return x + h


class Net(nn.Module):
    def __init__(self, n_links: int, w: int = 32, c: int = 96, emb: int = 8, drop: float = 0.1,
                 dils=(1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        self.emb = nn.Embedding(n_links, emb)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.stem = nn.Sequential(nn.Conv2d(CD, w, (3, 5), padding=(1, 2)), nn.GELU(),
                                  nn.Conv2d(w, w, (3, 5), padding=(1, 2)), nn.GELU())
        self.tcol = nn.Conv2d(w, 2 * w, (NROW, 1))
        self.inp = nn.Conv1d(2 * w + w + CD + CV + emb, c, 1)
        self.blocks = nn.Sequential(*[Block(c, d, drop) for d in dils])
        self.out = nn.Sequential(nn.GroupNorm(1, c), nn.GELU(), nn.Conv1d(c, H, 1))

    def forward(self, dyn, vec, link_idx):
        s = self.stem(dyn)                                       # [B, w, 13, L]
        x = torch.cat([self.tcol(s)[:, :, 0], s[:, :, -1], dyn[:, :, -1], vec,
                       self.emb(link_idx).T[None].expand(dyn.shape[0], -1, -1)], 1)
        return self.out(self.blocks(self.inp(x)))                 # [B, 6, L] logits


def out_mask(cond: torch.Tensor, B: int, L: int, dev) -> torch.Tensor:
    """Onset windows: only T+30 can be queued (public property of queue_onset windows)."""
    m = torch.ones(B, H, L, device=dev)
    m[cond.bool(), :H - 1] = 0
    return m


def loss_fn(logit, y, elig, om, lam: float):
    m = elig.float() * om
    bce = (Fn.binary_cross_entropy_with_logits(logit, y.float(), reduction="none") * m).sum() / m.sum().clamp(min=1)
    p = torch.sigmoid(logit) * om
    yy = y.float()
    inter = (p * yy * m).sum((1, 2)); union = ((p + yy - p * yy) * m).sum((1, 2))
    siou = 1 - (inter + 1) / (union + 1)
    return bce + lam * siou.mean(), bce.detach(), siou.mean().detach()


# ------------------------------------------------------------------ scoring (quick check; task2 code does the official one)

def best_k_set(p: np.ndarray) -> np.ndarray:
    order = np.argsort(-p, kind="stable"); ps = p[order]; cum = np.cumsum(ps)
    k = np.arange(1, len(p) + 1); score = cum / (k + (cum[-1] - cum))
    out = np.zeros(len(p), bool); out[order[:int(np.argmax(score)) + 1]] = True
    return out


def iou(pred, y, e):
    if not e.any():
        return 0.0
    p, t = pred & e, y & e
    u = (p | t).sum()
    return 1.0 if u == 0 else float((p & t).sum() / u)


def aggregate(rows):
    pc = {}
    for r in rows:
        pc.setdefault((r["panel"], r["cond"]), []).append(r["iou"])
    pm = {}
    for (p, c), v in pc.items():
        pm.setdefault(p, []).append(np.mean(v))
    fam = {}
    for p, v in pm.items():
        fam.setdefault(FAMILY[p], []).append(np.mean(v))
    on = np.mean([np.mean(v) for (p, c), v in pc.items() if c == 1])
    go = np.mean([np.mean(v) for (p, c), v in pc.items() if c == 0])
    return float(np.mean([np.mean(v) for v in fam.values()])), float(on), float(go)


# ------------------------------------------------------------------ main

@torch.no_grad()
def predict(net, D: PanelData, arrs: dict, t: np.ndarray, cond: np.ndarray, key: int, shift: int, bs: int = 128):
    net.eval()
    out = []
    dev = D.sp.device
    for i in range(0, len(t), bs):
        sl = slice(i, i + bs)
        f = lambda a, d=torch.float32: torch.as_tensor(arrs[a][sl]).to(dev, d)
        sp, fl, oc, pct = f("speed"), f("flow"), f("occ"), f("pct", torch.uint8)
        n = len(t[sl])
        dyn, vec = make_inputs(D, sp, fl, oc, pct, t[sl], cond[sl], np.full(n, key), np.full(n, shift))
        lg = net(dyn, vec, D.link_idx)
        p = torch.sigmoid(lg) * out_mask(torch.as_tensor(cond[sl], device=dev), n, D.L, dev)
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fold", choices=FK, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=1.0, help="soft-IoU weight")
    ap.add_argument("--onset-frac", type=float, default=0.3)
    ap.add_argument("--shift-p", type=float, default=0.7)
    ap.add_argument("--shift-max", type=int, default=12)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--ch", type=int, default=96)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=3000)
    ap.add_argument("--easy-frac", type=float, default=0.0,
                    help="share of ongoing samples drawn from origins above the persistence cap (prep_extra.py)")
    ap.add_argument("--first-frac", type=float, default=0.0,
                    help="share of ongoing samples replaced by first qualifying origins (origin t - 1 was not an ongoing "
                         "candidate), the kind the greedy window selector picks")
    ap.add_argument("--panel-weight", choices=["sqrt", "uniform"], default="sqrt")
    ap.add_argument("--panels", default=",".join(PANELS), help="smoke tests only")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    Ds, off = [], 0
    for p in args.panels.split(","):
        Ds.append(PanelData(args.data / f"{p}.npz", dev, off)); off += Ds[-1].L
    fk = FK.index(args.fold)
    tr = []
    for D in Ds:
        keep = D.cand_fold != fk if args.fold != "all" else np.ones(len(D.cand_t), bool)
        tr.append({c: np.nonzero(keep & (D.cand_cond == c))[0] for c in (0, 1)})
        tr[-1]["easy"] = np.nonzero(D.cand2_fold != fk)[0] if args.fold != "all" else np.arange(len(D.cand2_t))
        is_go = D.cand_cond == 0
        prev = np.isin(D.cand_t - 1, D.cand_t[is_go])
        tr[-1]["first"] = np.nonzero(keep & is_go & ~prev)[0]
    pw = np.sqrt([len(t[0]) + len(t[1]) for t in tr]) if args.panel_weight == "sqrt" else np.ones(len(tr))
    pw = pw / pw.sum()
    net = Net(off, args.width, args.ch, drop=args.drop).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    nparam = sum(p.numel() for p in net.parameters())
    log = open(args.out / "train.log", "a")
    say = lambda s: (print(s, flush=True), log.write(s + "\n"), log.flush())
    say(f"start {time.strftime('%F %T')} fold {args.fold} device {dev} params {nparam} args {vars(args)}")
    t0 = time.time()
    run = []

    def evaluate(tag):
        rows, probs = [], {}
        for D in Ds:
            ev = D.ev
            sel = np.nonzero(ev["fold"] == fk)[0] if args.fold != "all" else np.zeros(0, int)
            if not len(sel):
                continue
            arrs = {a: ev[a][sel] for a in ("speed", "flow", "occ", "pct")}
            t, cond = ev["t"][sel], ev["cond"][sel]
            ps = {sh: predict(net, D, arrs, t, cond, fk, sh) for sh in SHIFTS}
            y = D.q[torch.as_tensor(t, device=dev)[:, None] + torch.arange(1, H + 1, device=dev)].cpu().numpy()
            e = D.elig[torch.as_tensor(t, device=dev)[:, None] + torch.arange(1, H + 1, device=dev)].cpu().numpy()
            for i in range(len(t)):
                probs[(D.name, int(t[i]))] = {f"nn_{sh}": (ps[sh][i, H - 1] if cond[i] else ps[sh][i]) for sh in SHIFTS}
                if cond[i]:
                    pr = np.zeros((H, D.L), bool); pr[H - 1] = best_k_set(ps[0][i, H - 1])
                else:
                    pr = ps[0][i] >= 0.37
                rows.append({"panel": D.name, "cond": int(cond[i]), "iou": iou(pr, y[i], e[i])})
        if rows:
            s, on, go = aggregate(rows)
            say(f"eval {tag}: S_q {s:.4f} onset {on:.4f} ongoing {go:.4f} n {len(rows)}")
        return probs

    for step in range(1, args.steps + 1):
        net.train()
        pi = rng.choice(len(Ds), p=pw)
        D = Ds[pi]
        n_on = rng.binomial(args.bs, args.onset_frac) if len(tr[pi][1]) else 0
        ii = np.concatenate([rng.choice(tr[pi][1], n_on), rng.choice(tr[pi][0], args.bs - n_on)])
        t = D.cand_t[ii]; cond = D.cand_cond[ii]
        key = D.cand_fold[ii].astype(np.int64)          # tables built without the window's own fold
        if args.easy_frac and len(tr[pi]["easy"]):
            ez = np.nonzero((cond == 0) & (rng.random(args.bs) < args.easy_frac))[0]
            jj = rng.choice(tr[pi]["easy"], len(ez))
            t = t.copy(); t[ez] = D.cand2_t[jj]; key = key.copy(); key[ez] = D.cand2_fold[jj]
        if args.first_frac and len(tr[pi]["first"]):
            fz = np.nonzero((cond == 0) & (rng.random(args.bs) < args.first_frac))[0]
            jj = rng.choice(tr[pi]["first"], len(fz))
            t = t.copy(); t[fz] = D.cand_t[jj]; key = key.copy(); key[fz] = D.cand_fold[jj]
        rates = rng.choice([0.2, 0.3, 0.5], args.bs).astype(np.float32)
        mr = rng.random((args.bs, D.L), dtype=np.float32)
        sh = np.where(rng.random(args.bs) < args.shift_p, rng.integers(-args.shift_max, args.shift_max + 1, args.bs), 0)
        sp, fl, oc, pct, y, e = D.gather(t, mr, rates)
        dyn, vec = make_inputs(D, sp, fl, oc, pct, t, cond, key, sh)
        lg = net(dyn, vec, D.link_idx)
        om = out_mask(torch.as_tensor(cond, device=dev), args.bs, D.L, dev)
        loss, bce, siou = loss_fn(lg, y, e, om, args.lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run.append((loss.item(), bce.item(), siou.item()))
        if step % 500 == 0:
            a = np.mean(run, 0); run = []
            say(f"step {step} loss {a[0]:.4f} bce {a[1]:.4f} siou {a[2]:.4f} {time.time() - t0:.0f}s")
        if args.fold != "all" and step % args.eval_every == 0 and step < args.steps:
            evaluate(f"step {step}")
    torch.save(net.state_dict(), args.out / f"net_{args.fold}.pt")
    meta = {"fold": args.fold, "device": str(dev), "seconds": time.time() - t0, "params": nparam, "args": {k: str(v) for k, v in vars(args).items()},
            "torch": torch.__version__}
    if args.fold != "all":
        probs = evaluate("final")
        pickle.dump(probs, open(args.out / f"oof_{args.fold}.pkl", "wb"))
    else:
        rel = {}
        for D in Ds:
            for s, r in D.rel.items():
                arrs = {a: r[a] for a in ("speed", "flow", "occ", "pct")}
                p = predict(net, D, arrs, r["t"], r["cond"], 3, 0)
                for i, w in enumerate(r["wid"]):
                    rel[str(w)] = {"panel": D.name, "split": s, "cond": int(r["cond"][i]), "p": p[i]}
        pickle.dump(rel, open(args.out / "release.pkl", "wb"))
    json.dump(meta, open(args.out / f"meta_{args.fold}.json", "w"), indent=1)
    say(f"done {time.strftime('%F %T')} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
