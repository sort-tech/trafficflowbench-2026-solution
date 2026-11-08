"""Build a Task 2 queue submission for validation or private from scratch.

Pipeline (each stage is cached under task2/cache and task2/models; --fresh redoes everything):
  1. prep_cache.py      train mainline observations -> dense arrays
     prep_release.py    validation/private masked observations -> dense arrays (online features read rows < T only)
  2. windows.py         pseudo windows on train with the imitated organizer selector
  3. build_dataset.py   per-link features + labels for the pseudo windows
  4. fit                LightGBM onset model (T+30 cells) and ongoing model (T+5..T+30 cells)
  5. predict            release windows: window_history (T-60..T-5) + the released row at T only

Usage:
  build.py --split validation --out candidates/<name>/queue_validation.csv
  build.py --split all --out candidates/<name>/queue_submission.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq

from common import CACHE, H, PANELS, REL, STEP, Panel
from features import Clim, Static, dow_of, link_features, ongoing_rows, onset_rows, online_source, release_windows
from model import NON_FEATURES, ONLINE_FEATURES, SEED, VARIANTS, best_k_set, feature_cols, fit_dataset, make_dataset

MODELS = HERE / "models"
ROUNDS = {"onset": 300, "ongoing": 300}
N_SEEDS = {"onset": 3, "ongoing": 3}
ONGOING_THR = 0.42
# ongoing decision threshold per model variant (CV: full 0.40-0.45, notod/online 0.35)
THR_BY_VARIANT = {"full": 0.42, "notod": 0.35, "online": 0.35, "phys": 0.35, "onphys": 0.35, "aug": 0.35}
PY = [sys.executable, "-I"]


def ensure_stages(fresh: bool) -> None:
    if fresh or not all((CACHE / f"{p}_train.npz").exists() for p in PANELS):
        subprocess.run(PY + [str(HERE / "prep_cache.py")], check=True)
    if fresh or not (CACHE / "pseudo_windows.parquet").exists():
        subprocess.run(PY + [str(HERE / "windows.py")], check=True)
    if fresh or not all((CACHE / f"{p}_{s}.npz").exists() for p in PANELS for s in ("validation", "private")):
        subprocess.run(PY + [str(HERE / "prep_release.py")], check=True)
    missing = [p for p in PANELS if fresh or not (CACHE / f"ds_{p}.parquet").exists()]
    if missing:
        args = PY + [str(HERE / "build_dataset.py"), "--max-ongoing-train", "1200"]
        for p in missing:
            args += ["--panel", p]
        subprocess.run(args, check=True)


def load_train(cond: int) -> pd.DataFrame:
    """Training rows only (visible label; ongoing rows inside the input filter), read column-pruned."""
    filters = [("cond", "=", cond), ("vis", "=", 1)] + ([("keep", "=", 1)] if cond == 0 else [])
    drop = {"vis", "elig", "yc", "keep", "t", "cond", "greedy", "fold", "link"}
    cols = [c for c in pq.read_schema(CACHE / f"ds_{PANELS[0]}.parquet").names if c not in drop]
    parts = [pd.read_parquet(CACHE / f"ds_{p}.parquet", columns=cols, filters=filters) for p in PANELS]
    return pd.concat(parts, ignore_index=True)


def load_variant(name: str, var: str, fresh: bool) -> list[lgb.Booster]:
    """Seed-bagged boosters of one variant for one condition (fit and cache if missing)."""
    cond = 1 if name == "onset" else 0
    mdir = MODELS if var == "full" else MODELS / var
    mdir.mkdir(parents=True, exist_ok=True)
    paths = [mdir / f"{name}_s{i}.txt" for i in range(N_SEEDS[name])]
    if fresh or not all(p.exists() for p in paths):
        df = load_train(cond)
        feats = [f for f in feature_cols(df) if f not in VARIANTS[var]]
        ds = make_dataset(df, feats)
        del df
        for i, path in enumerate(paths):
            m = fit_dataset(ds, ROUNDS[name], {"seed": SEED + 1000 * i})
            m.save_model(str(path))
            print(f"fit {name} ({var}) seed {i}: {m.num_trees()} trees", flush=True)
        del ds
    return [lgb.Booster(model_file=str(p)) for p in paths]


def get_models(fresh: bool, mixes: dict[str, dict[str, float]] | None = None) -> dict[str, list[tuple[float, list[lgb.Booster]]]]:
    """mixes: condition -> {variant: weight}. Each variant is a seed-bagged model; probabilities are
    averaged over seeds, then combined with the weights."""
    mixes = mixes or {"onset": {"full": 1.0}, "ongoing": {"full": 1.0}}
    return {name: [(w, load_variant(name, var, fresh)) for var, w in mix.items()] for name, mix in mixes.items()}


def predict_mean(models, df: pd.DataFrame) -> np.ndarray:
    """models: list of boosters (seed bag) or list of (weight, seed bag) pairs."""
    if models and isinstance(models[0], tuple):
        tot = sum(w for w, _ in models)
        return sum(w * predict_mean(bag, df) for w, bag in models) / tot
    return np.mean([m.predict(df[m.feature_name()], num_threads=3) for m in models], axis=0)


def parse_mix_h(text: str | None) -> list[dict[str, float]] | None:
    """'1-2=v1:.1,phys:.3;3-4=...;5-6=...' -> six weight dicts for T+5 .. T+30 (every horizon must be covered)."""
    if not text:
        return None
    out: list = [None] * H
    for part in text.split(";"):
        hs, _, mix = part.partition("=")
        a, _, b = hs.partition("-")
        for h in range(int(a), int(b or a) + 1):
            out[h - 1] = parse_mix(mix)
    assert all(w is not None for w in out), "--ongoing-mix-h must cover horizons 1..6"
    return out


def parse_mix(text: str) -> dict[str, float]:
    out = {}
    for item in text.split(","):
        var, _, w = item.partition(":")
        out[var] = float(w) if w else 1.0
    return out


def predict_split(split: str, models: dict[str, list[lgb.Booster]], ongoing_thr: float = ONGOING_THR) -> pd.DataFrame:
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    out = []
    for panel in PANELS:
        st = Static(panel)
        P = Panel(panel)
        qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        months = sorted(pd.Series(W.month.unique()).tolist())
        clim = Clim(P, qc, months, onset_t)
        del P, qc
        tmpl = pd.read_csv(REL / "task2" / panel / split / "sample_submission_queue.csv", dtype=str)
        online_f = online_source(panel, split, st)
        for w, arr, t_idx in release_windows(panel, split, st):
            df = link_features(arr, st, clim, t_idx, online_f(t_idx))
            if w.condition == "queue_onset":
                df = onset_rows(df, clim, t_idx)
                p = predict_mean(models["onset"], df)
                pred = best_k_set(p)
                cells = pd.DataFrame({"h": H, "link": np.arange(st.L), "queue_pred": pred.astype(int)})
            else:
                from build_dataset import keep_mask
                df = ongoing_rows(df, clim, t_idx)
                p = predict_mean(models["ongoing"], df)
                p[keep_mask(df) == 0] = 0.0
                cells = pd.DataFrame({"h": df.h.to_numpy().astype(int), "link": df.link.to_numpy().astype(int),
                                      "queue_pred": (p >= ongoing_thr).astype(int)})
            T = pd.Timestamp(w.forecast_origin)
            ts = {h: (T + h * STEP).strftime("%Y-%m-%dT%H:%M:%SZ") for h in range(1, H + 1)}
            cells["timestamp"] = cells.h.map(ts)
            cells["link_id"] = st.link_ids[cells.link.to_numpy()]
            cells["window_id"] = w.window_id
            t = tmpl.loc[tmpl.window_id == w.window_id, ["window_id", "timestamp", "link_id"]]
            t = t.merge(cells[["window_id", "timestamp", "link_id", "queue_pred"]], how="left",
                        on=["window_id", "timestamp", "link_id"])
            t["queue_pred"] = t.queue_pred.fillna(0).astype(int)
            out.append(t[["window_id", "timestamp", "link_id", "queue_pred"]])
        print(panel, split, flush=True)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- lgb_v1 recipe (frozen features_v1.py)

V1_THR = 0.4


def v1_tag(causal_t: bool) -> str:
    return "v1_causal_t" if causal_t else "v1"


def ensure_stages_v1(fresh: bool, causal_t: bool) -> None:
    from build_dataset_v1 import out_dir
    if fresh or not all((CACHE / f"{p}_train.npz").exists() for p in PANELS):
        subprocess.run(PY + [str(HERE / "prep_cache.py")], check=True)
    if fresh or not (CACHE / "pseudo_windows.parquet").exists():
        subprocess.run(PY + [str(HERE / "windows.py")], check=True)
    if causal_t:
        from restore_t import MD as RMD, TARGETS
        if fresh or not all((RMD / f"{q}_{k}.txt").exists() for q in TARGETS for k in ("A", "B", "C", "all")):
            if fresh:
                for f in RMD.glob("*.txt"):
                    f.unlink()
                (CACHE / "restore_t_samples.parquet").unlink(missing_ok=True)
            subprocess.run(PY + [str(HERE / "restore_t.py"), "fit"], check=True)
    d = out_dir(causal_t)
    missing = [p for p in PANELS if fresh or not (d / f"ds_{p}.parquet").exists()]
    if missing:
        args = PY + [str(HERE / "build_dataset_v1.py"), "--max-ongoing-train", "1200"] + (["--causal-t"] if causal_t else [])
        for p in missing:
            args += ["--panel", p]
        subprocess.run(args, check=True)


def load_v1_dataset(d: Path, cond: int, drop_fold: str | None = None) -> lgb.Dataset:
    """Training rows of the v1 tables, read column-pruned (features + label), as a constructed Dataset.
    Same rows, column order and bins as the original fit(df, feature_cols(df)) call."""
    filters = [("cond", "=", cond), ("vis", "=", 1)] + ([("keep", "=", 1)] if cond == 0 else [])
    if drop_fold is not None:
        filters.append(("fold", "!=", drop_fold))
    names = pq.read_schema(d / f"ds_{PANELS[0]}.parquet").names
    feats = [c for c in names if c not in NON_FEATURES]
    df = pd.concat([pd.read_parquet(d / f"ds_{p}.parquet", columns=feats + ["y"], filters=filters) for p in PANELS],
                   ignore_index=True)
    ds = make_dataset(df, feats)
    del df
    return ds


def get_models_v1(fresh: bool, causal_t: bool) -> dict[str, list[lgb.Booster]]:
    """One booster per condition, seed SEED, 300 rounds, all v1 feature columns (as on 2026-10-25)."""
    from build_dataset_v1 import out_dir
    mdir = MODELS / v1_tag(causal_t)
    mdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, cond in (("onset", 1), ("ongoing", 0)):
        path = mdir / f"{name}.txt"
        if fresh or not path.exists():
            ds = load_v1_dataset(out_dir(causal_t), cond)
            m = fit_dataset(ds, ROUNDS[name])
            m.save_model(str(path))
            del ds
            print(f"fit {name} ({v1_tag(causal_t)}): {m.num_trees()} trees", flush=True)
        out[name] = [lgb.Booster(model_file=str(path))]
    return out


def base_rate_shift(p: np.ndarray, d: float) -> np.ndarray:
    """logit(p') = logit(p) + d."""
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + d)))


def split_rate_source(panel: str, split: str, st, pi_train: float):
    """f(t_idx) -> pi_split: eligible-cell queue rate of this panel in its own split, rows strictly before T,
    shrunk toward pi_train with one day of pseudo-cells (as in pp_eval2.py)."""
    z = np.load(CACHE / f"{panel}_{split}.npz")
    vis = ~np.isnan(z["speed"]) & (z["pct"] >= 75)
    q = vis & (np.nan_to_num(z["speed"], nan=1e9) <= st.vcut)
    cv = np.concatenate([[0], np.cumsum(vis.sum(1))])
    cq = np.concatenate([[0], np.cumsum(q.sum(1))])
    start = {"validation": pd.Timestamp("2031-03-01T00:00:00Z"), "private": pd.Timestamp("2031-04-01T00:00:00Z")}[split]
    off = int((start - pd.Timestamp("2030-06-01T00:00:00Z")) / STEP)
    m = 288 * st.L * 0.5
    def f(t_idx: int) -> float:
        t = t_idx - off                     # rows 0 .. t-1 of the split
        return float((cq[t] + m * pi_train) / (cv[t] + m))
    return f


def sorted_boosters(d: Path, pattern: str) -> list[lgb.Booster]:
    """All model files matching pattern (aug_m3: 3 shift seeds x 3 tree seeds), averaged with equal weight."""
    paths = sorted(d.glob(pattern))
    assert paths, f"no models {d}/{pattern}"
    return [lgb.Booster(model_file=str(p)) for p in paths]


def predict_split_v1(split: str, models: dict[str, list[lgb.Booster]], causal_t: bool, thr: float = V1_THR,
                     alpha: float = 0.0, log: list | None = None, snap_conf: float = 0.0,
                     go_mix: dict[str, float] | None = None, snap_log: list | None = None,
                     on_mix: dict[str, float] | None = None, onset_norm: bool = False,
                     small_thr: float | None = None, small_kmax: int = 5,
                     go_mix_h: list[dict[str, float]] | None = None, split_clim_lam: float | None = None,
                     split_clim_val: bool = False, split_log: list | None = None,
                     split_clim_all: float | None = None) -> pd.DataFrame:
    """snap_conf > 0: onset windows whose v1 expected IoU is below it get their time-of-day lookups moved to the
    nearby time with the most onsets (postprocess.snap_shift); others are untouched.
    go_mix / on_mix: {"v1": w, <variant>: w, ...} probability mixes of lgb_v1 with seed-bagged current-feature
    variants (models/<variant>), e.g. ongoing {"v1": .5, "phys": .5}; onset {"v1": .5, "onphys": .5}."""
    import features_v1 as F1
    from build_dataset import keep_mask as keep_c
    from build_dataset_v1 import keep_mask as keep_v1, onset_rows as onset_v1
    from postprocess import exp_iou, snap_shift
    if go_mix_h:
        go_mix = {v: 1.0 for w in go_mix_h for v in w}     # the variants to load; weights come from go_mix_h
    go_other = {v: (sorted_boosters(MODELS / "aug_m3", "ongoing_S*_s*.txt") if v == "aug_m3" else
                    sorted_boosters(MODELS / "aug_thr", "ongoing_s*.txt") if v == "aug_thr" else load_variant("ongoing", v, False))
                for v in (go_mix or {}) if v != "v1"}
    JIT = ("v1_jit", "v1_phys_jit", "v1_thr", "v1_thron", "v1_thr_notod")  # frozen-v1-feature onset models: time-of-day jitter (v1_jit.py),
                                               # bottleneck threshold features (v1_thr.py)
    on_jit = {v: [lgb.Booster(model_file=str(MODELS / v / f"onset_s{i}.txt")) for i in range(3)]
              for v in (on_mix or {}) if v in JIT}
    THR_ON = ("augthr", "onphysthr")           # current-feature onset models + threshold features (onset_thr2.py)
    on_other = {v: (sorted_boosters(MODELS / f"onset_{v}", "onset_s*.txt") if v in THR_ON else load_variant("onset", v, False))
                for v in (on_mix or {}) if v not in ("v1",) + JIT}
    need_online = any(set(m.feature_name()) & ONLINE_FEATURES for bag in list(go_other.values()) + list(on_other.values())
                      for m in bag)
    rest = None
    if causal_t:
        from restore_t import Restorer
        rest = Restorer("all")
    W = pd.read_parquet(CACHE / "pseudo_windows.parquet")
    out = []
    for panel in PANELS:
        st = F1.Static(panel)
        P = Panel(panel)
        qc = np.load(CACHE / f"{panel}_qclean.npy")
        onset_t = W[(W.panel == panel) & (W.condition == "queue_onset")].t.to_numpy()
        months = sorted(pd.Series(W.month.unique()).tolist())
        clim = F1.Clim(P, qc, months, onset_t)
        if any(v in on_jit for v in ("v1_thr", "v1_thron", "v1_thr_notod")) or "aug_thr" in go_other or any(v in on_other for v in THR_ON):
            from thr_feats import THR_COLS, fit_thresholds, window_features
            thr_all = fit_thresholds(P, np.ones(P.n_t // 288, bool))     # all train days
            if "v1_thron" in on_jit:
                from thr_online import adapted, release_obs
                obs_split = release_obs(panel, split, st, thr_all)
                off_split = int((pd.Timestamp({"validation": "2031-03-01T00:00:00Z", "private": "2031-04-01T00:00:00Z"}[split])
                                 - pd.Timestamp("2030-06-01T00:00:00Z")) / STEP)
        if on_jit:
            from v1_jit import add_extra, onset_static
            ost = onset_static(P, qc, onset_t, months)
        if go_other or on_other:
            st_c = Static(panel)
            clim_c = Clim(P, qc, months, onset_t)
            online_f = online_source(panel, split, st_c) if need_online else None
        if alpha:
            ev = P.visible & P.elig
            pi_train = float((P.q & ev).sum() / ev.sum())
            pi_split = split_rate_source(panel, split, st, pi_train)
        del P, qc
        tmpl = pd.read_csv(REL / "task2" / panel / split / "sample_submission_queue.csv", dtype=str)
        clim_base = clim
        clim_c_base = clim_c if (go_other or on_other) else None
        for w, arr, t_idx in F1.release_windows(panel, split, st):
            clim, clim_c = clim_base, clim_c_base
            if split_clim_all is not None:
                # every climatology input: lam * train + (1 - lam) * this split's complete days before T's day
                # (private: plus all of validation)
                from split_clim import BlendAll, split_clims
                sc_all = split_clims(panel, split, st, t_idx)
                if sc_all is not None:
                    clim = BlendAll(clim_base, sc_all[0], split_clim_all, sc_all[3])
                    if clim_c_base is not None:
                        clim_c = BlendAll(clim_c_base, sc_all[1], split_clim_all, sc_all[3])
                if split_log is not None:
                    split_log.append({"window_id": w.window_id, "days": sc_all[2] if sc_all else 0,
                                      "n_onset": sc_all[3] if sc_all else 0, "adapted": sc_all is not None})
            df = F1.link_features(arr, st, clim, t_idx, rest.fill(arr, st) if rest else None)
            d_logit = 0.0
            if alpha:
                pis = pi_split(t_idx)
                d_logit = alpha * np.log(max(pis, 1e-5) / max(pi_train, 1e-5))
                if log is not None:
                    log.append({"panel": panel, "split": split, "window_id": w.window_id, "pi_split": pis,
                                "pi_train": pi_train, "log_ratio": d_logit / alpha})
            if w.condition == "queue_onset":
                if split_clim_lam is not None and (split == "private" or split_clim_val):
                    from split_clim import Blend, onset_split_clim
                    sc = onset_split_clim(panel, split, st, t_idx)
                    if sc is not None and sc.n_onset >= 5:
                        df = F1.link_features(arr, st, Blend(clim, sc, split_clim_lam), t_idx, rest.fill(arr, st) if rest else None)
                    if split_log is not None:
                        split_log.append({"window_id": w.window_id, "n_onset": sc.n_onset if sc is not None else 0,
                                          "adapted": sc is not None and sc.n_onset >= 5})
                df = onset_v1(df)
                p = base_rate_shift(predict_mean(models["onset"], df), d_logit)
                if snap_conf:
                    e0 = exp_iou(p)
                    sh = snap_shift(clim, t_idx) if e0 < snap_conf else 0
                    if sh:
                        d2 = onset_v1(F1.link_features(arr, st, clim, t_idx + sh, rest.fill(arr, st) if rest else None))
                        p = base_rate_shift(predict_mean(models["onset"], d2), d_logit)
                    if snap_log is not None:
                        snap_log.append({"window_id": w.window_id, "exp_iou": e0, "shift_min": 5 * sh})
                if on_other or on_jit:
                    tot = on_mix.get("v1", 0.0) * p
                    if on_other:
                        dc = onset_rows(link_features(arr, st_c, clim_c, t_idx, online_f(t_idx) if online_f else None), clim_c, t_idx)
                        if any(v in on_other for v in THR_ON):
                            tf = window_features(arr, st.lanes, thr_all)
                            for c in THR_COLS:
                                dc[c] = tf[c].to_numpy()
                        tot = tot + sum(on_mix[v] * predict_mean(bag, dc) for v, bag in on_other.items())
                    if on_jit:
                        dj = add_extra(df.copy(), ost, t_idx)
                        if "v1_thr" in on_jit or "v1_thr_notod" in on_jit:
                            tf = window_features(arr, st.lanes, thr_all)
                            for c in THR_COLS:
                                dj[c] = tf[c].to_numpy()
                        tot = tot + sum(on_mix[v] * predict_mean(bag, dj) for v, bag in on_jit.items() if v != "v1_thron")
                        if "v1_thron" in on_jit:
                            # thresholds adapted with the complete earlier days of this split only
                            dj2 = dj.copy()
                            tf = window_features(arr, st.lanes, adapted(thr_all, obs_split, (t_idx - off_split) // 288, 0))
                            for c in THR_COLS:
                                dj2[c] = tf[c].to_numpy()
                            tot = tot + on_mix["v1_thron"] * predict_mean(on_jit["v1_thron"], dj2)
                    if onset_norm:
                        # rank by the weighted mean of p / max p, keep the number of links lgb_v1 alone would choose
                        parts = {"v1": p} if "v1" in on_mix else {}
                        if on_jit:
                            parts.update({v: predict_mean(bag, dj) for v, bag in on_jit.items()})
                        if on_other:
                            parts.update({v: predict_mean(bag, dc) for v, bag in on_other.items()})
                        sc = sum(on_mix[v] * q / max(q.max(), 1e-9) for v, q in parts.items()) / sum(on_mix.values())
                        kk = int(best_k_set(p).sum())
                        p = np.zeros_like(sc); p[np.argsort(-sc, kind="stable")[:kk]] = 1.0
                    else:
                        p = tot / sum(on_mix.values())
                pred = best_k_set(p)
                cells = pd.DataFrame({"h": H, "link": np.arange(st.L), "queue_pred": pred.astype(int)})
            else:
                df = F1.horizon_rows(df, clim, t_idx)
                p = base_rate_shift(predict_mean(models["ongoing"], df), d_logit) if alpha else predict_mean(models["ongoing"], df)
                p[keep_v1(df) == 0] = 0.0
                if go_other:
                    dc = ongoing_rows(link_features(arr, st_c, clim_c, t_idx, online_f(t_idx) if online_f else None),
                                      clim_c, t_idx)
                    if "aug_thr" in go_other:
                        tf = window_features(arr, st.lanes, thr_all)
                        lk = dc.link.to_numpy().astype(int)
                        for c in THR_COLS:
                            dc[c] = tf[c].to_numpy()[lk]
                    a1 = np.zeros((H, st.L)); a1[df.h.to_numpy() - 1, df.link.to_numpy()] = p
                    parts = {"v1": a1}
                    for v, bag in go_other.items():
                        p2 = predict_mean(bag, dc)
                        p2[keep_c(dc) == 0] = 0.0
                        a2 = np.zeros((H, st.L)); a2[dc.h.to_numpy() - 1, dc.link.to_numpy()] = p2
                        parts[v] = a2
                    hw = go_mix_h or [go_mix] * H        # weights per horizon T+5 .. T+30
                    mix = np.vstack([sum(w.get(v, 0.0) * parts[v][h] for v in parts) / sum(w.values())
                                     for h, w in enumerate(hw)])
                    p = mix[df.h.to_numpy() - 1, df.link.to_numpy()]
                th = thr
                if small_thr is not None:
                    # small queues: 1..5 links queued at T (empty T cells take T-5)
                    sp = np.where(np.isnan(arr["speed"][-1]), arr["speed"][-2], arr["speed"][-1])
                    k_t = int((np.nan_to_num(sp, nan=1e9) <= st.vcut).sum())
                    th = small_thr if 1 <= k_t <= small_kmax else thr
                cells = pd.DataFrame({"h": df.h.to_numpy().astype(int), "link": df.link.to_numpy().astype(int),
                                      "queue_pred": (p >= th).astype(int)})
            T = pd.Timestamp(w.forecast_origin)
            ts = {h: (T + h * STEP).strftime("%Y-%m-%dT%H:%M:%SZ") for h in range(1, H + 1)}
            cells["timestamp"] = cells.h.map(ts)
            cells["link_id"] = st.link_ids[cells.link.to_numpy()]
            cells["window_id"] = w.window_id
            t = tmpl.loc[tmpl.window_id == w.window_id, ["window_id", "timestamp", "link_id"]]
            t = t.merge(cells[["window_id", "timestamp", "link_id", "queue_pred"]], how="left",
                        on=["window_id", "timestamp", "link_id"])
            t["queue_pred"] = t.queue_pred.fillna(0).astype(int)
            out.append(t[["window_id", "timestamp", "link_id", "queue_pred"]])
        print(panel, split, flush=True)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "private", "all"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fresh", action="store_true", help="rebuild caches and refit models")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing output file")
    ap.add_argument("--onset-variant", choices=sorted(VARIANTS) + ["v1"], default="full")
    ap.add_argument("--ongoing-variant", choices=sorted(VARIANTS) + ["v1"], default="full")
    ap.add_argument("--causal-t", action="store_true", help="fill empty T-row cells with restore_t.py (v1 recipe)")
    ap.add_argument("--onset-snap", type=float, default=0.0,
                    help="v1 recipe: onset windows with expected IoU below this get time-of-day lookups snapped (e.g. 0.35)")
    ap.add_argument("--ongoing-phys-w", type=float,
                    help="v1 recipe: ongoing probability w*v1 + (1-w)*lgb_phys (use with --thr)")
    ap.add_argument("--small-queue-thr", type=float,
                    help="v1 recipe: ongoing threshold for windows with 1..5 links queued at T (T-5 where T is empty)")
    ap.add_argument("--onset-split-clim", type=float, help="v1 recipe: onset climatology lam * train + (1 - lam) * the "
                    "split's own earlier days (private: plus all of validation); validation windows only with --split-clim-val")
    ap.add_argument("--split-clim-val", action="store_true")
    ap.add_argument("--split-clim-all", type=float, help="v1 recipe: every climatology input (both climatologies, onset and "
                    "ongoing, validation and private) = lam * train + (1 - lam) * the split's complete earlier days")
    ap.add_argument("--ongoing-mix-h", help="v1 recipe: per-horizon ongoing mix, e.g. '1-2=v1:.1,phys:.3,aug:.2,onphys:.4;3-6=...'")
    ap.add_argument("--small-queue-kmax", type=int, default=5, help="small-queue windows: 1..kmax links queued at T")
    ap.add_argument("--onset-norm", action="store_true",
                    help="v1 recipe with --onset-mix: rank links by the mean of p / max p, keep lgb_v1's k")
    ap.add_argument("--base-rate-alpha", type=float, default=0.0,
                    help="v1 recipe: logit shift alpha*log(pi_split/pi_train), pi_split from the split's own rows before T")
    ap.add_argument("--onset-mix", help="probability ensemble, e.g. aug:0.5,onphys:0.5 (overrides --onset-variant)")
    ap.add_argument("--ongoing-mix", help="e.g. aug:0.25,phys:0.75 (overrides --ongoing-variant)")
    ap.add_argument("--thr", type=float, help="ongoing threshold (default: per variant, 0.35 for mixes)")
    args = ap.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists; candidates are recorded by md5 at merge time - use a new name or --overwrite")
    splits = ["validation", "private"] if args.split == "all" else [args.split]
    if "v1" in (args.onset_variant, args.ongoing_variant):
        # lgb_v1 recipe: frozen features, one seed, threshold 0.4 (both conditions)
        if (args.onset_variant, args.ongoing_variant) != ("v1", "v1"):
            raise SystemExit("v1 is a separate pipeline: use --onset-variant v1 --ongoing-variant v1 "
                             "(mixes with other variants: --onset-mix v1:.5,onphys:.5 / --ongoing-mix v1:.5,phys:.5)")
        for mix in (args.onset_mix, args.ongoing_mix):
            bad = sorted(set(parse_mix(mix)) - set(VARIANTS) - {"v1", "v1_jit", "v1_phys_jit", "v1_thr", "v1_thron", "v1_thr_notod", "aug_m3", "aug_thr", "augthr", "onphysthr"}) if mix else []
            if bad:
                raise SystemExit(f"unknown variant(s): {bad}")
        ensure_stages_v1(args.fresh, args.causal_t)
        models = get_models_v1(args.fresh, args.causal_t)
        thr = args.thr if args.thr is not None else V1_THR
        log: list = []
        snap_log: list = []
        split_log: list = []
        go_mix = parse_mix(args.ongoing_mix) if args.ongoing_mix else (
            {"v1": args.ongoing_phys_w, "phys": 1 - args.ongoing_phys_w} if args.ongoing_phys_w is not None else None)
        on_mix = parse_mix(args.onset_mix) if args.onset_mix else None
        sub = pd.concat([predict_split_v1(s, models, args.causal_t, thr, args.base_rate_alpha, log, args.onset_snap,
                                          go_mix, snap_log, on_mix, args.onset_norm, args.small_queue_thr,
                                          args.small_queue_kmax, parse_mix_h(args.ongoing_mix_h), args.onset_split_clim,
                                          args.split_clim_val, split_log, args.split_clim_all)
                         for s in splits], ignore_index=True)
        write_checked(sub, splits, args.out)
        if log:
            pd.DataFrame(log).to_csv(args.out.parent / "base_rate_log.csv", index=False)
        if snap_log:
            pd.DataFrame(snap_log).to_csv(args.out.parent / "onset_snap_log.csv", index=False)
        if split_log:
            pd.DataFrame(split_log).to_csv(args.out.parent / "onset_split_clim_log.csv", index=False)
        return
    if args.causal_t:
        raise SystemExit("--causal-t is implemented for the v1 recipe only")
    ensure_stages(args.fresh)
    mixes = {"onset": parse_mix(args.onset_mix) if args.onset_mix else {args.onset_variant: 1.0},
             "ongoing": parse_mix(args.ongoing_mix) if args.ongoing_mix else {args.ongoing_variant: 1.0}}
    for mix in mixes.values():
        bad = sorted(set(mix) - set(VARIANTS))
        if bad:
            raise SystemExit(f"unknown variant(s): {bad}")
    models = get_models(args.fresh, mixes)
    thr = args.thr if args.thr is not None else (0.35 if args.ongoing_mix else THR_BY_VARIANT[args.ongoing_variant])
    sub = pd.concat([predict_split(s, models, thr) for s in splits], ignore_index=True)
    write_checked(sub, splits, args.out)


def write_checked(sub: pd.DataFrame, splits: list[str], out: Path) -> None:
    # row set must equal the template exactly
    tmpl = pd.concat([pd.read_csv(REL / "task2" / p / s / "sample_submission_queue.csv", dtype=str)
                      for s in splits for p in PANELS], ignore_index=True)
    assert len(sub) == len(tmpl) and set(map(tuple, sub[["window_id", "timestamp", "link_id"]].values)) == \
        set(map(tuple, tmpl[["window_id", "timestamp", "link_id"]].values)), "row set differs from template"
    assert sub.queue_pred.isin([0, 1]).all()
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"wrote {len(sub):,} rows, positives {int(sub.queue_pred.sum()):,} -> {out}")


if __name__ == "__main__":
    main()
