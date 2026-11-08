"""Build a Task 1 state submission (Task 3 is scored on it) from scratch.

    python -I build.py --split validation --out candidates/<name>/state_submission.csv
    python -I build.py --split private    --out ...
    python -I build.py --train-only

Steps
  1. cache dense arrays of the release (common.load)
  2. if the models in --models are missing (or --retrain): build the training
     rows from every train day (actual masks, simulated Task 2 gaps) and fit the
     LightGBM residual models with fixed seeds and round counts
  3. features for every template cell of the split, predict, reconcile density,
     write rows in template order

Deterministic: fixed seeds, LightGBM deterministic=True, 3 threads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
import features as F  # noqa: E402
import pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", help="validation, private, or validation,private (one file)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--config", default=pipeline.CONFIG["name"], choices=sorted(pipeline.CONFIGS))
    ap.add_argument("--models", type=Path, help="model directory (default models/<config>)")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--panel", action="append")
    ap.add_argument("--stage2", action="store_true",
                    help="two-stage model (stage2.py): parity stage-1 models + residual stage-2 models")
    ap.add_argument("--selfsup", action="store_true",
                    help="with --stage2: add a per-split correction learned from the split's own visible cells")
    ap.add_argument("--gap-tag", help="Task 2 gap specialist model tag(s) (models/<tag>_y_*.txt), blended inside "
                                      "the gaps; several comma-separated tags are averaged")
    ap.add_argument("--gap-weight", type=float, default=0.5)
    ap.add_argument("--s2-set", default="v4", choices=["v4", "ramp"],
                    help="stage-2 chain: v4 (ds7, s1_ds7, gbm8s2*) or ramp (FEAT=7 ramp + conservation: ds13, s1_ds13, gbm8s2r*)")
    ap.add_argument("--s2-tags", default="gbm8s2",
                    help="comma-separated stage-2 model tags; residuals of several seeds are averaged")
    a = ap.parse_args()
    if a.stage2:
        return main_stage2(a)
    cfg = pipeline.CONFIGS[a.config]
    mdir = a.models or HERE / "models" / cfg["name"]
    t0 = time.time()
    log = {"config": cfg, "start": time.strftime("%Y-%m-%d %H:%M:%S")}
    if a.retrain or not pipeline.models_exist(mdir):
        t = time.time()
        pipeline.train_final(mdir, cfg)
        log["train_seconds"] = round(time.time() - t, 1)
    if a.train_only:
        print(f"trained in {time.time()-t0:.0f}s")
        return
    assert a.split and a.out, "--split and --out are required"
    models = pipeline.load_models(mdir)
    panels = a.panel or common.PANELS
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    first = True
    n = 0
    for split in a.split.split(","):
        assert split in ("validation", "private", "train"), split
        for p in panels:
            t = time.time()
            tpl = pipeline.predict_split(p, split, models, cfg)
            tpl.to_csv(tmp, mode="w" if first else "a", header=first, index=False, float_format="%.4f")
            first = False
            n += len(tpl)
            print(f"{split} {p}: {len(tpl):,} rows {time.time()-t:.0f}s", flush=True)
    os.replace(tmp, a.out)
    log["rows"] = n
    log["seconds"] = round(time.time() - t0, 1)
    log["split"] = a.split
    (a.out.parent / f"build_{a.split.replace(',', '_')}.json").write_text(json.dumps(log, indent=1))
    print(f"wrote {n:,} rows to {a.out} in {time.time()-t0:.0f}s")


# Round counts fixed at the early-stopping optimum found on GB10 (16 threads), so a
# rerun with another thread count does not stop at a different round.
S2_ENV = {"GAPS": "win", "FEAT": "4", "DS_DIR": "ds7", "N_PER_REGIME": "200000", "GAP_OS": "4",
          "LR": "0.05", "ROUNDS": "4000", "FIXED_ROUNDS": "y_v:3983,y_q:3999,y_k:4000"}
SEED_ROUNDS = {"11": "y_v:4000,y_q:3996,y_k:3999", "23": "y_v:4000,y_q:3999,y_k:4000"}


GAP_ENV = {"GAPS": "win", "FEAT": "4", "DS_DIR": "ds9", "N_PER_REGIME": "200000", "GAP_OS": "8",
           "EXTRA_WIN": "1", "GAP_ONLY": "1", "LR": "0.05", "ROUNDS": "3000",
           "FIXED_ROUNDS": "y_v:3000,y_q:2996,y_k:3000"}
# Ramp-flow gap specialists (10/30): ds9 rows with the v5 window and v6 ramp features
# (ds11, eval rows inside the gaps only), same params and rounds as gbm9gap.
# gap8 = v4 + ramps, gap8b = v4 + queue polygon / triangles + ramps; _s<seed> = seed variant.
RAMP_GAP_ENV = dict(GAP_ENV, FEAT="6", DS_DIR="ds11", EVAL_GAP_ONLY="1")
# Stage-2 chain with the v7 ramp + conservation features on every masked cell (10/30), window
# columns (g5-g7) left out. Round counts: the v4 chain stopped at its 4000 cap, so 4000.
S2R_ENV = dict(S2_ENV, FEAT="7", DS_DIR="ds13", DROP="g5_,g6_,g7_", FIXED_ROUNDS="y_v:4000,y_q:4000,y_k:4000")
S2_SETS = {"v4": {"env": S2_ENV, "ds": "ds7", "base": "gbm8s2", "cfg": "gbm_v7", "seed_rounds": SEED_ROUNDS},
           "ramp": {"env": S2R_ENV, "ds": "ds13", "base": "gbm8s2r", "cfg": "gbm_v7r", "seed_rounds": {}}}
RAMP_GAP_DROP = {"gap8": "g5_,g6_,g7_", "gap8b": "g5_corr,g5_dx,g5_plen,g5_pstate,g7_"}


def train_extra(tags, gap_tag, force: bool = False, S=None) -> None:
    """Seed variants of the stage-2 model (tag <base>_s<seed>) and the Task 2 gap specialist."""
    import shutil
    import subprocess
    S = S or S2_SETS["v4"]
    py = [sys.executable, "-u"]
    for tag in tags:
        if tag == S["base"] or (not force and all((HERE / "models" / f"{tag}_{t}.txt").exists() for t in pipeline.TARGETS)):
            continue
        seed = tag.rsplit("_s", 1)[1]
        env = dict(os.environ, **S["env"])
        env.update(TFB_SEED=seed, FIXED_ROUNDS=S["seed_rounds"].get(seed, S["env"]["FIXED_ROUNDS"]))
        subprocess.run(py + [str(HERE / "stage2.py"), "fit2", S["ds"], tag], env=env, check=True, cwd=HERE)
    wp = [p for p in common.PANELS if not p.startswith("D12_I405")]  # no Task 2 windows there
    built = set()
    for gt in (gap_tag.split(",") if gap_tag else []):
        if not force and all((HERE / "models" / f"{gt}_{t}.txt").exists() for t in pipeline.TARGETS):
            continue
        base, _, seed = gt.partition("_s")
        if base in RAMP_GAP_DROP:
            env = dict(os.environ, **RAMP_GAP_ENV, DROP=RAMP_GAP_DROP[base])
            if seed:
                env["TFB_SEED"] = seed
            if "ds11" not in built:
                subprocess.run(py + [str(HERE / "make_dataset.py"), ",".join(wp)], env=env, check=True, cwd=HERE)
                built.add("ds11")
            subprocess.run(py + [str(HERE / "train_gbm.py"), gt, "ds11"], env=env, check=True, cwd=HERE)
            continue
        env = dict(os.environ, **GAP_ENV)
        subprocess.run(py + [str(HERE / "make_dataset.py"), ",".join(wp)], env=env, check=True, cwd=HERE)
        for p in common.PANELS:
            if p not in wp:
                shutil.copy(HERE / "ds7" / f"eval_{p}.parquet", HERE / "ds9" / f"eval_{p}.parquet")
        subprocess.run(py + [str(HERE / "train_gbm.py"), gt, "ds9"], env=env, check=True, cwd=HERE)


def train_stage2(force: bool = False, S=None) -> None:
    """The stage-2 chain: dataset (ds7) -> parity stage-1 models -> filled-field rows -> stage-2 models.

    Without force, a step whose outputs already exist is skipped, so an interrupted run resumes.
    """
    import subprocess
    S = S or S2_SETS["v4"]
    env = dict(os.environ, **S["env"])
    py = [sys.executable, "-u"]
    run = lambda *args: subprocess.run(py + [str(HERE / args[0]), *args[1:]], env=env, check=True, cwd=HERE)
    dsn = S["ds"]
    ds = HERE / dsn
    for sub in ("models", "preds", "logs", dsn):
        (HERE / sub).mkdir(parents=True, exist_ok=True)
    have = lambda paths: (not force) and all(x.exists() for x in paths)
    if not have([ds / f"{k}_{p}.parquet" for p in common.PANELS for k in ("fit", "eval")]):
        run("make_dataset.py", ",".join(common.PANELS))
        force = True  # everything downstream depends on the dataset
    if not have([HERE / "models" / f"s1_{dsn}" / f"p{par}_{t}.txt" for par in (0, 1) for t in pipeline.TARGETS]):
        run("stage2.py", "fit1", dsn)
        force = True
    for p in common.PANELS:
        if not have([ds / f"s2{k}_{p}.parquet" for k in ("fit", "eval")]):
            run("stage2.py", "rows", dsn, p)
            force = True
    if not have([HERE / "models" / f"{S['base']}_{t}.txt" for t in pipeline.TARGETS]):
        run("stage2.py", "fit2", dsn, S["base"])


def main_stage2(a) -> None:
    import lightgbm as lgb
    S = S2_SETS[a.s2_set]
    cfg = pipeline.CONFIGS[S["cfg"]]  # gbm_v7: feature version 4, reconcile beta 0.2 / 0.8; gbm_v7r: version 7
    s1dir = HERE / "models" / f"s1_{S['ds']}"
    tags = a.s2_tags.split(",")
    s2file = lambda t, tag=S["base"]: HERE / "models" / f"{tag}_{t}.txt"
    t0 = time.time()
    log = {"model": f"stage2 (s1_{S['ds']} + {a.s2_tags})" + (f" + selfsup x{pipeline.SS_FOLDS}" if a.selfsup else ""), "env": S["env"], "reconcile": cfg["reconcile"],
           "start": time.strftime("%Y-%m-%d %H:%M:%S")}
    have = all((s1dir / f"p{par}_{t}.txt").exists() for par in (0, 1) for t in pipeline.TARGETS) and \
        all(s2file(t).exists() for t in pipeline.TARGETS)
    t = time.time()
    if a.retrain or not have:
        train_stage2(force=a.retrain, S=S)
    train_extra(tags, a.gap_tag, force=a.retrain, S=S)
    log["train_seconds"] = round(time.time() - t, 1)
    if a.train_only:
        return
    s1 = {par: {t: lgb.Booster(model_file=str(s1dir / f"p{par}_{t}.txt")) for t in pipeline.TARGETS} for par in (0, 1)}
    missing = [tg for tg in tags if not all(s2file(t, tg).exists() for t in pipeline.TARGETS)]
    assert not missing, f"stage-2 models missing for {missing}; train them with stage2.py fit2 {S['ds']} <tag> (TFB_SEED=...)"
    s2 = [{t: lgb.Booster(model_file=str(s2file(t, tg))) for t in pipeline.TARGETS} for tg in tags]
    gapm = None
    if a.gap_tag:
        gapm = [{t: lgb.Booster(model_file=str(HERE / "models" / f"{gt}_{t}.txt")) for t in pipeline.TARGETS}
                for gt in a.gap_tag.split(",")]
        gapm = gapm[0] if len(gapm) == 1 else gapm  # several tags: predictions averaged
        log["gap"] = {"tag": a.gap_tag, "weight": a.gap_weight}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    first, n = True, 0
    for split in a.split.split(","):
        if a.selfsup:
            t = time.time()
            for tpl in pipeline.selfsup_split(split, s1, s2, cfg, a.panel or common.PANELS, gap=gapm, gap_w=a.gap_weight):
                tpl.to_csv(tmp, mode="w" if first else "a", header=first, index=False, float_format="%.4f")
                first = False
                n += len(tpl)
            print(f"{split}: self-supervised correction {time.time()-t:.0f}s", flush=True)
            continue
        for p in (a.panel or common.PANELS):
            t = time.time()
            tpl = pipeline.predict_split_stage2(p, split, s1, s2, cfg, gapm, a.gap_weight)
            tpl.to_csv(tmp, mode="w" if first else "a", header=first, index=False, float_format="%.4f")
            first = False
            n += len(tpl)
            print(f"{split} {p}: {len(tpl):,} rows {time.time()-t:.0f}s", flush=True)
    os.replace(tmp, a.out)
    log.update(rows=n, seconds=round(time.time() - t0, 1), split=a.split)
    (a.out.parent / f"build_{a.split.replace(',', '_')}.json").write_text(json.dumps(log, indent=1))
    print(f"wrote {n:,} rows to {a.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
