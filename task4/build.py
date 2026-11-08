"""Task 4 최종 진입 스크립트. 기본 동작이 최종 제출본(nnls_own) 생성이다.

    python -I build.py --out final                 # final/nnls_own/odme_submission.csv (validation+private)
    python -I build.py --split validation --out OUTDIR [--cand name,name] [--jobs 3]   # 분할·후보 지정

최종 방법(nnls_own): min ||A f - c||^2 + 0.05 ||f - b||^2, f >= 0 (scipy.optimize.nnls).
  A = 공개 경로-링크 연결표, c = 해당 분할의 공개 링크 교통량, b = 해당 분할의 공개 사전값. λ=0.05 고정.
  외부 자료·학습·무작위 요소 없음. 시드 필요 없음(결정적 활성 집합 풀이). 같은 입력이면 같은 출력.
의존성: python 3.12, numpy 2.x, pandas 3.x, scipy. 데이터는 odme_lib.REL 경로(kaggle_public).
실행 시간: 약 6분(프로세스 3개, 한 프로세스당 스레드 1개). 파일 하나에 두 분할 행을 템플릿 순서 그대로 담는다.
다른 후보(nnls_w75, kl_own, blend_nk)는 LB에서 열세여서 실험 기록으로만 남긴다.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from odme_lib import (PANELS, REL, SPLITS, load_panel, pooled_structure, s_link, solve_kl,  # noqa: E402
                      solve_nnls)

W_DENOISE = 0.75   # 사전값 로그 잡음을 25% 줄이는 정도 (보류 컷 교차 검증에서 0.75가 최소)


def seed_denoised(P, split, w=W_DENOISE):
    priors = {s: P.prior(s) for s in SPLITS}
    fit = pooled_structure(P, [priors[s] for s in SPLITS])
    y = np.log(priors[split])
    return np.exp(fit + w * (y - fit))


def f_baseline(P, split):
    """공식 baseline_submission: train 교통량으로 풀고 사전값은 해당 분할 것 (재현용, 제출 후보 아님)."""
    m, c = P.counts(split)
    mt, ct = P.counts("train")
    full_t = np.zeros(len(P.link_ids))
    full_t[mt] = ct
    return solve_nnls(P.A[m], full_t[m], P.prior(split))


def f_nnls_own(P, split):
    m, c = P.counts(split)
    return solve_nnls(P.A[m], c, P.prior(split))


def f_nnls_w75(P, split):
    m, c = P.counts(split)
    return solve_nnls(P.A[m], c, seed_denoised(P, split))


def f_kl_own(P, split):
    m, c = P.counts(split)
    return solve_kl(P.A[m], c, P.prior(split))


def f_blend(P, split):
    m, c = P.counts(split)
    return 0.5 * f_nnls_w75(P, split) + 0.5 * solve_kl(P.A[m], c, seed_denoised(P, split))


CANDS = {
    "baseline_repro": f_baseline,
    "nnls_own": f_nnls_own,
    "nnls_w75": f_nnls_w75,
    "kl_own": f_kl_own,
    "blend_nk": f_blend,
}


def _work(args):
    panel, split, names = args
    P = load_panel(panel)
    b = P.prior(split)
    out = {}
    for n in names:
        f = np.maximum(CANDS[n](P, split), 0.0)
        assert np.isfinite(f).all()
        out[n] = (f, s_link(P, split, f), float(np.abs(f - b).sum() / b.sum()), float(f.sum() / b.sum()))
    return panel, out


def run(split, names, jobs):
    with Pool(jobs) as pool:
        res = dict(pool.map(_work, [(p, split, names) for p in PANELS], chunksize=1))
    frames, stats = {n: [] for n in names}, []
    for p in PANELS:
        tpl = pd.read_csv(REL / "task4" / p / split / "sample_submission_path_flow.csv", dtype=str)
        P = load_panel(p)
        for n in names:
            f, sl, dev, tot = res[p][n]
            x = tpl.copy()
            assert x.path_id.tolist() == P.path_ids, "템플릿 경로 순서가 다르다"
            x["path_flow"] = f
            frames[n].append(x)
            stats.append((split, n, p, sl, dev, tot))
    return {n: pd.concat(v, ignore_index=True) for n, v in frames.items()}, pd.DataFrame(
        stats, columns=["split", "cand", "panel", "S_link", "D_over_sumb", "sumf_over_sumb"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "private", "both"], default="both")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cand", default="nnls_own")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    names = [n for n in CANDS if n != "baseline_repro"] if a.cand == "all" else a.cand.split(",")
    splits = ["validation", "private"] if a.split == "both" else [a.split]
    per, allstats = {n: [] for n in names}, []
    for s in splits:
        fr, st = run(s, names, a.jobs)
        for n in names:
            per[n].append(fr[n])
        allstats.append(st)
    stats = pd.concat(allstats, ignore_index=True)
    a.out.mkdir(parents=True, exist_ok=True)
    for n in names:
        d = a.out / n
        d.mkdir(parents=True, exist_ok=True)
        x = pd.concat(per[n], ignore_index=True)
        x.to_csv(d / ("odme_submission.csv" if a.split == "both" else f"odme_{a.split}.csv"), index=False)
    stats.to_csv(a.out / f"stats_{a.split}.csv", index=False)
    print(stats.groupby(["split", "cand"])[["S_link", "D_over_sumb", "sumf_over_sumb"]].mean().round(4))


if __name__ == "__main__":
    main()
