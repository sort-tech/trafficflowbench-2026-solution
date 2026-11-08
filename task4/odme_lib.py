"""Task 4 (경로 교통량·OD 추정) 공통 함수. 데이터는 읽기만 한다."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

# 공개 데이터 루트. 환경변수 TFB_REL로 바꾸고, 없으면 이 폴더의 상위 data/kaggle_public.
REL = Path(os.environ.get("TFB_REL", Path(__file__).resolve().parent.parent / "data" / "kaggle_public"))
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S",
          "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]
SPLITS = ["train", "validation", "private"]
REG_LAMBDA = 0.05


@dataclass
class Panel:
    name: str
    path_ids: list
    link_ids: list
    A: np.ndarray            # (links, paths) 0/1 incidence
    paths: pd.DataFrame      # path_id, origin_zone, destination_zone
    o: np.ndarray            # origin zone index per path
    d: np.ndarray            # destination zone index per path
    template: dict           # split -> template frame (row order of submission)

    def prior(self, split: str, rel: Path = REL) -> np.ndarray:
        f = pd.read_csv(rel / "task4" / self.name / split / "synthetic_weak_prior.csv", dtype={"path_id": str})
        return f.set_index("path_id").reindex(self.path_ids).path_flow.to_numpy(float)

    def counts(self, split: str, rel: Path = REL):
        """(measured mask over links, count vector over measured links)."""
        f = pd.read_csv(rel / "task4" / self.name / split / "synthetic_link_counts.csv", dtype={"link_id": str})
        s = f.set_index("link_id")["count"].reindex(self.link_ids)
        m = ~s.isna().to_numpy()
        return m, s.to_numpy(float)[m]


def load_panel(name: str, rel: Path = REL) -> Panel:
    net = rel / "corridors" / name / "network"
    paths = pd.read_csv(net / "path_set.csv", dtype=str)
    inc = pd.read_csv(net / "path_link_incidence.csv", dtype=str)
    path_ids = paths.path_id.tolist()
    link_ids = inc.link_id.drop_duplicates().tolist()   # 공식 코드와 같은 순서
    pi = {x: i for i, x in enumerate(path_ids)}
    li = {x: i for i, x in enumerate(link_ids)}
    A = np.zeros((len(link_ids), len(path_ids)))
    A[inc.link_id.map(li).to_numpy(), inc.path_id.map(pi).to_numpy()] = 1.0
    o = paths.origin_zone.str[1:].astype(int).to_numpy()
    d = paths.destination_zone.str[1:].astype(int).to_numpy()
    tpl = {s: pd.read_csv(rel / "task4" / name / s / "sample_submission_path_flow.csv", dtype=str) for s in SPLITS}
    return Panel(name, path_ids, link_ids, A, paths, o, d, tpl)


# ---------------------------------------------------------------- 점수
def s_link(P: Panel, split: str, f: np.ndarray) -> float:
    m, c = P.counts(split)
    return max(0.0, 1.0 - np.abs(P.A[m] @ f - c).sum() / c.sum())


def unique_rows(A: np.ndarray, c: np.ndarray):
    """같은 행(같은 경로 집합)을 한 제약으로 묶는다. 같은 행의 교통량은 평균."""
    key = {}
    for i in range(A.shape[0]):
        key.setdefault(A[i].tobytes(), []).append(i)
    idx = [v[0] for v in key.values()]
    cc = np.array([c[v].mean() for v in key.values()])
    return A[idx], cc


# ---------------------------------------------------------------- 풀이기
def solve_nnls(A: np.ndarray, c: np.ndarray, b: np.ndarray, lam: float = REG_LAMBDA) -> np.ndarray:
    """공식 기준 모델: min ||Af-c||^2 + lam ||f-b||^2, f>=0."""
    aa = np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])])
    bb = np.concatenate([c, np.sqrt(lam) * b])
    f, _ = nnls(aa, bb)
    return f


def solve_kl(A: np.ndarray, c: np.ndarray, b: np.ndarray, iters: int = 200, tol: float = 1e-10) -> np.ndarray:
    """min KL(f || b) s.t. A f = c. 쌍대 문제를 뉴턴법으로 푼다: f = b * exp(A^T lam)."""
    M, cc = unique_rows(A, c)
    # 선형 종속 행은 쌍대 헤시안이 특이해지므로 약한 릿지로 처리
    lam = np.zeros(M.shape[0])
    b = np.maximum(b, 1e-12)

    def obj(l):
        e = M.T @ l
        return float(b @ np.exp(np.clip(e, -50, 50)) - cc @ l)

    cur = obj(lam)
    for _ in range(iters):
        f = b * np.exp(np.clip(M.T @ lam, -50, 50))
        g = M @ f - cc
        if np.abs(g).max() <= tol * max(1.0, cc.max()):
            break
        H = (M * f) @ M.T
        H[np.diag_indices_from(H)] += 1e-9 * np.trace(H) / len(H)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-8:
            new = obj(lam - t * step)
            if new < cur:
                break
            t *= 0.5
        lam = lam - t * step
        cur = new
    return b * np.exp(np.clip(M.T @ lam, -50, 50))


def mart(A: np.ndarray, c: np.ndarray, b: np.ndarray, sweeps: int = 2000, tol: float = 1e-8) -> np.ndarray:
    """곱셈형 대수 재구성(MART). solve_kl과 같은 극한이며 교차 검증용."""
    M, cc = unique_rows(A, c)
    f = np.maximum(b, 1e-12).copy()
    for _ in range(sweeps):
        worst = 0.0
        for r in range(M.shape[0]):
            msk = M[r] > 0
            cur = f[msk].sum()
            ratio = cc[r] / max(cur, 1e-12)
            f[msk] *= ratio
            worst = max(worst, abs(ratio - 1))
        if worst < tol:
            break
    return f


# ---------------------------------------------------------------- 사전값 잡음 제거
def design(P: Panel, deg: int = 4) -> np.ndarray:
    """로그 사전값의 구조: 출발지 효과 + 도착지 효과 + 거리(구간 수)의 다항식."""
    n = len(P.path_ids)
    no, nd = P.o.max() + 1, P.d.max() + 1
    X = np.zeros((n, no + nd + deg))
    X[np.arange(n), P.o] = 1
    X[np.arange(n), no + P.d] = 1
    g = np.log(P.d - P.o)
    for k in range(deg):
        X[:, no + nd + k] = g ** (k + 1)
    return X


def pooled_structure(P: Panel, priors: list, deg: int = 4) -> np.ndarray:
    """세 분할 사전값의 로그 평균에 구조 모형을 맞춘 적합값(로그)."""
    X = design(P, deg)
    ybar = np.mean([np.log(b) for b in priors], axis=0)
    coef = np.linalg.lstsq(X, ybar, rcond=None)[0]
    return X @ coef


def solve_ridge_clip(A: np.ndarray, c: np.ndarray, b: np.ndarray, lam: float = REG_LAMBDA, rounds: int = 6) -> np.ndarray:
    """NNLS(lam)의 빠른 근사: 닫힌 형태 풀이 후 음수를 0으로 고정하고 반복. 교차 검증용."""
    M, cc = unique_rows(A, c)
    free = np.ones(M.shape[1], bool)
    f = b.copy()
    for _ in range(rounds):
        Mf = M[:, free]
        r = cc - Mf @ b[free]
        f = np.zeros_like(b)
        f[free] = b[free] + Mf.T @ np.linalg.solve(Mf @ Mf.T + lam * np.eye(len(M)), r)
        neg = free & (f < 0)
        if not neg.any():
            break
        free &= ~neg
        f[~free] = 0
    return np.maximum(f, 0)
