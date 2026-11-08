"""후보 파일 검증: 템플릿과 행이 정확히 같은지, 값이 유한·비음수인지, 후보 간 차이."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from odme_lib import PANELS, REL

root = Path(__file__).resolve().parent
tpl = pd.concat([pd.read_csv(REL / "task4" / p / s / "sample_submission_path_flow.csv", dtype=str)
                 for s in ["validation", "private"] for p in PANELS], ignore_index=True)
files = {d.name: d / "odme_submission.csv" for d in sorted((root / "candidates").iterdir()) if (d / "odme_submission.csv").exists()}
files["baseline_repro"] = root / "reference" / "baseline_repro" / "odme_submission.csv"
F = {}
for n, p in files.items():
    x = pd.read_csv(p, dtype={"panel": str, "departure_time": str, "path_id": str, "origin_zone": str, "destination_zone": str})
    key = ["panel", "departure_time", "path_id", "origin_zone", "destination_zone"]
    assert x[key].equals(tpl[key]), n
    f = x.path_flow.to_numpy(float)
    assert np.isfinite(f).all() and (f >= 0).all(), n
    F[n] = f
    print(n, len(x), "OK", "Σf", round(f.sum()))
b = pd.concat([pd.read_csv(REL / "task4" / p / s / "synthetic_weak_prior.csv") for s in ["validation", "private"] for p in PANELS]).path_flow.to_numpy()
names = list(F)
print("정규화 L1 거리 Σ|x-y|/Σy (행=x, 열=y):")
print(pd.DataFrame([[np.abs(F[a] - F[c]).sum() / F[c].sum() for c in names] for a in names], index=names, columns=names).round(3).to_string())
print("대 사전값 Σ|f-b|/Σb:", {n: round(np.abs(F[n] - b).sum() / b.sum(), 3) for n in names})
