"""Fold models (3-month folds, seed SEED) for an out-of-fold check in a work folder: v1 onset/ongoing (monthly_v1) and
the current-feature ongoing variants phys / aug / onphys (monthly_official)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monthly_official
import monthly_v1

if __name__ == "__main__":
    monthly_v1.fold_models(False)
    for var in ("phys", "aug", "onphys"):
        monthly_official.VAR = var
        monthly_official.fold_models()
        print("folds", var, flush=True)
