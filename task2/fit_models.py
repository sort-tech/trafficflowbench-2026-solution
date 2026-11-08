"""Fit one model family in its own process (keeps the peak memory of a from-scratch run per family).

  fit_models.py v1          models/v1/{onset,ongoing}.txt          (lgb_v1 recipe)
  fit_models.py <variant>   models/<variant>/ongoing_s{0,1,2}.txt  (e.g. phys, aug; current features)
Models that already exist are left alone; build.py then only predicts.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import get_models_v1, load_variant

if __name__ == "__main__":
    name = sys.argv[1]
    if name == "v1":
        get_models_v1(False, False)
    else:
        load_variant(sys.argv[2] if len(sys.argv) > 2 else "ongoing", name, False)
