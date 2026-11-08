#!/bin/bash
# Create the two Python 3.12 environments used by final/run_all.sh: .venv (main) and .nn_venv (torch, CPU).
#   PYTHON312=/path/to/python3.12 ./setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON312:-python3.12}
"$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
"$PY" -m venv .nn_venv
.nn_venv/bin/python -m pip install -q --upgrade pip
if [ "$(uname)" = Linux ]; then   # CPU wheel; the network step never uses a GPU
  .nn_venv/bin/python -m pip install -q -r requirements_nn.txt --extra-index-url https://download.pytorch.org/whl/cpu
else
  .nn_venv/bin/python -m pip install -q -r requirements_nn.txt
fi
.venv/bin/python -c "import numpy, pandas, lightgbm, scipy, polars; print('main ok', numpy.__version__, pandas.__version__, lightgbm.__version__)"
.nn_venv/bin/python -c "import torch; print('nn ok', torch.__version__)"
