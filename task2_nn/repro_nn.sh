#!/bin/bash
# Task 2 network candidates from scratch: data -> 3 seeds (full train) -> mix with i5 LightGBM -> candidates.
# From shipped weights instead of training (the run_all path): repro_final.sh. Training reproduces the cpu_r* weights
# bit for bit only on GB10 (aarch64, torch 2.14.1, CPU, THREADS=2); on an Apple M4 the loss already differs at step 500.
#
# Needs on one machine: the task2 LightGBM environment with task2/cache and task2/models (PY_LGB, e.g. KAGGLE/.venv)
# and a PyTorch environment (PY_NN). If they are on different machines, run step 2 where PY_NN is, with data/ copied
# there, and copy runs/<TAG>_r*/release.pkl back.
#
#   PY_LGB=../.venv/bin/python PY_NN=/path/to/torch/python ./repro_nn.sh <TAG>    (default DEVICE=cpu, THREADS=2)
#
# DEVICE=cpu adds --deterministic (torch.use_deterministic_algorithms, fixed seeds): repeated runs give identical model
# files on the same machine and thread count (1, 4 and 8 threads each give a different md5 from 2: keep THREADS=2
# to reproduce cpu3_*). DEVICE=cuda is the exploration setting used for nn3_* (GB10).
# OOF=1 also fits the 3-month fold models and prints the out-of-fold mix table (combine_eval.py).
#
# Times measured on GB10 (ARM 20 cores, NVIDIA GB10) and the Mac (Apple M-series, 10 cores):
#   1. prep.py + prep_extra.py (Mac)                       about 2 min, data/ 1.7 GB
#   2. train.py --fold all, one seed, 12,000 steps
#        cuda, alone on the GPU                            about 12 min (27-36 min when 3-4 runs share the GPU)
#        cpu --deterministic, 6 threads                    about 2 h (0.6 s/step)
#        cpu --deterministic, 2 threads, busy machine      about 4 h 45 min (measured, 3 runs in parallel)
#      three seeds run one after another here; run them in parallel by hand if threads allow
#   3. build_mix.py per candidate (LightGBM part as task2/build.py, 3 threads)   about 4 min
#      optional: trunc test per candidate                                         about 6 min
set -euo pipefail
cd "$(dirname "$0")"
TAG=${1:?usage: repro_nn.sh <tag>}
PY_LGB=${PY_LGB:-../.venv/bin/python}
PY_NN=${PY_NN:?set PY_NN to a python with torch}
DEVICE=${DEVICE:-cpu}
THREADS=${THREADS:-2}
DET=$([ "$DEVICE" = cpu ] && echo --deterministic || true)
mkdir -p logs runs candidates
t0=$(date +%s)
stamp() { echo "[$(date '+%F %T')] +$(( $(date +%s) - t0 ))s $*"; }

stamp "1. data"
[ -f data/D7_I405_S.npz ] || "$PY_LGB" -I prep.py > logs/${TAG}_prep.log 2>&1
[ -f data/extra_D7_I405_S.npz ] || "$PY_LGB" -I prep_extra.py >> logs/${TAG}_prep.log 2>&1

# seed configurations of r1, r2, r3
CONF=("--seed 20261101" "--seed 20261102" "--seed 20261103 --panel-weight uniform --easy-frac 0.3")
FOLDS=(all); [ "${OOF:-0}" = 1 ] && FOLDS=(A B C all)
for i in 0 1 2; do
  for k in "${FOLDS[@]}"; do
    stamp "2. train ${TAG}_r$((i + 1)) fold $k on $DEVICE"
    out=runs/${TAG}_r$((i + 1))
    [ -f $out/meta_$k.json ] && continue
    "$PY_NN" train.py --data data --out $out --fold $k --device "$DEVICE" $DET --threads "$THREADS" ${CONF[$i]} \
      > logs/${TAG}_r$((i + 1))_$k.log 2>&1
  done
done
REL=(runs/${TAG}_r1/release.pkl runs/${TAG}_r2/release.pkl runs/${TAG}_r3/release.pkl)
if [ "${OOF:-0}" = 1 ]; then
  stamp "2b. out-of-fold mix table"
  "$PY_LGB" -I combine_eval.py runs/${TAG}_r1 runs/${TAG}_r2 runs/${TAG}_r3 > logs/${TAG}_combine.txt 2>&1
fi

stamp "3. candidates (i5 options, network mixed into ongoing only)"
I5="--onset-mix v1:0.5,v1_thr:0.5 --ongoing-mix v1:0.15,phys:0.15,aug:0.35,onphys:0.35 --thr 0.37 --small-queue-thr 0.31"
# go50g = the 11-03 submission cpu3g_go50 (climatology gate 0.5: no new queue on links free at T with clim >= 0.5)
for spec in "go50g 0.5 --gate-clim 0.5" "go50 0.5" "go70 0.7" "go30 0.3"; do
  set -- $spec; name=$1; w=$2; shift 2
  out=candidates/${TAG}_$name/queue_submission.csv
  mkdir -p "$(dirname "$out")"
  "$PY_LGB" -I build_mix.py --nn "${REL[@]}" --w-onset 0 --w-ongoing "$w" "$@" --out "$out" -- $I5 > logs/${TAG}_build_$name.log 2>&1
  stamp "   ${TAG}_$name $(md5sum "$out" 2>/dev/null | cut -d' ' -f1 || md5 -q "$out")"
done
"$PY_LGB" -I safety.py candidates/${TAG}_go*/queue_submission.csv | grep -v "^   "
stamp "done"
