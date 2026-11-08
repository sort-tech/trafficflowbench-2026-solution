#!/bin/bash
# TrafficFlowBench 2026 (team sortdev): rebuild the two selected final submissions (A, B) from the public release.
#
#   TFB_REL=/path/to/kaggle_public final/run_all.sh [out_dir]
#
# Steps: Task 4 (ODME) -> Task 2 (queue) -> Task 1 (state; Task 3 is scored on it) -> merge -> format check.
# Every builder starts from the release only (--fresh / no caches). Seeds are fixed inside each builder.
# bash, macOS or Linux.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
export TFB_REL=${TFB_REL:-$ROOT/data/kaggle_public}
OUT=$(mkdir -p "${1:-$ROOT/final/out}" && cd "${1:-$ROOT/final/out}" && pwd)
PY=(${PYTHON:-$ROOT/.venv/bin/python} -I)
mkdir -p $OUT/logs
if [ "$(uname)" = Darwin ]; then TIME=(/usr/bin/time -l); else TIME=(/usr/bin/time -v); fi
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/logs/run_all.log; }

# --- selected configuration (edit together with final/README.md) ---
T4_CAND=nnls_own                                  # task4/build.py writes <out>/nnls_own/odme_submission.csv
# Task 2 has the two final choices; Task 1 and Task 4 are shared, so one run writes both submissions.
#   A = gated network blend (task2_nn, cpu3g_go50, LB 0.88376 with v7): i5 ongoing probabilities mixed 0.5 with a
#       3-seed spatio-temporal CNN (CPU deterministic weights task2_nn/weights/cpu_r{1,2,3}); the mix may not add a
#       queue on a link that is empty at T where the train climatology is >= 0.5 (those cells keep the i5 decision).
#   B = same blend with network weight 0.7 (task2_nn, cpu3g_go70, LB 0.88324 with v7); same weights and gate.
#   i5 (lgb_onthr_sq31_pao1535, LB 0.88019) is the LightGBM-only base of both: h5 with ongoing 0.15 v1 + 0.15 lgb_phys
#       + 0.35 lgb_aug + 0.35 lgb_onphys. h5 = onset 0.5 lgb_v1 + 0.5 v1_thr (bottleneck threshold features),
#       threshold 0.37, 0.31 for windows with 1-5 queued links. Add C:lgb_onthr_sq31_pao1535 to also write it.
# task2/repro_g5.sh trains every LightGBM model from the release in an empty folder and writes g5, h5, i5 and
# sclim75p; task2_nn/repro_final.sh then builds the network candidate from that folder.
# Entry "K:nn/<name>:<weight>" is a network candidate (ongoing mix weight), "K:<name>" a task2 candidate.
T2_CANDS=(A:nn/cpu3g_go50:0.5 B:nn/cpu3g_go70:0.7)
# Network candidate options (see task2_nn/repro_final.sh): NN_GATE climatology gate,
# NN_TRAIN=1 retrains the three networks (bit-identical to the shipped weights only on GB10, ~4 h 45 min).
NN_GATE=${NN_GATE:-0.5} NN_TRAIN=${NN_TRAIN:-0}
T2_EXTRA=${T2_EXTRA:-}
# Task 1 = v7 (gbm_v8s2rx3g_ss3f_ramp): cache -> ds13 (ds7 rows, ramp flow + conservation features) ->
# stage-1 odd/even -> fill -> stage-2 x3 seeds (lr 0.05 x 4000 fixed) + four ramp gap specialists
# (gap8, gap8b, seeds 0/11, ds11) at 0.75 -> in-split self-supervised correction, 3 disjoint 15% masks.
T1_ARGS=(--stage2 --retrain --s2-set ramp --selfsup --s2-tags gbm8s2r,gbm8s2r_s11,gbm8s2r_s23
         --gap-tag gap8,gap8_s11,gap8b,gap8b_s11 --gap-weight 0.75)
export SS_FOLDS=3
export TFB_THREADS=${TFB_THREADS:-8} OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
# -------------------------------------------------------------------

log "Task 4"
(cd $ROOT/task4 && "${TIME[@]}" "${PY[@]}" build.py --out $OUT/task4) > $OUT/logs/task4.log 2>&1

log "Task 2"
env PY=${PY[0]} "${TIME[@]}" $ROOT/task2/repro_g5.sh $OUT/task2_build $T2_EXTRA > $OUT/logs/task2.log 2>&1
for c in "${T2_CANDS[@]}"; do
  K=${c%%:*} C=${c#*:}
  if [ "${C%%/*}" = nn ]; then
    N=${C#nn/}; W=${N#*:}; N=${N%%:*}
    log "Task 2 network ($N, weight $W)"
    env PY=${PY[0]} PY_NN=${PY_NN:-$ROOT/.nn_venv/bin/python} NAME=$N W_GO=$W GATE=$NN_GATE TRAIN=$NN_TRAIN \
      "${TIME[@]}" $ROOT/task2_nn/repro_final.sh $OUT/task2_build $OUT/task2_nn > $OUT/logs/task2_nn_$K.log 2>&1
    SRC=$OUT/task2_nn/$N/queue_submission.csv
  else
    SRC=$OUT/task2_build/candidates/$C/queue_submission.csv
  fi
  mkdir -p $OUT/task2_$K && cp $SRC $OUT/task2_$K/queue_submission.csv
done
log "Task 1"
(cd $ROOT/task1 && "${TIME[@]}" "${PY[@]}" build.py "${T1_ARGS[@]}" --split validation,private \
   --out $OUT/task1/state_submission.csv) > $OUT/logs/task1.log 2>&1

log "merge"
FILES=($OUT/task1/state_submission.csv $OUT/task4/$T4_CAND/odme_submission.csv)
for c in "${T2_CANDS[@]}"; do
  K=${c%%:*}
  "${PY[@]}" $ROOT/trafficflowbench-public/src/merge_submissions.py \
    --state $OUT/task1/state_submission.csv --queue $OUT/task2_$K/queue_submission.csv \
    --odme $OUT/task4/$T4_CAND/odme_submission.csv --key $TFB_REL/submission_key.csv \
    --output $OUT/submission_$K.csv > $OUT/logs/merge_$K.log 2>&1
  "${PY[@]}" $ROOT/submit/check_submission.py $OUT/submission_$K.csv --template $TFB_REL/sample_submission.csv | tee -a $OUT/logs/run_all.log
  FILES+=($OUT/task2_$K/queue_submission.csv $OUT/submission_$K.csv)
done

for f in "${FILES[@]}"; do
  echo "$(md5 -q $f 2>/dev/null || md5sum $f | cut -d' ' -f1)  $f"
done | tee -a $OUT/logs/run_all.log
log "done -> $OUT/submission_A.csv, $OUT/submission_B.csv"
