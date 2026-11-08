#!/bin/bash
# Task 2 network candidates cpu3g_go50 / cpu3g_go70 (i5 + 0.5 / 0.7 network on ongoing, climatology gate 0.5) from a fresh task2 build.
#
#   task2_nn/repro_final.sh <task2 build dir> <out dir>
#
# <task2 build dir>: the folder task2/repro_g5.sh filled (its *.py, cache/ and models/; i5 is built from them again).
# Steps (bash, macOS or Linux; Python 3.12 venvs):
#   1. prep.py      task2 cache + release -> <out>/data/<panel>.npz (only release arrays and tables are used here)
#   2. rel_predict  shipped weights task2_nn/weights/cpu_r{1,2,3}/net_all.pt -> <out>/rel_r{1,2,3}.pkl (PY_NN: torch 2.14.1)
#      TRAIN=1 instead retrains the three models (train.py --device cpu --deterministic --threads 2). That is bit-identical
#      to the shipped weights only on GB10 (aarch64 Linux, torch 2.14.1+cu130, CPU path): about 4 h 45 min there with the
#      three seeds in parallel. On other machines the training path differs from step ~500 on (see RESULTS.md).
#   3. build_mix.py --w-ongoing 0.5 --gate-clim 0.5 with the i5 options -> <out>/cpu3g_go50/queue_submission.csv
#   4. safety.py (low-confidence onset windows) and md5 against the submitted file.
# Other candidates: NAME / W_GO / GATE (/ WIDE=1), e.g. NAME=cpu3g_go70 W_GO=0.7, NAME=cpu3g_go50_c00 GATE=0.0.
# Expected md5: cpu3g_go50 d7233f01..., cpu3g_go70 08dbae54..., cpu3g_go30 e6540ed0..., cpu3g_go50_c00 5dd59829...,
#   cpu3g_go50_c07 e98adff9..., cpu3g_go50_w03 (W_GO=0.5 GATE=0.3 WIDE=1) 5c4eba60...
set -euo pipefail
NN=$(cd "$(dirname "$0")" && pwd)
KAGGLE=$(dirname "$NN")
T2=$(cd "${1:?usage: repro_final.sh <task2 build dir> <out dir>}" && pwd)
OUT=$(mkdir -p "${2:?out dir}" && cd "$2" && pwd)
PY=${PY:-$KAGGLE/.venv/bin/python}
PY_NN=${PY_NN:-$KAGGLE/.nn_venv/bin/python}
export TFB_REL=${TFB_REL:-$KAGGLE/data/kaggle_public} T2_DIR=$T2 NN_DATA=$OUT/data
if [ "$(uname)" = Darwin ]; then TIME=(/usr/bin/time -l); else TIME=(/usr/bin/time -v); fi
md5of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }
mkdir -p $OUT/logs
step() { local name=$1; shift; echo "== $name $(date +%T)"; "${TIME[@]}" "$@" > $OUT/logs/$name.log 2>&1 || { tail -20 $OUT/logs/$name.log; exit 1; }
         grep -iE " real|maximum resident|elapsed \(wall" $OUT/logs/$name.log | tr -s ' ' | paste -sd' ' -; }
[ -f $OUT/data/D7_I405_S.npz ] || step prep "$PY" -I $NN/prep.py
REL=()
for r in 1 2 3; do
  if [ "${TRAIN:-0}" = 1 ]; then
    CONF=("--seed 20261101" "--seed 20261102" "--seed 20261103 --panel-weight uniform --easy-frac 0.3")
    if [ ! -f $OUT/runs/r$r/release.pkl ]; then   # a second candidate in the same out dir reuses the trained models
    [ $r = 3 ] && step prep_extra "$PY" -I $NN/prep_extra.py
    step train_r$r "$PY_NN" $NN/train.py --data $OUT/data --out $OUT/runs/r$r --fold all --device cpu --deterministic --threads 2 ${CONF[$((r - 1))]}
    fi
    REL+=($OUT/runs/r$r/release.pkl)
  else
    [ -f $OUT/rel_r$r.pkl ] || step predict_r$r "$PY_NN" $NN/rel_predict.py $NN/weights/cpu_r$r --out $OUT/rel_r$r.pkl --data $OUT/data
    REL+=($OUT/rel_r$r.pkl)
  fi
done
I5=(--onset-mix v1:0.5,v1_thr:0.5 --ongoing-mix v1:0.15,phys:0.15,aug:0.35,onphys:0.35 --thr 0.37 --small-queue-thr 0.31)
NAME=${NAME:-cpu3g_go50}; W_GO=${W_GO:-0.5}; GATE=${GATE:-0.5}
WIDE_ARG=(); [ "${WIDE:-0}" = 1 ] && WIDE_ARG=(--gate-wide)
mkdir -p $OUT/$NAME
step build_$NAME "$PY" -I $NN/build_mix.py --nn "${REL[@]}" --w-onset 0 --w-ongoing $W_GO --gate-clim $GATE ${WIDE_ARG[@]+"${WIDE_ARG[@]}"} \
  --out $OUT/$NAME/queue_submission.csv -- "${I5[@]}"
"$PY" -I $NN/safety.py $OUT/$NAME/queue_submission.csv | grep -v "^   "
echo "$NAME $(md5of $OUT/$NAME/queue_submission.csv) (submitted: cpu3g_go50 11-03 d7233f01b47c9f71bc36ed3dfd942ef5, cpu3g_go70 11-04 08dbae5412257a7e0ed30748bbe9e9c5)"
