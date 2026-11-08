#!/bin/bash
# T-after-removal test for the nn3 candidates (build_mix.py --trunc).
set -euo pipefail
cd "$(dirname "$0")"
I5="--onset-mix v1:0.5,v1_thr:0.5 --ongoing-mix v1:0.15,phys:0.15,aug:0.35,onphys:0.35 --thr 0.37 --small-queue-thr 0.31"
OUT=${TRUNC_OUT:-/tmp}
PREFIX=$1; shift
for spec in "go50 0.5" "go30 0.3" "go70 0.7"; do
  set -- $spec "$@"; name=$1; w=$2; shift 2
  ../.venv/bin/python -I build_mix.py --nn "$@" --w-onset 0 --w-ongoing "$w" --out "$OUT/trunc_${PREFIX}_$name.csv" --trunc \
    --compare candidates/${PREFIX}_$name/queue_submission.csv -- $I5 > logs/trunc_${PREFIX}_$name.log 2>&1
  tail -n 2 logs/trunc_${PREFIX}_$name.log
done
