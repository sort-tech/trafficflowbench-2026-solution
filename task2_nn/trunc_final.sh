#!/bin/bash
# T-after-removal test of the two final network candidates (fixed climatology gate 0.5): cpu3g_go50 (A), cpu3g_go70 (B).
set -euo pipefail
cd "$(dirname "$0")"
I5="--onset-mix v1:0.5,v1_thr:0.5 --ongoing-mix v1:0.15,phys:0.15,aug:0.35,onphys:0.35 --thr 0.37 --small-queue-thr 0.31"
REL="runs/cpu_r1/release.pkl runs/cpu_r2/release.pkl runs/cpu_r3/release.pkl"
OUT=${TRUNC_OUT:-/tmp}
for spec in "go50 0.5" "go70 0.7"; do
  set -- $spec; name=$1; w=$2
  /usr/bin/time -l ../.venv/bin/python -I build_mix.py --nn $REL --w-onset 0 --w-ongoing $w --gate-clim 0.5 \
    --out $OUT/trunc_cpu3g_$name.csv --trunc --compare candidates/cpu3g_$name/queue_submission.csv -- $I5 \
    > logs/trunc_cpu3g_$name.log 2>&1
  echo "$(date +%T) cpu3g_$name: $(grep -E 'compare with' logs/trunc_cpu3g_$name.log)"
done
