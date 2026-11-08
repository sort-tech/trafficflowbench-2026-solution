#!/bin/bash
# Rebuild the Task 2 queue submissions from scratch in a clean folder (public data only).
#   task2/repro_g5.sh <dest_dir> [--aug-m3]
# g5 = onset lgb_v1, ongoing 0.2 v1 + 0.2 phys + 0.6 aug, threshold 0.37 (expected md5 1aa5a72e4373e3a6b646bce7d3563ee6).
# h5 = g5 with onset v1 .5 + v1_thr .5 (bottleneck threshold features) and threshold 0.31 for windows with 1-5 queued
#      links at T (lgb_onthr_sq31, expected md5 689418388a459b805963aaf9d1afdd44).
# i5 = h5 with ongoing v1 .15 / phys .15 / aug .35 / onphys .35 (lgb_onthr_sq31_pao1535, md5 2b9a602fd95bf2438ec50008fc7f42b8).
# sclim75p = i5 with the private onset climatology blended with validation + earlier private days (lam 0.75;
#      lgb_i5_sclim75p, md5 c5de3558bc17b73a77049186c07bb370).
# --aug-m3 also rebuilds lgb_v1_ongpm26 (aug -> aug_m3, 3 shift-seed tables x 3 tree seeds; md5 03c3730d3add388394fde6f4f59767d1).
# Only task2/*.py is copied; cache/ and models/ start empty. Stages run one at a time to stay under ~6 GB.
# bash, macOS or Linux.
set -e
SRC=$(cd "$(dirname "$0")" && pwd)
KAGGLE=$(dirname "$SRC")
DEST=${1:?usage: repro_g5.sh <dest_dir> [--aug-m3]}
AUGM3=${2:-}
PY=${PY:-$KAGGLE/.venv/bin/python}
export TFB_REL=${TFB_REL:-$KAGGLE/data/kaggle_public}
if [ "$(uname)" = Darwin ]; then TIME=(/usr/bin/time -l); else TIME=(/usr/bin/time -v); fi
md5of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }
mkdir -p "$DEST/logs"
[ -e "$DEST/cache" ] || [ -e "$DEST/models" ] && { echo "$DEST already has cache/ or models/; use an empty folder"; exit 1; }
cp "$SRC"/*.py "$DEST/"
cd "$DEST"
G1=(--panel D7_I405_N --panel D7_I10_W)
G2=(--panel D7_I210_E --panel D12_I5_S --panel D7_I210_W)
G3=(--panel D12_I5_N --panel D7_I10_E --panel D7_I405_S)
step() { local name=$1; shift; echo "== $name $(date +%T)"; "${TIME[@]}" "$PY" -I "$@" > logs/$name.log 2>&1 || { tail -20 logs/$name.log; exit 1; }
         grep -iE "real|maximum resident|elapsed \(wall" logs/$name.log | tr -s ' ' | paste -sd' ' -; }
step prep_cache prep_cache.py
step windows windows.py
step prep_release prep_release.py
step ds_1 build_dataset.py --max-ongoing-train 1200 "${G1[@]}"
step ds_2 build_dataset.py --max-ongoing-train 1200 "${G2[@]}"
step ds_3 build_dataset.py --max-ongoing-train 1200 "${G3[@]}"
step ds_v1_1 build_dataset_v1.py --max-ongoing-train 1200 "${G1[@]}"
step ds_v1_2 build_dataset_v1.py --max-ongoing-train 1200 "${G2[@]}"
step ds_v1_3 build_dataset_v1.py --max-ongoing-train 1200 "${G3[@]}"
# one process per model family keeps the peak memory down; build.py then only predicts
step fit_v1 fit_models.py v1
step fit_phys fit_models.py phys
step fit_aug fit_models.py aug
step g5 build.py --split all --onset-variant v1 --ongoing-variant v1 --ongoing-mix v1:0.2,phys:0.2,aug:0.6 --thr 0.37 \
     --out candidates/lgb_v1_ongpa26/queue_submission.csv
md5of candidates/lgb_v1_ongpa26/queue_submission.csv
step thr_build v1_thr.py build
mkdir -p models/cv models/v1_thr   # v1_thr.py fit writes there; earlier steps do not create models/cv
step thr_fit v1_thr.py fit
step h5 build.py --split all --onset-variant v1 --ongoing-variant v1 --onset-mix v1:0.5,v1_thr:0.5 \
     --ongoing-mix v1:0.2,phys:0.2,aug:0.6 --thr 0.37 --small-queue-thr 0.31 \
     --out candidates/lgb_onthr_sq31/queue_submission.csv
md5of candidates/lgb_onthr_sq31/queue_submission.csv
# i5 = h5 with ongoing v1 .15 / phys .15 / aug .35 / onphys .35 (final A)
step fit_onphys fit_models.py onphys
I5=(--split all --onset-variant v1 --ongoing-variant v1 --onset-mix v1:0.5,v1_thr:0.5
    --ongoing-mix v1:0.15,phys:0.15,aug:0.35,onphys:0.35 --thr 0.37 --small-queue-thr 0.31)
step i5 build.py "${I5[@]}" --out candidates/lgb_onthr_sq31_pao1535/queue_submission.csv
md5of candidates/lgb_onthr_sq31_pao1535/queue_submission.csv
# sclim75p = i5 with private onset climatology 0.75 train + 0.25 (all of validation + the private days before T's day)
step sclim75p build.py "${I5[@]}" --onset-split-clim 0.75 --out candidates/lgb_i5_sclim75p/queue_submission.csv
md5of candidates/lgb_i5_sclim75p/queue_submission.csv
if [ "$AUGM3" = "--aug-m3" ]; then
  for S in 1 2 3; do
    step ds_m3_S${S}_1 build_dataset.py --max-ongoing-train 1200 --shift-seed $S --out-dir cache/aug_m3/S$S "${G1[@]}"
    step ds_m3_S${S}_2 build_dataset.py --max-ongoing-train 1200 --shift-seed $S --out-dir cache/aug_m3/S$S "${G2[@]}"
    step ds_m3_S${S}_3 build_dataset.py --max-ongoing-train 1200 --shift-seed $S --out-dir cache/aug_m3/S$S "${G3[@]}"
    step fit_m3_S$S aug_m3.py fit $S full
  done
  step ongpm26 build.py --split all --onset-variant v1 --ongoing-variant v1 --ongoing-mix v1:0.2,phys:0.2,aug_m3:0.6 --thr 0.37 \
       --out candidates/lgb_v1_ongpm26/queue_submission.csv
  md5of candidates/lgb_v1_ongpm26/queue_submission.csv
fi
echo "== done $(date +%T)"
