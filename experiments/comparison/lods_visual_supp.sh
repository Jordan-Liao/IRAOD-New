#!/usr/bin/env bash
# LODS SEM visual check on 7 more domains (alpha 0.4, the matrix value), CPU only.
# Visualization only: target VAL GT boxes are read only to pick and outline crops;
# nothing here feeds a hyperparameter, checkpoint or method choice. No AP is computed.
#   bash experiments/comparison/lods_visual_supp.sh   # from a code dir with tools/lods/vis_sem.py
# Ran on gpu-138 (2026-11-05) with the defaults below; SEM copies were sha256-checked
# against lods_matrix.csv sem_sha256 (canonical files: gpu-67 <ART>/<dataset>/<domain>/lods_sem_seed42/).
set -euo pipefail
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PY=${PY:-/home/zechuan/miniforge3/envs/iraod/bin/python}
OUT=${OUT:-/home/zechuan/iraod_artifacts/comparison/lods_visual/supp_20261105}
SEM_ROOT=${SEM_ROOT:-$OUT/sem}               # <SEM_ROOT>/<dataset>_<domain>/sem_iter_160000.pth
RSAR=${RSAR:-/home/zechuan/iraod_data/RSAR}  # val/{images,annfiles}, corruptions/<domain>/val/images
DIOR_VAL=${DIOR_VAL:-/mnt/SSD_4TB/zechuan/dior_splits}   # <domain>/val
DIOR_ANN=${DIOR_ANN:-"/home/zechuan/iraod_data/DIOR/Annotations/Oriented Bounding Boxes"}
mkdir -p "$OUT"
# modes exclude images shown by earlier modes in $OUT, so a rerun needs an empty $OUT to reproduce
if compgen -G "$OUT/*_alpha0.4.json" > /dev/null; then echo "OUT=$OUT already has grids; use an empty OUT" >&2; exit 1; fi
run() {  # dataset domain mode classes...   (modes of one domain must run in order small, slender, dense)
  local ds=$1 dom=$2 mode=$3; shift 3
  local sem=$SEM_ROOT/${ds}_${dom}/sem_iter_160000.pth img ann used=()
  if [ "$ds" = rsar ]; then
    ann=$RSAR/val/annfiles; img=$RSAR/corruptions/$dom/val/images; [ "$dom" = clean ] && img=$RSAR/val/images
  else img=$DIOR_VAL/$dom/val ann=$DIOR_ANN; fi
  # images already shown by this domain's earlier modes are excluded, so each mode shows new images
  for prev in "$OUT/${ds}_${dom}"_*_alpha0.4.json; do
    [ -f "$prev" ] || continue
    case "$prev" in *"_${mode}_alpha0.4.json") continue;; esac
    mapfile -t -O "${#used[@]}" used < <("$PY" -c 'import json,sys; [print(r["image"]) for r in json.load(open(sys.argv[1]))]' "$prev")
  done
  nice -n 19 ionice -c3 "$PY" tools/lods/vis_sem.py --sem "$sem" --img-dir "$img" --ann-dir "$ann" \
    --fmt "$ds" --classes "$@" --mode "$mode" --n 3 --alpha 0.4 --seed 0 --exclude "${used[@]}" \
    --out "$OUT/${ds}_${dom}_${mode}_alpha0.4.jpg" > "$OUT/${ds}_${dom}_${mode}.log" 2>&1
}
for dom in clean gaussian_white_noise point_target; do
  run rsar $dom small ship
  run rsar $dom slender ship bridge harbor
  run rsar $dom dense ship car tank
done
for dom in clean snow gaussian_noise pixelate; do
  run dior $dom small vehicle
  run dior $dom slender ship bridge harbor
  run dior $dom dense vehicle ship storagetank
done
