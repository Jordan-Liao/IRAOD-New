#!/usr/bin/env bash
set -euo pipefail
GPU=$1
CORRUPT=$2
CODE=/mnt/SSD2_8TB/zechuan/IRAOD-New-strict-af
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
SRC_RUN=/mnt/shared/zechuan/iraod_artifacts/dior_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64
SOURCE=$SRC_RUN/train/epoch_100.pth
CFG=$SRC_RUN/reproducibility/resolved_config.py
DATA=/mnt/shared/zechuan/iraod_data/DIOR
SPLITS=/mnt/shared/zechuan/iraod_artifacts/dior_splits
Q=/mnt/shared/zechuan/iraod_artifacts/comparison/dior/queue_0f98a48
ROOT=/mnt/shared/zechuan/iraod_artifacts/comparison/dior/${CORRUPT}/seed_42/0f98a48
OUT=$ROOT/methods/A/eval_${CORRUPT}
mkdir -p "$OUT" "$ROOT/methods/A"
printf "%s\n" "0f98a48b539f9260055fc305784ffbc924f50451" > "$ROOT/git_commit.txt"
if [ -f "$OUT/eval_status" ] && grep -q "eval_exit=0" "$OUT/eval_status"; then
  echo "skip_done A $CORRUPT"; exit 0
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$CODE PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 IRAOD_RUNTIME_READY=1
export CONDA_PREFIX=/home/zechuan/miniforge3/envs/iraod
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$CODE"
set +e
"$PY" test.py "$CFG" "$SOURCE" --eval mAP --work-dir "$OUT" \
  --cfg-options data.test.ann_file=$DATA/ImageSets/test.txt \
    data.test.img_prefix=$SPLITS/${CORRUPT}/test/ \
  > "$OUT/eval.log" 2>&1
ec=$?
set -e
echo "eval_exit=$ec name=A corrupt=$CORRUPT ckpt=$SOURCE $(date --iso-8601=seconds)" | tee "$OUT/eval_status"
exit $ec
