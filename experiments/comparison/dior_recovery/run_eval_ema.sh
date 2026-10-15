#!/usr/bin/env bash
set -euo pipefail
GPU=$1
METHOD=$2
CORRUPT=$3
CODE=/mnt/SSD2_8TB/zechuan/IRAOD-New-strict-af
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
CFG=/mnt/shared/zechuan/iraod_artifacts/dior_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/reproducibility/resolved_config.py
DATA=/mnt/shared/zechuan/iraod_data/DIOR
SPLITS=/mnt/shared/zechuan/iraod_artifacts/dior_splits
ROOT=/mnt/shared/zechuan/iraod_artifacts/comparison/dior/${CORRUPT}/seed_42/0f98a48
if [ -d "$ROOT/methods/${METHOD}/ddp2/work" ]; then WORK=$ROOT/methods/${METHOD}/ddp2/work; else WORK=$ROOT/methods/${METHOD}/work; fi
OUT=$ROOT/methods/${METHOD}/eval_${CORRUPT}
mkdir -p "$OUT"
if [ -f "$OUT/eval_status" ] && grep -q "eval_exit=0" "$OUT/eval_status"; then
  echo "skip_done_eval $METHOD $CORRUPT"; exit 0
fi
EMA=$(ls -1t "$WORK"/iter_*_ema.pth 2>/dev/null | head -1 || true)
if [ -z "${EMA:-}" ]; then echo "NO_EMA $METHOD $CORRUPT"; exit 2; fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$CODE PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 IRAOD_RUNTIME_READY=1
export CONDA_PREFIX=/home/zechuan/miniforge3/envs/iraod
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$CODE"
set +e
"$PY" test.py "$CFG" "$EMA" --eval mAP --work-dir "$OUT" \
  --cfg-options data.test.ann_file=$DATA/ImageSets/test.txt \
    data.test.img_prefix=$SPLITS/${CORRUPT}/test/ \
  > "$OUT/eval.log" 2>&1
ec=$?
set -e
echo "eval_exit=$ec name=$METHOD corrupt=$CORRUPT ema=$EMA $(date --iso-8601=seconds)" | tee "$OUT/eval_status"
exit $ec
