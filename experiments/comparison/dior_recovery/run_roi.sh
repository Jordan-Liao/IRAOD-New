#!/usr/bin/env bash
set -euo pipefail
GPU=$1; METHOD=$2; CORRUPT=$3; ROLE=$4
source /mnt/shared/zechuan/iraod_artifacts/comparison/dior/queue_0f98a48/ckpt_path.sh
CODE=/mnt/SSD2_8TB/zechuan/IRAOD-New-strict-af
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
Q=/mnt/shared/zechuan/iraod_artifacts/comparison/dior/queue_0f98a48
CFG=/mnt/shared/zechuan/iraod_artifacts/dior_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/reproducibility/resolved_config.py
if [ "$ROLE" = ema ]; then CKPT=$(ckpt_ema "$METHOD" "$CORRUPT"); else CKPT=$(ckpt_stu "$METHOD" "$CORRUPT"); fi
[ -n "$CKPT" ] && [ -f "$CKPT" ]
OUT=/mnt/shared/zechuan/iraod_artifacts/comparison/deliverables/results/paper_comparison/roi_features/DIOR/${CORRUPT}/${METHOD}/${ROLE}
mkdir -p "$OUT"
if [ -f "$OUT/index.json" ]; then echo skip_roi $METHOD $CORRUPT $ROLE; exit 0; fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$CODE PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 IRAOD_RUNTIME_READY=1
export CONDA_PREFIX=/home/zechuan/miniforge3/envs/iraod
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$CODE"
set +e
"$PY" "$Q/extract_roi_pre_fc_cls.py" "$CFG" "$CKPT" --out-dir "$OUT" \
  --ann-file /mnt/shared/zechuan/iraod_data/DIOR/ImageSets/vis16.txt \
  --img-prefix /mnt/shared/zechuan/iraod_artifacts/dior_splits/${CORRUPT}/test/ \
  > "$OUT/roi.log" 2>&1
ec=$?
set -e
echo "roi_exit=$ec method=$METHOD corrupt=$CORRUPT role=$ROLE ckpt=$CKPT $(date --iso-8601=seconds)" | tee "$OUT/roi_status"
exit $ec
