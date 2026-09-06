#!/usr/bin/env bash
set -euo pipefail
GPU=$1; METHOD=$2; CORRUPT=$3
source /mnt/shared/zechuan/iraod_artifacts/comparison/dior/queue_0f98a48/ckpt_path.sh
CODE=/mnt/SSD2_8TB/zechuan/IRAOD-New-strict-af
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
CFG=/mnt/shared/zechuan/iraod_artifacts/dior_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/reproducibility/resolved_config.py
CKPT=$(ckpt_ema "$METHOD" "$CORRUPT")
OUT=/mnt/shared/zechuan/iraod_artifacts/comparison/deliverables/results/paper_comparison/visualizations/DIOR/${CORRUPT}/${METHOD}
mkdir -p "$OUT"
if [ -f "$OUT/vis_status" ] && grep -q vis_exit=0 "$OUT/vis_status"; then echo skip_vis $METHOD $CORRUPT; exit 0; fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$CODE PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 IRAOD_RUNTIME_READY=1
export CONDA_PREFIX=/home/zechuan/miniforge3/envs/iraod
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$CODE"
set +e
"$PY" test.py "$CFG" "$CKPT" --show-dir "$OUT" --show-score-thr 0.3 \
  --cfg-options data.test.ann_file=/mnt/shared/zechuan/iraod_data/DIOR/ImageSets/vis16.txt \
    data.test.img_prefix=/mnt/shared/zechuan/iraod_artifacts/dior_splits/${CORRUPT}/test/ \
  > "$OUT/vis.log" 2>&1
ec=$?
set -e
echo "vis_exit=$ec method=$METHOD corrupt=$CORRUPT ckpt=$CKPT n=16 $(date --iso-8601=seconds)" | tee "$OUT/vis_status"
exit $ec
