#!/usr/bin/env bash
"""VLST Phase 2: Single corruption quick validation.

Runs ST baseline vs ST+Proto on noise_suppression to verify feature-level
signal exists before full experiment.
"""

set -e

CORRUPTION="noise_suppression"
EPOCHS=2
GPU=0

echo "=========================================="
echo "VLST Phase 2: Quick Validation"
echo "Corruption: $CORRUPTION"
echo "Epochs: $EPOCHS"
echo "=========================================="

cd /myfile/mycode/IRAOD-New

# Prepare corruption paths
export CORRUPT=$CORRUPTION
TARGET_VAL_IMG="/myfile/dataset/RSAR/corruptions/${CORRUPT}/val/images/"
TARGET_TEST_IMG="/myfile/dataset/RSAR/corruptions/${CORRUPT}/test/images/"

if [ ! -d "$TARGET_VAL_IMG" ]; then
    echo "Error: Corruption directory not found: $TARGET_VAL_IMG"
    exit 1
fi

echo ""
echo "Target domain: $TARGET_VAL_IMG"
echo ""

# ST baseline (no CGA, no VLST)
echo "[1/3] Running ST baseline..."
CUDA_VISIBLE_DEVICES=$GPU \
CGA_SCORER=none \
python tools/train.py \
    configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_st_baseline_rsar_orthonet.py \
    --work-dir work_dirs/st_baseline_${CORRUPTION}_phase2 \
    --cfg-options \
        data.train.img_prefix_u="${TARGET_VAL_IMG}" \
        data.val.img_prefix="${TARGET_TEST_IMG}" \
        data.test.img_prefix="${TARGET_TEST_IMG}" \
        runner.max_epochs=${EPOCHS} \
        evaluation.interval=${EPOCHS} \
        checkpoint_config.interval=${EPOCHS} \
        log_config.interval=50

echo ""
echo "[2/3] Running ST + Proto..."
CUDA_VISIBLE_DEVICES=$GPU \
CGA_SCORER=sarclip \
python tools/train.py \
    configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_vlst_rsar_orthonet.py \
    --work-dir work_dirs/st_proto_${CORRUPTION}_phase2 \
    --cfg-options \
        data.train.img_prefix_u="${TARGET_VAL_IMG}" \
        data.val.img_prefix="${TARGET_TEST_IMG}" \
        data.test.img_prefix="${TARGET_TEST_IMG}" \
        runner.max_epochs=${EPOCHS} \
        evaluation.interval=${EPOCHS} \
        checkpoint_config.interval=${EPOCHS} \
        log_config.interval=50

echo ""
echo "[3/3] Comparing results..."
echo ""
echo "ST baseline:"
grep -E "mAP|Epoch\(val\)" work_dirs/st_baseline_${CORRUPTION}_phase2/*.log.json 2>/dev/null | tail -5 || echo "  (logs not found)"

echo ""
echo "ST + Proto:"
grep -E "mAP|Epoch\(val\)" work_dirs/st_proto_${CORRUPTION}_phase2/*.log.json 2>/dev/null | tail -5 || echo "  (logs not found)"

echo ""
echo "=========================================="
echo "Phase 2 validation complete"
echo "=========================================="
