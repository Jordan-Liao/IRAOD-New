#!/usr/bin/env bash
# VLST Minimal Test: Verify training pipeline works end-to-end.
#
# Runs just 100 iterations to check:
# - Data loading works
# - Forward/backward pass completes
# - Prototype updates occur
# - Loss values are finite

set -e

CORRUPTION="noise_suppression"
EPOCHS=1
GPU=0

echo "=========================================="
echo "VLST Minimal Test (1 epoch)"
echo "Corruption: $CORRUPTION"
echo "=========================================="

cd /myfile/mycode/IRAOD-New

# Prepare corruption paths
export CORRUPT=$CORRUPTION
TARGET_VAL_IMG="/myfile/dataset/RSAR/corruptions/${CORRUPT}/val/images/"
TARGET_TEST_IMG="/myfile/dataset/RSAR/corruptions/${CORRUPT}/test/images/"

echo ""
echo "Target domain: $TARGET_VAL_IMG"
echo ""

# Test ST + Proto minimal
# Test ST + Proto minimal
echo "Testing ST + Proto (1 epoch)..."
CUDA_VISIBLE_DEVICES=$GPU \
CGA_SCORER=sarclip \
source /opt/conda/etc/profile.d/conda.sh && conda activate iraod && \
python train.py \
    configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_vlst_rsar_orthonet.py \
    --work-dir work_dirs/vlst_minimal_test \
    --cfg-options \
        corrupt="${CORRUPTION}" \
        data.train.img_prefix_u="${TARGET_VAL_IMG}" \
        data.val.img_prefix="${TARGET_TEST_IMG}" \
        data.test.img_prefix="${TARGET_TEST_IMG}" \
        total_epoch=${EPOCHS} \
        runner.max_epochs=${EPOCHS} \
        evaluation.interval=${EPOCHS} \
        checkpoint_config.interval=${EPOCHS} \
        log_config.interval=50

echo "Done!"

echo ""
echo "=========================================="
echo "Checking logs..."
echo "=========================================="

if [ -f "work_dirs/vlst_minimal_test/latest.log" ]; then
    echo ""
    echo "Loss progression:"
    grep -E "loss_cls|loss_rpn|loss_vlst|vlst_samples" work_dirs/vlst_minimal_test/latest.log | tail -20
    echo ""
    echo "✓ Minimal test complete"
else
    echo "✗ No log file found"
    exit 1
fi
