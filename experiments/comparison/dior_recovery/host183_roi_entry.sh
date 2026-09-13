#!/usr/bin/env bash
set -euo pipefail
CODE=/home/zechuan/iraod_code/checkouts/roi183-path-equivalence-860d4a0-bundle
PYTHON=/home/zechuan/anaconda3/envs/iraod/bin/python
BINDING=/mnt/SSD1_8TB/zechuan/IRAOD-New-host183-7ae4ac3/experiments/comparison/host_binding.py
cd "$CODE"
exec env PYTHONPATH="$CODE" PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$BINDING" native \
  "$CODE/experiments/comparison/dior_recovery/extract_roi_pre_fc_cls.py" "$@"
