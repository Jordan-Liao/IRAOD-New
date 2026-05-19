#!/usr/bin/env bash
set -euo pipefail

SRC_HOST="172.25.71.123"
SRC_PORT="2873"

mkdir -p /root/mycode /root/dataset

rsync -av --info=progress2 --partial --append-verify \
  "rsync://${SRC_HOST}:${SRC_PORT}/iraod_new/" \
  /root/mycode/IRAOD-New/

rsync -av --info=progress2 --partial --append-verify \
  "rsync://${SRC_HOST}:${SRC_PORT}/sarclip_code/" \
  /root/mycode/SARCLIP/

rsync -av --info=progress2 --partial --append-verify \
  "rsync://${SRC_HOST}:${SRC_PORT}/rsar_dataset/" \
  /root/dataset/RSAR/

rsync -av --info=progress2 --partial --append-verify \
  "rsync://${SRC_HOST}:${SRC_PORT}/sarclip_dataset/" \
  /root/dataset/SARCLIP/
