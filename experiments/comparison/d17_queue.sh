#!/usr/bin/env bash
# Usage: d17_queue.sh QUEUE GPUS
# Claims cells (lines "METHOD DOMAIN SEED") from QUEUE via mkdir and runs train_d17_seed_1gpu.sh.
# A GPU pair ("a,b") takes E/F first, then any cell as DDP2; a single GPU never takes E/F.
Q=$1
GPUS=$2
W=/home/zechuan/iraod_jobs/train_d17_seed_1gpu.sh
C=$Q.claims
mkdir -p "$C"
pair=0; [[ "$GPUS" == *,* ]] && pair=1
for pass in 1 2; do
  while read -r m d s; do
    [[ -z "$m" ]] && continue
    ef=0; [[ "$m" == E || "$m" == F ]] && ef=1
    (( pair == 0 && ef == 1 )) && continue
    (( pair == 1 && pass == 1 && ef == 0 )) && continue
    mkdir "$C/$m.$d.$s" 2>/dev/null || continue
    echo "start $(date --iso-8601=seconds) gpus=$GPUS" > "$C/$m.$d.$s/status"
    if bash "$W" "$GPUS" "$m" "$d" "$s" $((29500 + ${GPUS%%,*} * 10)) > "$C/$m.$d.$s/log" 2>&1; then
      echo "done $(date --iso-8601=seconds)" >> "$C/$m.$d.$s/status"
    else
      echo "fail rc=$? $(date --iso-8601=seconds)" >> "$C/$m.$d.$s/status"
    fi
  done < "$Q"
done
echo "SLOT_DONE gpus=$GPUS $(date --iso-8601=seconds)" >> "$C/slots.log"
