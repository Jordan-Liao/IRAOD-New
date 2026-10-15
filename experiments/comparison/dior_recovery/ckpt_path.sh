ckpt_ema() {
  local m=$1 c=$2
  if [ "$m" = A ]; then echo /mnt/shared/zechuan/iraod_artifacts/dior_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/train/epoch_100.pth; return; fi
  local b=/mnt/shared/zechuan/iraod_artifacts/comparison/dior/$c/seed_42/0f98a48/methods/$m
  if [ -d "$b/ddp2/work" ]; then ls -1t "$b/ddp2/work"/iter_*_ema.pth | head -1; else ls -1t "$b/work"/iter_*_ema.pth | head -1; fi
}
ckpt_stu() {
  local m=$1 c=$2
  if [ "$m" = A ]; then echo ""; return; fi
  local b=/mnt/shared/zechuan/iraod_artifacts/comparison/dior/$c/seed_42/0f98a48/methods/$m
  if [ -d "$b/ddp2/work" ]; then echo "$b/ddp2/work/iter_185.pth"; else echo "$b/work/iter_185.pth"; fi
}
