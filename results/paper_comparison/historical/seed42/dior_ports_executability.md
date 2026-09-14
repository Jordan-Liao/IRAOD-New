# Faithful official SFOD ports (executability)

Reviewed 2026-09-06 on frozen `/mnt/SSD2_8TB/zechuan/IRAOD-New-strict-af@0f98a48`
and `/mnt/SSD2_8TB/zechuan` + `/home/zechuan` (maxdepth 3): zero IRG/LPLD/AASFOD/SF-YOLO trees.

| Method | Official | In-repo OBB | Executable now | Reason |
|---|---|---|---|---|
| IRG-SFDA | github.com/vibashan/irg-sfda (Faster R-CNN HBB, instance-relation graph) | no | **N/A** | No OBB/RIoU IRG module; cannot attach to frozen OrthoNet-50 + Oriented R-CNN source without a new graph implementation. |
| LPLD | github.com/CV-Det/LPLD | no | **N/A** | No code/config/ckpt on `.67`; pseudo-label distillation and proposal correspondence have no verified rotated-box port. Relabeling B would not be a faithful implementation. |
| Simple-SFOD / SF-UT | Mean Teacher / Unbiased Teacher ST | yes = method B | **skip** | Equivalent to executed RSAR and DIOR **B ST**. Do not duplicate. |
| DRU | github.com/lbktrinh/DRU | no | **N/A** | Absent from 0f98a48. A detector-independent core compatible with the fixed source and budget has not been verified or implemented; an approximate substitute would not establish a faithful port. |
| AASFOD / A²SFOD | github.com/ChuQiaosong/AASFOD | no | **N/A** | Absent from 0f98a48. No executable port verifies the mechanism, rotated assignment/IoU/NMS and no-target-GT recipe under this source. |
| SF-YOLO | github.com/vs-cv/sf-yolo | no | **N/A** | YOLO detector ≠ Oriented R-CNN; directly swapping detectors or retraining the source breaks the shared OrthoNet source. No isolated OBB core port exists. |

Do not launch GPU for these until a faithful OBB patch reuses `epoch_100.pth` and the no-target-GT protocol.

Official URLs above follow the original user plan. This integration did not
re-review those official repositories or prove that a faithful port is impossible.
N/A means absent/unverified under the frozen protocol, not a negative result.
