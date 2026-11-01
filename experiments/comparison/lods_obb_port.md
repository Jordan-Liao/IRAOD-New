# LODS (our reimplementation) — OBB port and deviation log

Method id `LODS`. Paper: Li et al., "Source-Free Object Detection by Learning
to Overlook Domain Style", CVPR 2022 (Oral). Official code:
`github.com/Flashkong/Source-Free-Object-Detection-by-Learning-to-Overlook-Domain-Style`
@ `2dc11b1` (cited below as `file:line`).

Code: `sfod/lods_sem.py` (SEM), `sfod/rotated_lods.py` (`LODSTeacher`),
`tools/lods/train_sem.py`, `tools/lods/vis_sem.py`,
configs `configs/unbiased_teacher/sfod/lods_oriented_rcnn_{rsar,dior}_orthonet_strict.py`,
launchers `experiments/comparison/rsar_lods_seed.sh` and the `LODS` case of
`train_d17_seed_1gpu.sh`. Tests: `tools/cga_research/tests/test_lods_obb.py`.

## What is ported as in the official code

| Mechanism | Official | Port |
|---|---|---|
| SEM architecture | VGG16 `features[:19]` encoder (frozen, ceil-mode pools), mirrored decoder, `fc1`/`fc2` 1024→512→512 (`Enhance/net.py:6-36`) | same (`lods_sem.py`) |
| SEM AdaIN | target mean/std = `fc(style stats ++ content stats)` (`Enhance/function.py:15-29`) | same |
| SEM training losses | decoder step: `loss_c` + 4-level reconstruction `loss_const`; fc step: 50·style(g, style) + 1·style(g, content), 4 levels (`net.py:97-140`, generic branch) | same |
| SEM optimisation | two Adam, lr 1e-4, decay `lr/(1+5e-5·i)`, batch 8, 128 crops, h-flip, 160 000 iters, content & style both = target images (`Enhance/train.py:22-162`, README:59) | same per-step settings; length mapped to 53 334 iters (budget table) |
| SEM inference | per image, random target image as style resized to content size, alpha blend, clip to [0,255] (`enhance_base.py:14-40,83-129`) | same (`LODSTeacher.enhance`) |
| Teacher/student | teacher gets the original image, student gets SEM(original) (`faster_rcnn_style.py:66-68`); student classifies the teacher's proposals (`:101-102`) | same; student RoIs = teacher RoIs index-for-index |
| Teacher proposals | teacher RPN forced to eval (`faster_rcnn_style.py:75-76`) → test-mode proposals | teacher `rpn_head.simple_test_rpn` (eval EMA model, detector test cfg) |
| Supervision | CE on teacher-confident RoIs (max prob ≥ 0.8, background included) (`train_target.py:606-613`); `cons_loss` × 0 (`:640`) | same; `cons_loss` not computed (weight 0) |
| Instance graph | node feature = flatten(prob ⊗ fc7); GW-KL with cosine affinities; transport prior `T = I + 0.5·mask·cos(teacher logits)`, mask = same confident foreground class, background ignored (`train_target.py:615-635`, `gromovWasserstein.py`) | same |
| Global graph | GW-KL over spatial positions of the shared base feature, `T = I` (`train_target.py:637-638`) | same on backbone C4 (see OBB adaptations) |
| Loss weights | label 1, gw_ins 0.1, gw_global 0.1, gw_add_rate 0.5 (`train_target.py:140-154,641-645`) | same |
| Teacher update | EMA α = 0.999 after every student step (`train_target.py:103-106,653-655`) | `momentum=0.999` (UT-style update at the start of the next step, as every EMA method here) |
| Param groups | `RCNN_cls_score` lr ×0.1; no bias weight decay (`train_target.py:496-512`; `cfgs/res101.yml:11-12`; `config.py:46`) | `paramwise_cfg`: `fc_cls` lr ×0.1, `bias_decay_mult=0` |
| Untrained parts | student RPN and `RCNN_bbox_pred` receive no loss | same (no RPN/box loss; `use_bbox_reg=False`) |

## Budget mapping (shared protocol wins)

Rule for the whole comparison: per-step constants (EMA keep rate 0.999, loss
weights 1/0.1/0.1, threshold 0.8, gw_add_rate 0.5, SEM lr / lr-decay
coefficient / loss weights / batch 8 / crop 128, param-group multipliers) keep
their official values; only schedule events are scaled to the shared 1-epoch
budget. LODS has one schedule event, the SEM stage length.

| Item | Official | Here |
|---|---|---|
| Detector schedule | bs 1, 3 epochs (clipart) / 1 epoch (foggy) (README:162-171), `USE_FLIPPED` doubles the epoch (`train_target.py:391`) | 1 epoch, global batch 32 (1 GPU × 32), RSAR iter_266 / DIOR ~185 |
| LR | SGD 1e-4 constant (`train_target.py:60-62`; `inv_lr_scheduler(gamma=0, power=-1)`, `utils.py:9-16`), momentum 0.9 | SGD lr 0.02, momentum 0.9, wd 1e-4, shared 100-iter linear warmup |
| Input scale | shorter side 600, max 1000 | shared 800×800 RResize, RRandomFlip h/v/diag |
| SEM stage | separate stage of 160 000 iters × 8 ahead of a 3-epoch detector stage (generic branch, README:59,162) | 160 000 × (1 / 3) = **53 334 iters × 8** (~45 min on one A100) ahead of the 1-epoch detector stage, once per (target domain, seed); reads only the unlabeled adaptation images |

Effect of the mapping: the SEM decoder/fc see 1/3 of the official crop count
(~427k vs 1.28M 128-px crops), and because the official per-step lr decay
`lr/(1+5e-5·i)` is kept, training stops at lr ≈ 2.7e-5 instead of ≈ 1.1e-5.
The SEM is therefore less converged than the official one (expected: weaker
style fidelity and more decoder blur/artifacts; see the visual check). The
detector stage is the shared 1×32, 1-epoch run, so the teacher receives
265 EMA updates at 0.999 (≈ 23 % student weight) instead of ~6000 bs-1 updates
in the official clipart recipe.

## Deviations

OBB / detector adaptations (forced by the shared Oriented R-CNN detector):

1. HBB proposals → rotated proposals from the OrientedRPN (`(cx, cy, w, h, θ)`, le90); RoIAlign → `RoIAlignRotated` (7×7, `RotatedSingleRoIExtractor` level mapping); per-image box ROI tensors via `rbbox2roi`.
2. Teacher proposal count follows our detector's test cfg (OrientedRPN: nms_pre 2000/level, NMS 0.8, max 2000 per image) instead of py-faster-rcnn's TEST cfg (pre 6000, post 300, NMS 0.7, min size 16, `config.py:191-199`). Both are "the teacher RPN in eval mode"; the RoI count per graph is ~2000 instead of 300.
3. Degenerate teacher proposals (non-finite, w ≤ 0 or h ≤ 0) are dropped before RoI pooling — the same rule `UnbiasedTeacher.create_pseudo_results` applies to pseudo boxes. LODS creates no pseudo boxes, so that UT function is not called; the rule is kept on the teacher geometry LODS does use.
4. fc7 = output of `RotatedShared2FCBBoxHead.shared_fcs` (1024-d, post-ReLU) instead of res101 `layer4`+avg-pool (2048-d). Background is mmdet's last class index instead of index 0.
5. Global graph on backbone C4 (OrthoNet-50 stage 3 output, stride 16, 1024 ch) = the official res101 `RCNN_base4` (= `layer3`) stage. FPN levels are not used for this loss.
6. Rotated IoU is not needed by the official algorithm (correspondence is identity on the teacher's proposal list); it is used only in the tests to verify that correspondence.

Batching / protocol adaptations:

7. Official target training is batch 1. Here a batch of 32 is processed as 32 per-image graphs (instance GW and global GW each averaged over images; the official `view(B·C, HW)` folding only coincides at B=1); the label CE averages over all confident RoIs of the batch.
8. SEM runs on the unpadded image region; the global graph uses the unpadded part of C4 (batch padding does not exist at bs 1).
9. Shared protocol: lr/batch/epoch/warmup/input scale/flip as above; final-iteration EMA teacher evaluated on corruption TEST; no target labels anywhere; `strict_source_free=True weight_l=0 weight_u=1 use_bbox_reg=False`.
10. Frozen parts follow the shared detector config (stem + stage 1 frozen, BN in eval mode) instead of res101's `set_bn_fix` (all BN affine params frozen, `resnet_style.py:57-84`).
11. The second (strong) view of `StrictSourceFreeDOTADataset` is reduced to a `Collect` of metadata and is ignored: LODS has no colour-jitter view; the only augmented view is SEM(original).

SEM-specific:

12. Encoder weights: Oxford VGG16 (VGG_ILSVRC_16_layers.caffemodel, sha256 `a6196bc4…`), conv1_1–conv4_1, BGR, converted at `/mnt/shared/zechuan/iraod_artifacts/third_party/vgg16_oxford/encoder_split_enc_keys.pt` (sha256 `5d5e4b94…`, also used by the SF-YOLO port). The official `vgg16_ori.pth` is "extracted from py-faster-rcnn", i.e. the same ImageNet caffemodel; byte identity with the Google-Drive file is not verified. Input convention is the official one (BGR − PIXEL_MEANS 102.98/115.95/122.77).
13. SEM crop source: images resized with the detector's 800×800 keep-ratio rule before the 128 random crop (official resizes to a fixed 600×800, `Enhance/train.py:29-31`, matching its own detector scale).
14. Scenario branch: the generic `voc2clipart/voc2wc/KC` branch (style dir = target images, `random_style`, `loss_c` on, 4-level style losses, ImageNet VGG16 encoder, alpha 1.0 from README:162). The `city2foggy` branch is not used: it needs a hand-made mean-fog style image and an encoder taken from a VGG16 source detector, which does not exist for an OrthoNet source. See open question below.
15. Style images are re-read from the target image directory of the current sample (`dirname(img_meta['filename'])`), uniform over all target images, like `load_style_img` over `imdb._image_index`.

## Open questions (method-relevant)

- **alpha / branch.** Official commands use `style_add_alpha 1.0` with random target styles (clipart) and `0.4` with a fixed mean-fog style (city2foggy); the argparse default is 0.5 and no value is given for watercolor/KITTI. Our corruptions (SAR chaff/noise, DIOR-C weather/noise) are closer in kind to foggy, but the foggy recipe cannot be reproduced (see 14). The port uses the generic branch with alpha 1.0. Changing it is a one-line `model.cfg.style_add_alpha` override, but it changes the method and must not be tuned on target labels.
- **Proposal count** (deviation 2): 2000 vs 300 teacher RoIs changes the instance-graph size and the confident-background share of the CE.

## Visual check

`tools/lods/vis_sem.py` grids (original | SEM | original zoom with GT outline | SEM zoom). GT is read only to pick and outline targets in the figure. Paths and verdict: see the PR description.
