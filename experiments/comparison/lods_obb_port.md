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
| SEM optimisation | two Adam, lr 1e-4, decay `lr/(1+5e-5·i)`, batch 8, 128 crops, h-flip, 160 000 iters, content & style both = target images (`Enhance/train.py:22-162`, README:59) | same, 160 000 iters (budget table) |
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
their official values; schedule events are scaled to the shared 1-epoch
budget. The SEM stage is the exception, for parity with the other style-network
port: the SF-YOLO TAM (same fc-AdaIN decoder recipe) ran its full official
160 000 steps, one seed-42 fit per (dataset, domain) shared by detector seeds
42/43/44 (`experiments/comparison/train_tam.py:6,65-66,152-153,209-210` in
`/mnt/shared/zechuan/iraod_code/worktrees/accepted-d668a77-67-20260912`;
`tam_runs_84a1aeb/rsar/chaff/seed_42/{training_config,terminal}.json`:
`steps=formal_steps=160000`, `detector_seeds=[42,43,44]`). LODS SEM follows
the same rule.

| Item | Official | Here |
|---|---|---|
| Detector schedule | bs 1, 3 epochs (clipart) / 1 epoch (foggy) (README:162-171), `USE_FLIPPED` doubles the epoch (`train_target.py:391`) | 1 epoch, global batch 32 (1 GPU × 32), RSAR iter_266 / DIOR ~185 |
| LR | SGD 1e-4 constant (`train_target.py:60-62`; `inv_lr_scheduler(gamma=0, power=-1)`, `utils.py:9-16`), momentum 0.9 | SGD lr 0.02, momentum 0.9, wd 1e-4, shared 100-iter linear warmup |
| Input scale | shorter side 600, max 1000 | shared 800×800 RResize, RRandomFlip h/v/diag |
| SEM stage | separate stage, 160 000 iters × 8, before a 3-epoch detector stage (README:59,162) | **full 160 000 iters × 8** (~1.6 h alone on one A100), one seed-42 fit per (dataset, domain) at `<ART>/rsar/<domain>/lods_sem_seed42/` (DIOR: `ext_runs_e1/LODS_sem_s42/<domain>/`), reused by detector seeds 42/43/44; reads only the unlabeled adaptation images |

Effect: the SEM matches the official one in length and recipe. The detector
stage is the shared 1×32, 1-epoch run, so the teacher receives 265 EMA updates
at 0.999 (≈ 23 % student weight) instead of ~6000 bs-1 updates in the official
clipart recipe; the per-update style augmentation is official. Seeds 43/44 of a
domain must start after its seed-42 SEM file exists (the launchers train it
only when missing).

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
14. SEM training follows the generic `voc2clipart/voc2wc/KC` branch (style dir = target images, `random_style`, `loss_c` on, 4-level style losses, ImageNet VGG16 encoder); the `city2foggy` training branch cannot be used (it needs a hand-made mean-fog style image and an encoder taken from a VGG16 source detector, which does not exist for an OrthoNet source). Adaptation-time style mixing uses the city2foggy value **`style_add_alpha = 0.4`** (README:164,171), see "Alpha decision". Alpha never enters SEM training: `Enhance/train.py:142,154` call the network without alpha, so `net.py:97` uses its default 1.0 in every scenario; the 160k SEM fits are therefore alpha-independent.
15. Style images are re-read from the target image directory of the current sample (`dirname(img_meta['filename'])`), uniform over all target images, like `load_style_img` over `imdb._image_index`.

## Open questions (method-relevant)

- **alpha**: decided, see "Alpha decision".
- **Proposal count** (deviation 2): 2000 vs 300 teacher RoIs changes the instance-graph size and the confident-background share of the CE.

## Alpha decision

`style_add_alpha = 0.4` for RSAR and DIOR (coordinator decision, fixed a priori from
the shift type only). Our targets are synthetic corruptions of the same scenes
as the source, which matches the official Cityscapes→Foggy-Cityscapes setting
(alpha 0.4). The generic/clipart value 1.0 targets cross-style shifts we do not
have. No target labels and no target-test result were used.

The first gate run used alpha 1.0. It was stopped at about iteration 160 of 265,
unevaluated, and marked `invalid_alpha1`
(`.../rsar/chaff/seed_42/lods_e1/methods/invalid_alpha1_LODS/INVALID_ALPHA1`).
No AP exists for it, by design: having both results would turn the alpha choice
into selection on target GT.

## Visual check

`tools/lods/vis_sem.py` grids (columns: original | SEM | original zoom with GT
outline | SEM zoom), detector input scale, random target style, same images and
style draws for both alphas. GT is read only to pick and outline targets in the
figure. Full-size files are in `/mnt/shared/zechuan/iraod_artifacts/comparison/lods_visual/`
(gpu-67). 1000-px copies and per-image metrics (`*.json`: Sobel-gradient
correlation inside GT boxes, target-vs-ring contrast) are in
`experiments/comparison/lods_visual/`. SEMs: RSAR chaff
`<ART>/rsar/chaff/lods_sem_seed42/sem_iter_160000.pth`, DIOR fog
`<ART>/dior/fog/lods_sem_seed42/sem_iter_160000.pth` (160k iters each).

Used setting, alpha 0.4:

| Grid | Verdict |
|---|---|
| `rsar_chaff_small_ships_alpha0.4` | River/land layout and most ships survive as blurred blobs; fake worm/checker texture and a sepia tint over land and water; fine ship structure lost. Gradient corr in ship boxes 0.19–0.38; ship-vs-ring contrast 1.2–1.8 (orig 1.2–1.6), i.e. ships not erased. |
| `rsar_chaff_slender_alpha0.4` | Bridges and long linear structures visible at full scale (corr 0.24–0.63); zoomed detail smeared into texture. |
| `dior_fog_dense_alpha0.4` | Layout and dense parked vehicles/tanks preserved as small blobs; fog largely removed; mild texture (corr 0.11–0.49). |
| `dior_fog_slender_alpha0.4` | Marina ships and bridges keep their shape (corr 0.19–0.67). |

Rejected setting, alpha 1.0 (kept for the record):

| Grid | Finding |
|---|---|
| `rsar_chaff_small_ships_alpha1.0` | Destroys structure: water filled with checkerboard/speckle texture, most small ships erased. Gradient correlation in ship boxes **0.05–0.12**; ship-vs-ring contrast falls from 1.2–1.6 to 0.4–1.3. |
| `rsar_chaff_slender_alpha1.0` | Bridges/wakes mostly lost (corr 0.16–0.34). |
| `dior_fog_dense_alpha1.0` | Layout kept, small vehicles faint or gone, embossed fake texture (corr 0.01–0.30). |
| `dior_fog_slender_alpha1.0` | Shapes kept, detail lost (corr 0.00–0.39). |

`diag_rsar_chaff_small_ships_alpha0_reconstruction`: decoder(encoder(x)) without
style keeps every ship in place (corr 0.53–0.57). The damage therefore comes from
the fc-AdaIN style step, not from misaligned geometry in the port.
