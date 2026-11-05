# DETR-group DIOR seed-42 matrix: final README

Scope: 9 methods (A, B, E, PETS, SFUT, SFYOLO, C, D, F) x 21 DIOR domains = 189 cells, seed 42. The 21 domains are clean, cloudy, brightness, contrast, defocus_blur, elastic_transform, fog, frost, gaussian_blur, gaussian_noise, glass_blur, impulse_noise, jpeg_compression, motion_blur, pixelate, saturate, shot_noise, snow, spatter, speckle_noise, zoom_blur. SFYOLO covers all 21 domains. An earlier six-domain restriction (SFYOLO on defocus, gaussian, glass, impulse, shot and speckle only) was a worker error; the 15 missing SFYOLO cells were added on 2026-11-05 as new cells, not reruns, and the first 174 queue lines are unchanged.

Files: detr_dior_matrix_final.csv (189 rows, one per method x domain, same columns as the RSAR final CSV), detr_dior_matrix_summary.csv (method x kind x metric with mean, sample std and n), and this README. The per-cell result jsons are in metrics_json/.

Source: ARS-DETR DIOR seed-42 source, epoch 72 (sha256 7cb9dbe0…, test AP50 0.5400), staged and sha-verified on each host.

Metrics: AP50 and mAP50-95 come from tools/eval_map50_95.py run on each cell's predictions.pkl on its producing host, CPU only (CUDA_VISIBLE_DEVICES empty). The AP50 column is the tool's value at IoU 0.50. It is compared with the test.py full-precision mAP written in each cell's eval.log (the 'mAP' field of the OrderedDict). On all 189 cells the difference is exactly 0 (max |delta| = 0.0, 0 cells with delta > 1e-6). The AP50_json column is the rounded AP50 value that test.py stores in eval_*.json and eval.log ('AP50', 2 decimals), so it differs from AP50 by up to about 0.0005 by rounding. Each pkl must contain 11738 results, which the tool enforces.

Summary: 21-domain = all 21 domains including clean; 20-corruption = the 20 corruptions excluding clean. Mean is rounded half-up to four decimals; std is the sample std (ddof=1) across domains for seed 42.

Mixed hardware (gpu_type is written to every done marker and is in the CSV per cell):

| host | gpu_type | cells |
|---|---|---|
| <other-host> | RTX 4090D 48GB | 14 |
| <other-host> | RTX 4090D 48GB | 58 |
| gpu-138 | RTX A6000 48GB | 69 |
| gpu-67 | A100 64GB | 48 |

Per method, the cells are split across hardware (methods with cells on more than one GPU type are mixed):

- A: RTX 4090D 48GB 11, RTX A6000 48GB 10
- B: RTX 4090D 48GB 6, RTX A6000 48GB 15
- E: RTX 4090D 48GB 5, RTX A6000 48GB 16
- PETS: A100 64GB 7, RTX 4090D 48GB 10, RTX A6000 48GB 4
- SFUT: A100 64GB 9, RTX 4090D 48GB 7, RTX A6000 48GB 5
- SFYOLO: A100 64GB 8, RTX 4090D 48GB 6, RTX A6000 48GB 7
- C: A100 64GB 8, RTX 4090D 48GB 9, RTX A6000 48GB 4
- D: A100 64GB 8, RTX 4090D 48GB 9, RTX A6000 48GB 4
- F: A100 64GB 8, RTX 4090D 48GB 9, RTX A6000 48GB 4

Cross-hardware check on DIOR: none. DIOR has no cross-hardware replicate. The B clean cell is a single cell on gpu-138 (RTX A6000 48GB), AP50 0.526574969291687 (B clean DIOR = 0.526575). The gate 0.561453 (<other-host>, RTX 4090D) vs 0.568461 (gpu-138, RTX A6000), delta = -0.0070, is an RSAR B-clean replicate pair; its two replicate runs were stopped by the user, and it is not a DIOR result and is not in this matrix. Cross-hardware agreement on DIOR therefore remains unmeasured. gpu_type in every row is the audit trail.

CGA (C, D, F): the fixed code code-cgafix2 is used, with cga.py sha256 5cd1ed64ac880643959e2b2fe67ea1fd6991af9fe5f5f68552decafd0a5bac22. These are run as-is with the same thresholds and no tuning. The run-level CGA statistics are computed from the diag_window lines in each train.log. Each diag_window line covers one window of calls (duplicate lines from the logger are removed), and the windows are summed over the run. keep = 1 - dropped/total; agree_frac = agree/total; blended_frac = blended/total; mean_label_prob is the total-weighted mean of the per-window values. The per-cell values are in the table below and in the CSV notes column. Per-method ranges over the 21 domains:

- C: keep 1.0000–1.0000; agree_frac 0.1670–0.6535; blended_frac 0.3334–0.8330; mean_label_prob 0.1749–0.5878
- D: keep 0.9899–1.0000; agree_frac 0.0000–0.0332; blended_frac 0.9643–1.0000; mean_label_prob 0.0101–0.0660
- F: keep 0.9894–1.0000; agree_frac 0.0000–0.0162; blended_frac 0.9788–1.0000; mean_label_prob 0.0093–0.0410

keep is 1.0 for every C cell because no C window drops a candidate. For D and F, drops are rare (204 dropped out of 38662 candidates for D, 205 out of 41985 for F, summed over 21 cells each), blending applies to almost every candidate (blended_frac 0.96–1.0), so agree_frac is near 0 and keep stays near 1. The DIOR and RSAR runs use the same cga.py, the same D/F environment (veto_soft, VETO_PRED_THR 0.7, VETO_LABEL_THR 0.1, PROTECT_DET_SCORE 0.9, BLEND_DET_WEIGHT 0.7) and the same diag_window counters, so these definitions are comparable. Recomputed on RSAR with the same whole-run definition (diag_window windows summed over the run, duplicates removed; the RSAR CSV notes do not cite keep rates): C keep 1.0 (dropped 0), blended_frac 0.416–0.752, mean_label_prob 0.274–0.358; D keep 0.361–0.671, blended_frac 0.325–0.563, agree_frac 0.009–0.108, mean_label_prob 0.014–0.120; F keep 0.365–0.675, blended_frac 0.330–0.567, agree_frac 0.011–0.108, mean_label_prob 0.016–0.118 (8 domains per method). Total drops: RSAR D 6171 and F 6159 over 8 cells each; DIOR D 204 and F 205 over 21 cells each. The RSAR README's earlier 23–28% keep figure came from a single last window and is withdrawn. The much lower DIOR drop rate is observed, but its cause is not verified. A plausible explanation is that SARCLIP's top-1 probability rarely reaches VETO_PRED_THR 0.7 on DIOR optical images; this is not verified, because pred_prob is not logged.

| domain | C keep / agree / blended / mlp | D keep / agree / blended / mlp | F keep / agree / blended / mlp |
|---|---|---|---|
| clean | 1.0000 / 0.6535 / 0.3334 / 0.5878 | 0.9914 / 0.0113 / 0.9801 / 0.0199 | 0.9914 / 0.0126 / 0.9788 / 0.0199 |
| cloudy | 1.0000 / 0.6339 / 0.3522 / 0.5767 | 0.9908 / 0.0118 / 0.9790 / 0.0197 | 0.9919 / 0.0108 / 0.9811 / 0.0197 |
| brightness | 1.0000 / 0.6375 / 0.3499 / 0.5757 | 0.9975 / 0.0068 / 0.9907 / 0.0187 | 0.9968 / 0.0083 / 0.9884 / 0.0185 |
| contrast | 1.0000 / 0.6247 / 0.3624 / 0.5626 | 0.9933 / 0.0118 / 0.9816 / 0.0192 | 0.9930 / 0.0118 / 0.9812 / 0.0191 |
| defocus_blur | 1.0000 / 0.4779 / 0.4891 / 0.4105 | 0.9986 / 0.0083 / 0.9904 / 0.0188 | 0.9986 / 0.0090 / 0.9897 / 0.0187 |
| elastic_transform | 1.0000 / 0.4330 / 0.5583 / 0.3816 | 0.9899 / 0.0010 / 0.9889 / 0.0133 | 0.9894 / 0.0010 / 0.9885 / 0.0135 |
| fog | 1.0000 / 0.6235 / 0.3586 / 0.5564 | 0.9969 / 0.0174 / 0.9795 / 0.0234 | 0.9975 / 0.0161 / 0.9814 / 0.0234 |
| frost | 1.0000 / 0.4582 / 0.5339 / 0.4063 | 0.9990 / 0.0138 / 0.9852 / 0.0227 | 0.9990 / 0.0147 / 0.9842 / 0.0225 |
| gaussian_blur | 1.0000 / 0.5074 / 0.4668 / 0.4480 | 0.9987 / 0.0115 / 0.9872 / 0.0205 | 0.9993 / 0.0119 / 0.9874 / 0.0200 |
| gaussian_noise | 1.0000 / 0.3090 / 0.6910 / 0.2953 | 0.9971 / 0.0151 / 0.9820 / 0.0201 | 0.9970 / 0.0162 / 0.9808 / 0.0205 |
| glass_blur | 1.0000 / 0.4473 / 0.4758 / 0.3925 | 1.0000 / 0.0000 / 1.0000 / 0.0410 | 1.0000 / 0.0000 / 1.0000 / 0.0410 |
| impulse_noise | 1.0000 / 0.2958 / 0.7039 / 0.2742 | 0.9975 / 0.0081 / 0.9893 / 0.0163 | 0.9964 / 0.0092 / 0.9872 / 0.0163 |
| jpeg_compression | 1.0000 / 0.5544 / 0.4395 / 0.5144 | 0.9927 / 0.0112 / 0.9816 / 0.0176 | 0.9927 / 0.0111 / 0.9816 / 0.0177 |
| motion_blur | 1.0000 / 0.5028 / 0.4667 / 0.4378 | 0.9941 / 0.0135 / 0.9806 / 0.0281 | 0.9949 / 0.0143 / 0.9806 / 0.0283 |
| pixelate | 1.0000 / 0.6331 / 0.3644 / 0.5697 | 0.9987 / 0.0108 / 0.9879 / 0.0222 | 0.9987 / 0.0121 / 0.9866 / 0.0232 |
| saturate | 1.0000 / 0.6210 / 0.3655 / 0.5594 | 0.9916 / 0.0111 / 0.9805 / 0.0230 | 0.9917 / 0.0119 / 0.9798 / 0.0231 |
| shot_noise | 1.0000 / 0.1670 / 0.8330 / 0.1749 | 1.0000 / 0.0000 / 1.0000 / 0.0187 | 1.0000 / 0.0000 / 1.0000 / 0.0187 |
| snow | 1.0000 / 0.3066 / 0.6867 / 0.2867 | 1.0000 / 0.0000 / 1.0000 / 0.0660 | 1.0000 / 0.0032 / 0.9968 / 0.0094 |
| spatter | 1.0000 / 0.4172 / 0.5806 / 0.3866 | 0.9975 / 0.0025 / 0.9950 / 0.0101 | 0.9975 / 0.0030 / 0.9945 / 0.0103 |
| speckle_noise | 1.0000 / 0.2630 / 0.7351 / 0.2634 | 1.0000 / 0.0000 / 1.0000 / 0.0105 | 0.9986 / 0.0139 / 0.9847 / 0.0234 |
| zoom_blur | 1.0000 / 0.4612 / 0.4991 / 0.3715 | 0.9974 / 0.0332 / 0.9643 / 0.0448 | 1.0000 / 0.0000 / 1.0000 / 0.0093 |

TAM provenance: all 21 DIOR tam.pth files (one per domain, seed 42) carry training_code_sha 84a1aeb, and their target-image set is target VAL only. train_tam.py fits the TAM on the target VAL images with no annotations or GT-presence filtering (discover_target_images), so the fit does not read target labels. The six domains that overlap the R-CNN group (defocus_blur, gaussian_noise, glass_blur, impulse_noise, shot_noise, speckle_noise) have sha256 equal to the R-CNN group's tam_runs_84a1aeb seed-42 files for the same domains. Each tam.pth records image_manifest = /home/zechuan/iraod_jobs/ext_runs/TAM/<domain>/image_manifest.json, which does not exist on any host checked (<other-host> for all 21 files; <other-host>, gpu-138 and gpu-67 for brightness); so the manifest that the fit used is not on record. An earlier note that only six SFYOLO domains had a usable TAM was a worker error: all 21 TAM files exist and are used, and the protocol requires SFYOLO on all 21 domains.

Results (seed 42, mean and sample std across domains):

| method | 21-domain AP50 mean (std) | 21-domain mAP50-95 mean (std) | 20-corruption AP50 mean (std) | 20-corruption mAP50-95 mean (std) |
|---|---|---|---|---|
| A | 0.3351 (0.1382) | 0.2084 (0.0835) | 0.3249 (0.1333) | 0.2022 (0.0804) |
| B | 0.3442 (0.1304) | 0.2056 (0.0761) | 0.3351 (0.1267) | 0.2006 (0.0744) |
| E | 0.3442 (0.1311) | 0.2058 (0.0748) | 0.3354 (0.1280) | 0.2011 (0.0736) |
| PETS | 0.3276 (0.1442) | 0.2044 (0.0869) | 0.3172 (0.1396) | 0.1981 (0.0841) |
| SFUT | 0.3531 (0.1344) | 0.2181 (0.0815) | 0.3433 (0.1300) | 0.2120 (0.0787) |
| SFYOLO | 0.3606 (0.1130) | 0.2208 (0.0687) | 0.3528 (0.1099) | 0.2159 (0.0666) |
| C | 0.2823 (0.1311) | 0.1692 (0.0758) | 0.2725 (0.1263) | 0.1636 (0.0732) |
| D | 0.1091 (0.0493) | 0.0685 (0.0313) | 0.1056 (0.0478) | 0.0663 (0.0303) |
| F | 0.1186 (0.0649) | 0.0742 (0.0388) | 0.1122 (0.0595) | 0.0703 (0.0354) |

Process notes (all 189 counted cells are rc=0 with eval_exit=0; infra failures are never counted):
- SARCLIP default-path gap on <other-host>: all 20 E cells except E clean had one infra attempt (attempt a1) on <other-host> (the E/F configs hard-code vlst_pretrained to <private-root>/sarclip/ViT-B-32/vit_b_32_model.safetensors, which was missing on <other-host>; the cell script re-exports SARCLIP_PRETRAINED from SARCLIP_PRETRAINED_PATH, so the environment variable was not the cause). A symlink to the identical checkpoint (sha 5cb1fd24…) was added; the 20 cells were requeued: the 16 E cells that finished on gpu-138 and the 4 that finished on <other-host> (fog, frost, motion_blur, speckle_noise) each had that single a1 attempt; E clean finished on <other-host> with no infra attempt. The notes column records each cell's host and attempts.
- VGG base-path gap: the SFYOLO config points at <BASE>/third_party/vgg16_oxford/encoder_conv1_1_to_conv4_1.pt. It was missing on <other-host> and <other-host> and was placed there (sha 158e3923…). SFYOLO gaussian_noise had one infra attempt (a1) on <other-host>; SFYOLO defocus_blur, glass_blur, impulse_noise, shot_noise and speckle_noise each had a <other-host> a1 attempt and a <other-host> attempt; speckle_noise had two <other-host> attempts.
- Five SFYOLO cells (defocus_blur, glass_blur, impulse_noise, shot_noise, speckle_noise) hit INFRA_FAILED_GIVEUP on <other-host> and were requeued as new lock dirs: defocus_blur, shot_noise and speckle_noise finished on <other-host>; glass_blur and impulse_noise finished on gpu-138.
- Claims: each GPU lane ran a cell only while its agent-dispatch claim was held. Claims on idle lanes were released by the claim loop after their last cell had ended; the lane logs show no cell started on a GPU after its claim was released.
- The worker's INFRA_FAILED_OOM label applies to any non-zero exit, not only OOM; counting is unaffected.
