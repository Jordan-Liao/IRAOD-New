# DIOR-R strict A–F 结果（seed=42，单种子）

Source clean TEST mAP50 = 0.2949；val = 0.4099。
按三位小数分别为 TEST 0.295 / val 0.410；mPC A/B/C/D/E/F = 0.271/0.275/0.275/0.268/0.276/0.271。
mPC=brightness/cloudy/contrast 算术平均；rPC=mPC/clean×100%。Δada=方法−A（同腐蚀）。
**mean recovery 不用于排名。** cloudy 上 $A_{clean}-A_{corr}\approx0.005$，微小 Δada 会被放大。

| 方法 | mPC | rPC | mean Δada | mean recovery | 稳定性 |
|---|---:|---:|---:|---:|---|
| A | 0.2712 | 92.0% | +0.0000 | +0.000 | cloudy:unstable;contrast:unstable；N/A_single_seed |
| B | 0.2755 | 93.4% | +0.0043 | +0.590 | cloudy:unstable;contrast:unstable；N/A_single_seed |
| C | 0.2754 | 93.4% | +0.0043 | +0.400 | cloudy:unstable;contrast:unstable；N/A_single_seed |
| D | 0.2681 | 90.9% | -0.0031 | -0.366 | cloudy:unstable;contrast:unstable；N/A_single_seed |
| E | 0.2757 | 93.5% | +0.0045 | +0.622 | cloudy:unstable;contrast:unstable；N/A_single_seed |
| F | 0.2707 | 91.8% | -0.0005 | -0.152 | cloudy:unstable;contrast:unstable；N/A_single_seed |

逐格 AP50：
| 腐蚀 | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| brightness | 0.242 | 0.245 | 0.247 | 0.240 | 0.245 | 0.242 |
| cloudy | 0.290 | 0.297 | 0.294 | 0.287 | 0.298 | 0.288 |
| contrast | 0.282 | 0.285 | 0.285 | 0.278 | 0.285 | 0.282 |

- 单 seed，禁止写成显著/SOTA。
- B/C/E 相对 A 为小幅正 Δ；D 三格均略负；F 近似 A。
- 适应：Corruption val 图像-only 5863；评估：Corruption test + ImageSets/test.txt OBB GT。
- 无 target GT / LoRA；final EMA `iter_185_ema.pth`。
- 18/18 腐蚀单元格（A 为 source，B–F 为 final EMA），加 clean val/test 共 20 次评估；per-class 400 行。
- 固定 test 前 16 图（11726–11741），288 张可视化，33 组 RoI / 528 个特征文件。详见逐图及 RoI manifests。
- RoI 的 pre-NMS feature 与 post-NMS preds 无逐行映射，不能直接作为带预测类别的实例 t-SNE 输入。
- 最终中文 DOCX 与完整产物索引见 `full_comparison_delivery_cn.md`；未运行外部忠实 OBB port，Simple-SFOD/SF-UT 以等价 B 已跑。
