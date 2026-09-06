# RSAR 七 corruption 汇总（seed=42，单种子，无 std）

Source clean mAP50 = 0.5314。mPC=七 corruption 算术平均；rPC=mPC/clean×100%。Δada=方法−A（同 corruption）。

| 方法 | mPC | rPC | mean Δada | mean recovery |
|---|---:|---:|---:|---:|
| A Source Only | 0.3171 | 59.7% | +0.0000 | +0.000 |
| B ST | 0.3239 | 61.0% | +0.0068 | -0.515 |
| C CLIP-CGA | 0.3336 | 62.8% | +0.0165 | -0.317 |
| D SARCLIP-CGA | 0.3067 | 57.7% | -0.0105 | -0.951 |
| E VLST | 0.3314 | 62.4% | +0.0142 | -0.240 |
| F CGA+VLST | 0.2974 | 56.0% | -0.0197 | -0.476 |

- 强干扰（noise/smart/am）上 C 与 E 常为正 Δada；F 多数为负。
- 弱干扰（chaff/pt/gauss）上 B–F 相对 A 多为小幅负迁移。
- 单 seed，禁止写成显著/SOTA。IRG/LPLD 等忠实 OBB 移植仍为 N/A；DIOR 18/18 已完成，见 `dior_result_summary_cn.md`。

最终中文报告：`IRAOD_full_comparison_report_cn.docx`；完整交付索引与限制见 `full_comparison_delivery_cn.md`。本文件全部 RSAR 数值保持 b852746 不变。
- **mean recovery 不用于排名。** recovery=(mAP_m−mAP_A)/(mAP_A_clean−mAP_A)。chaff/point_target/gaussian 上 (clean−A) 很小，微小负 Δada 会被放大成大负 recovery（与 mean Δada 符号可相反）。主表只用 mPC、rPC、逐格 Δada。
