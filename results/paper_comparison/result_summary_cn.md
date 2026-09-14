# IRAOD 最终三种子比较

本交付对应 130/130 新训练和 192/192 native-ID TEST 评测终态；A 为每数据集固定 source，不复制为三种子。B–F 为独立 clean/腐蚀适配后的 final EMA。

OBB / VOC AP50，Source-Free；目标标签仅用于离线评测。Source-available、target-supervised oracle 和额外消融未运行。B/D/E/F 表仅复用已完成组件组合，不是新增实验。

## 完整原始数据

raw_results.csv：192 单元；per_class_summary.csv：2048 类别 AP 行（保留打印精度）；per_corruption_summary.csv：含 clean 的逐域均值与样本标准差；statistical_tests.csv：配对种子检验；recovery.csv：逐种子逐腐蚀恢复比。
以下表中 AP、mPC 和 ΔA 使用 0–1 单位，rPC_percent 使用百分比；LaTeX 主表将 AP 和 ΔA 转成百分点显示。

## 结果

| 数据集 | 方法 | 指标 | mean ± sample std |
| --- | --- | --- | --- |
| RSAR | A | mPC | 0.31714159 ± N/A（固定 source） |
| RSAR | A | delta_A | 0 ± N/A（固定 source） |
| RSAR | A | rPC_percent | 59.680413 ± N/A（固定 source） |
| RSAR | A | clean_mAP50 | 0.53139979 ± N/A（固定 source） |
| RSAR | B | mPC | 0.32397348 ± 0.0013170707 |
| RSAR | B | delta_A | 0.0068318964 ± 0.0013170707 |
| RSAR | B | rPC_percent | 60.966054 ± 0.2478493 |
| RSAR | B | clean_mAP50 | 0.52444673 ± 0.0043453706 |
| RSAR | C | mPC | 0.3308357 ± 0.0038866089 |
| RSAR | C | delta_A | 0.013694112 ± 0.0038866089 |
| RSAR | C | rPC_percent | 62.257401 ± 0.73139075 |
| RSAR | C | clean_mAP50 | 0.52124306 ± 0.0095560208 |
| RSAR | D | mPC | 0.30551394 ± 0.0040151413 |
| RSAR | D | delta_A | -0.011627641 ± 0.0040151413 |
| RSAR | D | rPC_percent | 57.492297 ± 0.75557826 |
| RSAR | D | clean_mAP50 | 0.51036712 ± 0.00074412627 |
| RSAR | E | mPC | 0.3297109 ± 0.0014902687 |
| RSAR | E | delta_A | 0.012569318 ± 0.0014902687 |
| RSAR | E | rPC_percent | 62.045735 ± 0.2804421 |
| RSAR | E | clean_mAP50 | 0.52706116 ± 0.0015745292 |
| RSAR | F | mPC | 0.29553531 ± 0.0037835495 |
| RSAR | F | delta_A | -0.021606272 ± 0.0037835495 |
| RSAR | F | rPC_percent | 55.614496 ± 0.7119968 |
| RSAR | F | clean_mAP50 | 0.5146666 ± 0.0036590583 |
| DIOR | A | mPC | 0.2711852 ± N/A（固定 source） |
| DIOR | A | delta_A | 0 ± N/A（固定 source） |
| DIOR | A | rPC_percent | 91.958405 ± N/A（固定 source） |
| DIOR | A | clean_mAP50 | 0.29489985 ± N/A（固定 source） |
| DIOR | B | mPC | 0.27548735 ± 0.00084214622 |
| DIOR | B | delta_A | 0.0043021474 ± 0.00084214622 |
| DIOR | B | rPC_percent | 93.417255 ± 0.28557024 |
| DIOR | B | clean_mAP50 | 0.29810867 ± 0.00097657147 |
| DIOR | C | mPC | 0.27542467 ± 7.5070578e-05 |
| DIOR | C | delta_A | 0.0042394731 ± 7.5070578e-05 |
| DIOR | C | rPC_percent | 93.396003 ± 0.025456296 |
| DIOR | C | clean_mAP50 | 0.2967183 ± 0.0010834295 |
| DIOR | D | mPC | 0.26851475 ± 0.00057953717 |
| DIOR | D | delta_A | -0.0026704487 ± 0.00057953717 |
| DIOR | D | rPC_percent | 91.052861 ± 0.19652 |
| DIOR | D | clean_mAP50 | 0.29938016 ± 0.00078019263 |
| DIOR | E | mPC | 0.27575486 ± 0.00053378095 |
| DIOR | E | delta_A | 0.0045696629 ± 0.00053378095 |
| DIOR | E | rPC_percent | 93.507969 ± 0.18100414 |
| DIOR | E | clean_mAP50 | 0.29951511 ± 0.0020072738 |
| DIOR | F | mPC | 0.26862463 ± 0.0018563377 |
| DIOR | F | delta_A | -0.0025605725 ± 0.0018563377 |
| DIOR | F | rPC_percent | 91.09012 ± 0.62948073 |
| DIOR | F | clean_mAP50 | 0.29772091 ± 0.0026828139 |

## 主要观察与统计限制

1. RSAR：C 的观察 mPC 均值最高，为 33.0836%。F 相对固定 A 的配对平均增益为 -2.1606 个百分点。最高均值不等于方法间差异显著；不因负结果删除或重调方法。
2. RSAR 的 ΔA 双侧 t 检验 Holm p（B–F）：B=0.0402699、C=0.0516306、D=0.0516306、E=0.0232656、F=0.0402699。这些是固定 source 条件下、依赖差值正态假设的 n=3 检验，不能与无分布假设的证据强度混同。
3. DIOR：E 的观察 mPC 均值最高，为 27.5755%。F 相对固定 A 的配对平均增益为 -0.2561 个百分点。最高均值不等于方法间差异显著；不因负结果删除或重调方法。
4. DIOR 的 ΔA 双侧 t 检验 Holm p（B–F）：B=0.0375993、C=0.000522512、D=0.0375993、E=0.0180695、F=0.139461。这些是固定 source 条件下、依赖差值正态假设的 n=3 检验，不能与无分布假设的证据强度混同。
5. ΔA 精确双侧 sign-flip 最小原始 p=0.25；最小 Holm p=1。仅 8 种符号排列，低功效；本次不据此宣称分布无关的显著提升。腐蚀域不是独立重复样本，置信区间仅反映适配随机性。

## 指标定义

1. 先在种子内平均全部腐蚀，再跨种子统计；std 使用 ddof=1。n=3、低功效，置信区间仅反映固定 source 条件下的适配随机性。
2. 历史 rPC=100×mPC/固定 A clean TEST；不是 source VAL，也不改成 method clean。method_clean_normalized_percent 另列。
3. Recovery=(method−A_corruption)/(A_clean_TEST−A_corruption)。分母非正时 N/A，接近零时不稳定；保留负结果，不截断、不排名。

## 证据与边界

132 ROI groups / 1,267,816 image-role records / 18,468,211 detections；3,520 固定子集可视化，24 joint embeddings。定性覆盖仍为 seed42，不冒充三种子的 ROI 导出。复用已接受 streaming 审计，不重扫百万 NPZ。
原始 seed42 文件逐字节保存在 historical/seed42；producer_metadata_audit.csv 保留 recorded/actual source producer 更正，原 sidecar 不变。
完整逐图 prediction/ROI 清单与大文件位置见 artifact_manifest.json；源码、checkpoint、实际 ID sidecar 和 class AP 路径见 raw_results.csv、checkpoints.json。

复现入口：experiments/comparison/reproduction_commands.sh。本交付不声称未运行的 baseline/oracle 已完成。
