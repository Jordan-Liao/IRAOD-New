# IRAOD 最终比较交付

主报告：[`IRAOD_full_comparison_report_cn.docx`](IRAOD_full_comparison_report_cn.docx)。
RSAR 42/42 和 DIOR-R 18/18 数值完成；A 是 source，B–F 是最后迭代 EMA。
RSAR 数值固定沿用 `b8527464c0201efb195d1f311f82bd86fb4976ca`。

| 项 | DIOR 结果 / 索引 |
|---|---|
| Source clean val / test | 0.40987735986709595 / 0.2948998510837555（0.410 / 0.295） |
| mPC A/B/C/D/E/F | 0.271 / 0.275 / 0.275 / 0.268 / 0.276 / 0.271 |
| raw / summary / LaTeX | `dior_raw_results.csv`、`dior_per_corruption.csv`、`dior_summary_metrics.csv`、`dior_main_table.tex` |
| 原始评估定位 | `dior_evaluation_manifest.csv`：20 条 eval JSON / config / checkpoint |
| 逐类别 | `dior_per_class.csv`：400 行（20 次评估 × 20 类）；类 AP 保留三位小数 |
| 检测图 | `dior_visualization_files.csv`：288 条逐文件记录；原始目录汇总仍保留 `dior_visualization_manifest.csv` |
| 同图选择 | `dior_visualization_selection.json`：test 前 16 个 ID 11726–11741，A–F 完全相同 |
| RoI | `dior_roi_manifest.csv`：33 组，528 个 NPY；远端 index.json 含 image_id / n_roi / preds |
| 审计 / 忠实移植边界 | `dior_audit.md`、`dior_checkpoint_manifest.json`、`dior_ports_executability.md` |
| 源与运行快照 | `../../experiments/comparison/dior_source/`、`../../experiments/comparison/dior_recovery/` |

二进制远端根（gpu-67）：
`/mnt/shared/zechuan/iraod_artifacts/comparison/deliverables/results/paper_comparison/`。
不提交 checkpoint、数据、全量图片/特征或日志；本报告不嵌图，避免挑图与仓库膨胀。

## 解释与限制

RSAR C 的 mPC 最高（0.3336），DIOR E 最高（0.2757），完整 F 在两数据集的 mean Δada 均为负。
单 seed=42，std / 显著性 N/A；rPC 使用各数据集 source clean **TEST**。
mean recovery 在小分母下不稳定，不排名；DIOR cloudy/contrast 均被标记 unstable。

RoI 虽在同一 `fc_cls` 输入位置提取，但保存的是 pre-NMS feature 和 post-NMS preds，
没有逐行映射。不能直接用于按预测类别着色的实例 t-SNE；33 组文件完成不是该分析已就绪。
RSAR 既有 per-class 只有 78 行，定性 manifest 是占位条目；本次没有扩展为全矩阵定性覆盖。
这些限制说明原始计划的扩展字段并非全部完成，不能用数值矩阵完成替代。

IRG/LPLD/DRU/AASFOD/SF-YOLO 没有经过验证的忠实 OBB 移植，均 N/A；理由逐方法记录，
不填写 HBB/UDA/其他数据集数字。Simple-SFOD/SF-UT **等价 B 已跑**，不另计独立官方复现。

## Recovery commands

从仓库根目录按需执行（本次集成未启动 GPU）：

```bash
ssh gpu-67 'bash -s -- cell B brightness 4' < results/paper_comparison/dior_recovery_commands.sh
ssh gpu-67 'bash -s -- matrix 4 4,5' < results/paper_comparison/dior_recovery_commands.sh
ssh gpu-67 'bash -s -- source 4' < results/paper_comparison/dior_recovery_commands.sh
```

脚本调用原远端 `queue_0f98a48`，保留现有 skip-if-done 行为：
训练匹配任意 `iter_*_ema.pth`，评估查看 `eval_exit=0`，RoI 查看 `index.json` 存在。
因此这不是通用的任意中断点恢复器；遇到中断/部分输出须先确认 `iter_185_ema.pth`
与成功状态，再人工隔离残缺产物后恢复。不得把中间 checkpoint 当作最终结果或按测试分数挑选。
快照保留绝对路径，仅供原远端环境复现，不宣称本地训练可运行。

## 重新生成 DOCX

使用已有包含 `python-docx` 的 Python，不新增依赖：

```bash
/path/to/existing-docx-env/bin/python tools/build_full_comparison_report.py
```

生成器只读取已版本化 CSV 并写入 DOCX；不读取用户主 checkout 的
`对比实验的plan.docx`，不加载远端二进制。原始字段依据用户指定的
`对比实验的plan.original.docx` 对照，原始计划不提交。
