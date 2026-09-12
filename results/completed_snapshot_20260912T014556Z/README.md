# IRAOD 已完成结果快照

**截至 2026-09-12 01:45:56 UTC（北京时间 09:45:56）：912 个批准的 TEST 角色单元中，671 个有可纳入统计的原生 TEST 指标；540 个适配 TRAIN 单元中，483 个已终结且保留冻结最终 Student/EMA，其中 5 个是已知 NaN 模型。快照整理完成不代表整个实验计划完成。**

仅只读采集 `.67/.221/.183/.134` 的既有产物；没有训练、推理、补评估、改锁或停止任务。截止时间之后的完成项单列，不回填主表。文件名中的日期不是完成时间。

## 1. 全家族覆盖

TRAIN 表示终结状态与冻结最终双权重同时存在，不等于 TEST 已评估，也不保证模型数值健康。A 是 **2 个 source 模型**在 12 个域的复用，不是 12 次训练；Student 是同一次 TRAIN 的权重角色，不能再算一次训练。

| 方法 | RSAR TRAIN | DIOR TRAIN | RSAR TEST EMA / Student | DIOR TEST EMA / Student | 边界 |
|---|---:|---:|---:|---:|---|
| A / Source | 1/1 源模型 | 1/1 源模型 | source 8/8 | source 4/4 | 固定 seed42 |
| B / ST | 24/24 | 12/12 | 24/24 / 24/24 | 12/12 / 12/12 | 共同 1 epoch |
| C / CLIP-CGA | 24/24 | 12/12 | 24/24 / 24/24 | 12/12 / 12/12 | 共同 1 epoch |
| D / SARCLIP-CGA | 24/24 | 12/12 | 24/24 / 24/24 | 12/12 / 12/12 | 共同 1 epoch |
| E / VLST | 24/24 | 12/12 | 24/24 / 24/24 | 12/12 / 12/12 | 共同 1 epoch |
| F / CGA+VLST | 24/24 | 12/12 | 24/24 / 24/24 | 12/12 / 12/12 | 共同 1 epoch |
| IRG | 24/24 | 12/12 | 22/24 / 22/24 | 12/12 / 12/12 | EMA/Student 各缺 2 个 NaN 域种子 |
| LPLD | 24/24 | 12/12 | 22/24 / 22/24 | 12/12 / 12/12 | EMA/Student 各缺 2 个 NaN 域种子 |
| SFUT | 24/24 | 12/12 | 23/24 / 23/24 | 12/12 / 12/12 | EMA/Student 各缺 1 个 NaN 域种子 |
| AASFOD | 24/24 | 12/12 | 9/24 / 9/24 | 0/12 / 0/12 | TSD + 披露的分阶段端口 |
| B_REG | 24/24 | 12/12 | 0/24 / 未要求 | 12/12 / 未要求 | Student 保留，未自动扩张评估范围 |
| F_text_only | 24/24 | 12/12 | 19/24 / 未要求 | 12/12 / 未要求 | Student 保留，未自动扩张评估范围 |
| F_veto_only | 24/24 | 12/12 | 20/24 / 未要求 | 12/12 / 未要求 | Student 保留，未自动扩张评估范围 |
| SFYOLO | 15/24 | 12/12 | 0/24 / 0/24 | 0/12 / 0/12 | 2 epoch；TAM 前置 |
| LoRA-CGA | 3/24 | 9/12 | 0/24 / 未要求 | 0/12 / 未要求 | 监督 LoRA Oracle 附录 |
| LoRA-CGA+VLST | 0/24 | 12/12 | 0/24 / 未要求 | 0/12 / 未要求 | 监督 LoRA Oracle 附录 |

| 单元 | 已完成 / 批准总量 | 说明 |
|---|---:|---|
| B_REG / F_text_only / F_veto_only TRAIN | 108/108 | TEST EMA 仍仅 75/108；此前 87 行报告含 12 个 source |
| IRG / LPLD / SFUT TRAIN | 108/108 终结 | 5 个已知 NaN；不是 108 个健康模型的证据 |
| AASFOD / TSD | 36/36 / 36/36 | AASFOD 已评估仅 RSAR 9 个模型 × 2 角色 |
| TAM | 9/12 | RSAR 5/8 + DIOR 4/4；每域 seed42 一次 fit，不乘 3 |
| LoRA adapter | 2/2 | RSAR、DIOR 各一个 TRAIN 监督的 10-epoch final |
| SFYOLO TRAIN | 27/36 | RSAR 15/24 + DIOR 12/12；没有原生 TEST 指标 |
| Oracle TRAIN | 24/72 | RSAR 3/48 + DIOR 21/24；没有原生 TEST 指标 |
| 完整 TEST ROI | 132/732 组 | 原 core 132 组保留；新增 600 组无输出，其中 10 组明确 held |
| 已有可视化 / embedding | 3520 / 24 | 原 core 已验收产物；不是新增扩展完成量 |
| DRU | 明确排除 | 缺少 source-trained decoder-depth predictions；不使用代理实现 |

**分母来源：** core 12 source + 180 B–F EMA；既有 B–F Student 180；5 个新端口 180 TRAIN × 2 TEST 角色；3 个消融 108 EMA；Oracle 72 EMA，合计 **912**。TRAIN 为 B–F 180 + 新端口 180 + 消融 108 + Oracle 72 = **540**，source 模型与前置单独列。ROI 为既有 core 132 + B–F seed43/44 新增 240 + 新端口 360 = **732**；新端口可视化 9600、联合 embedding 72，以及 B–F 新增可视化 6400、embedding 48 仍未完成。

完整单元级状态：[coverage.csv](coverage.csv)、[training_status.csv](training_status.csv)、[prerequisites.csv](prerequisites.csv)、[roi_coverage.csv](roi_coverage.csv)。

## 2. 实际原生 TEST 指标

**单位：下表为百分数**，由原始 `metric.mAP ∈ [0,1] × 100` 转换；Δ 为匹配同数据集、同域、同 source identity 的 **A/source/seed42** 的百分点差。EMA 与 Student 分开，不互相替代。

每个独立 seed 先对该数据集**全部域（含 clean）等权平均**，再计算 seed 均值和样本标准差 （ddof=1）。`n` 仅计全域完整的真实 seeds；未完整 seeds 的真实指标仍留在 CSV，但不进入该全域均值。因此 `n=1` 的 std 为 NA，source seed42 不伪装成 3 个重复；IRG/LPLD/SFUT 的 RSAR `n=2` 是缺失感知统计，不能与完整 3-seed 结果声称同等覆盖。下表不是显著性排名。

| 方法 / 角色 | RSAR mAP50% ± std (n) | Δ pp | 有效单元 | DIOR mAP50% ± std (n) | Δ pp | 有效单元 |
|---|---:|---:|---:|---:|---:|---:|
| A / source | 34.392 ± NA (1) | 0.000 | 8/8 | 27.711 ± NA (1) | 0.000 | 4/4 |
| B / ema | 34.903 ± 0.143 (3) | 0.511 | 24/24 | 28.114 ± 0.064 (3) | 0.403 | 12/12 |
| B / student | 19.630 ± 0.837 (3) | -14.763 | 24/24 | 26.002 ± 0.270 (3) | -1.709 | 12/12 |
| C / ema | 35.464 ± 0.459 (3) | 1.071 | 24/24 | 28.075 ± 0.024 (3) | 0.363 | 12/12 |
| C / student | 23.390 ± 0.291 (3) | -11.002 | 24/24 | 25.496 ± 0.247 (3) | -2.216 | 12/12 |
| D / ema | 33.112 ± 0.358 (3) | -1.280 | 24/24 | 27.623 ± 0.045 (3) | -0.088 | 12/12 |
| D / student | 19.482 ± 3.057 (3) | -14.910 | 24/24 | 19.703 ± 0.636 (3) | -8.008 | 12/12 |
| E / ema | 35.438 ± 0.148 (3) | 1.046 | 24/24 | 28.169 ± 0.025 (3) | 0.458 | 12/12 |
| E / student | 18.167 ± 0.769 (3) | -16.226 | 24/24 | 25.602 ± 0.768 (3) | -2.110 | 12/12 |
| F / ema | 32.293 ± 0.377 (3) | -2.100 | 24/24 | 27.590 ± 0.187 (3) | -0.122 | 12/12 |
| F / student | 16.391 ± 1.046 (3) | -18.001 | 24/24 | 18.427 ± 0.232 (3) | -9.285 | 12/12 |
| IRG / ema | 36.681 ± 0.033 (2) | 2.289 | 22/24 | 27.988 ± 0.092 (3) | 0.277 | 12/12 |
| IRG / student | 17.030 ± 0.910 (2) | -17.363 | 22/24 | 26.021 ± 0.127 (3) | -1.690 | 12/12 |
| LPLD / ema | 35.104 ± 0.247 (2) | 0.712 | 22/24 | 28.126 ± 0.124 (3) | 0.414 | 12/12 |
| LPLD / student | 15.653 ± 0.308 (2) | -18.740 | 22/24 | 26.307 ± 0.138 (3) | -1.404 | 12/12 |
| SFUT / ema | 35.713 ± 0.080 (2) | 1.321 | 23/24 | 27.867 ± 0.002 (3) | 0.155 | 12/12 |
| SFUT / student | 17.888 ± 0.302 (2) | -16.505 | 23/24 | 26.025 ± 0.115 (3) | -1.686 | 12/12 |
| AASFOD / ema | NA ± NA (0) | NA | 9/24 | NA ± NA (0) | NA | 0/12 |
| AASFOD / student | NA ± NA (0) | NA | 9/24 | NA ± NA (0) | NA | 0/12 |
| B_REG / ema | NA ± NA (0) | NA | 0/24 | 28.166 ± 0.047 (3) | 0.455 | 12/12 |
| F_text_only / ema | NA ± NA (0) | NA | 19/24 | 27.574 ± 0.044 (3) | -0.138 | 12/12 |
| F_veto_only / ema | 35.586 ± NA (1) | 1.193 | 20/24 | 28.171 ± 0.031 (3) | 0.460 | 12/12 |

`F_veto_only` 的 RSAR 全域均值只来自 1 个完整 seed；20 个已评估单元不是 20 个独立重复。`F_text_only` 的 RSAR 有 19 个真实单元，但没有一个全域完整 seed，因此全域均值为 NA。AASFOD 仅局部覆盖，不能用局部均值冒充全数据集结果。

### AASFOD 已评估的局部域（RSAR）

| 域 | 实际 seeds | source mAP50% | EMA mAP50% ± std | Student mAP50% ± std |
|---|---|---:|---:|---:|
| clean | 42;43;44 | 53.140 | 15.744 ± 2.090 | 10.813 ± 2.128 |
| chaff | 43;44 | 47.193 | 15.119 ± 1.953 | 8.885 ± 0.804 |
| gaussian_white_noise | 42;43;44 | 52.814 | 14.841 ± 1.765 | 11.486 ± 2.453 |
| point_target | 42 | 51.584 | 20.234 ± NA | 11.587 ± NA |

所有真实 per-domain/per-seed/per-class 数值（包含不完整家族）见 [raw_metrics.csv](raw_metrics.csv)、[per_domain.csv](per_domain.csv)、[per_seed.csv](per_seed.csv)、[per_class.csv](per_class.csv)、[summary.csv](summary.csv)。per-class AP 保留原 `class_ap.txt` 的打印精度（通常 3 位小数），不是重新计算的高精度 AP；原始 mAP 直接保留 JSON 浮点数。空 CSV 字段 = NA，绝不自动填 0。

### 可观察结论与限制

1. 完整 3-seed 的 RSAR EMA 中，C 为 **35.464 ± 0.459%**，对 source **+1.071 pp**；E 为 **35.438 ± 0.148%**。两者仅差 **0.026 pp**，不作显著性判断。IRG 的 **36.681 ± 0.033%** 只有 seeds42/43，不作为完整 3-seed 优胜结论。
2. F/EMA 对 source 的全域变化为 RSAR **−2.100 pp**、DIOR **−0.122 pp**；B–F 的 Student 结果与 EMA 差异明显，不能把“保留 Student”或“TRAIN 已完成”写成 EMA/TEST 的替代成绩。
3. DIOR 的 F_veto_only/EMA 为 **28.171 ± 0.031%（+0.460 pp）**；F_text_only/EMA 为 **27.574 ± 0.044%（−0.138 pp）**。RSAR 消融覆盖不完整，不外推同一结论；SFYOLO 和 Oracle 尚无 TEST，不给排名。

## 3. 预算、前置与未完成边界

| 分组 | 协议与前置 | 不允许的比较 |
|---|---|---|
| 严格共同 1 epoch | 固定 source，目标 VAL image-only，种子42/43/44；RSAR final iter266、DIOR iter185 | 不混入 source/GT/LoRA 信息 |
| AASFOD 披露端口 | 每模型 TSD，post-hoc Dropout、分阶段 alignment/FNS 与预算重标定 EMA | 不声称 Dropout 在 source 训练过，不忽略额外前置成本 |
| SFYOLO extended budget | 2 detector epochs；RSAR iter531、DIOR iter369；每域 TAM 160000 外循环 / 320000 更新 | 不与共同 1-epoch 方法等成本排名 |
| Oracle 附录 | dataset-specific TRAIN 监督 LoRA final，10 epochs、batch64、AdamW lr1e−4、rank8/alpha16；两种检测适配变体 | 不混入严格 image-only 方法，不把 adapter TRAIN accuracy 当 TEST mAP |

尚缺：**57 个 TRAIN**；TEST 中 **231 个无原生指标 + 5 个 Student held + 5 个已记录但无效的 NaN 零分**；TAM 的 RSAR `am_noise_horizontal / am_noise_vertical / smart_suppression` 尚无最终产物，既有 receipt 表明在 `.67` 重做完整 freshfit，不能把 `.183` 中断的 partial 算作完成。RSAR SFYOLO 剩余 9 格依赖这 3 个 TAM。

Oracle 的 RSAR 3 个截止前完成格实际是 `clean/42/LoRA-CGA`、`am_noise_horizontal/42/LoRA-CGA`、`am_noise_horizontal/43/LoRA-CGA`，**不是 clean 的 3 个 seeds**。DIOR 为 LoRA-CGA 9/12、+VLST 12/12。采集时看到 RSAR `am_noise_horizontal/44/LoRA-CGA` 于 **01:54:11 UTC** 完成，晚于截止，保留在 [post_cutoff.csv](post_cutoff.csv)，主计数不包含它。

## 4. 排除、失败和中断不能混称

| 类别 | 明确证据 / 范围 | 本快照处理 |
|---|---|---|
| 已知 NaN | RSAR seed44：IRG/LPLD × point_target；IRG/LPLD/SFUT × noise_suppression | TRAIN 终结但异常；5 个 Student TEST held，10 个 EMA/Student ROI held |
| 已存在的 NaN EMA 零分 | 上述 5 个 EMA checkpoint 经独立 CPU 检查，各有 195 个 state tensors 含 NaN，约占 state elements 的 99.30%；原生 evaluator 输出 0 | 仅 `native_mAP50/native_AP50` 保留原值；合格 `mAP50/AP50` 为 NA，不纳入均值；不新增正式 EMA hold，也不因低分排除 |
| 真实 48GB OOM | `.183`：SFYOLO 6 格；RSAR Oracle clean 5 格。旧 B_REG/F 48GB 失败另有终结证据 | 是容量失败，不是模型得分；新环境有效重试按同一逻辑格只计一次 |
| 基础设施 / 路径 / ABI | 旧运行与移交中的 canonical config、adapter 路径、依赖/ABI 问题；已修复后存在合法 final 的格 | 旧失败保留为历史，不扩大成科学排除，不按重试次数增加分母 |
| 定时人工中断 | 2026-09-10 01:00 UTC 的 `.221/.134` 人工 STOP | 与科学失败区分；`.134` 仍 paused，本报告只读 |
| STOP 后未授权尝试 | `.221` 于 01:13:41 UTC 自动重启，01:18:57 再次制止，01:22:48 确认停止 | 不视为授权延长预算或成功结果；后续新 START 单独授权 |
| 协调中断，不是人工 STOP | `.183` 上 3 个 RSAR TAM（约 50–52k partial）；SFYOLO gwn44 / noise_suppression42 | partial 无 final，不计完成；不冒充科学失败或“从未尝试” |

可定位的原生失败记录见 [attempts.csv](attempts.csv)。其行是物理历史尝试而非独立实验；逻辑格是否完成以 [training_status.csv](training_status.csv) 为准。Oracle clean44/+VLST 的既有输入竞争尝试无合法 final，也不能随其同架构 OOM 扩大记为第 6 个 OOM。

[ema_numerical_evidence.json](ema_numerical_evidence.json) 保存上述 5 个冻结 EMA 的逐模型 tensor/element 计数、原路径和文件大小。该复核在快照之后对既有 checkpoint 原地只读执行，`map_location="cpu"` 且禁用 CUDA；不是新增训练、TEST 或改变截止时间。Student 的异常未被用来推断 EMA 异常。

## 5. 溯源与复算

- [provenance.json](provenance.json) 记录批准 runtime manifests、source producer 修正、各主机采集时间、固定截止与精确排除角色。每个 TEST/TRAIN 单元有真实原路径及指标/终结时间。历史 accepted core、B–F Student、IRG/LPLD/SFUT EMA 的固定 checkpoint/eval 选择被保留，并对照现存原生指标文件；没有按最佳分数或最新 mtime 挑重试。
- 新归集的 `.221` 既有预测使用 `44ac421` 的 `report_inputs.prediction_evidence` 和 `class_table` 在远端单线程 CPU 检查 **232 个唯一评估目录**：完整 TEST image IDs / 顺序、预测数量与类别形状、有限检测框/分数、类别 AP。RSAR 每格 8538 张，DIOR 11738 张。结合原生 execution、eval_status、final checkpoint、source identity、配置与实际 evaluator `331d2131b84651f0a2930a3d53faeefad8701531`；没有新推理或 checkpoint tensor 加载。
- 复算以 CSV 为输入：先筛 `status=valid_native_test`，按 `(dataset, domain, method, seed, role)` 唯一键聚合；source 使用相同 dataset/domain 的 A/source/42。全域完整 seed 才进入 `summary.csv`，std 使用样本标准差；不做插值、零填补、显著性检验或跨预算排名。

原 core ROI/可视化/embedding 沿用其已验收的产物索引。全批准 ROI 范围为 **7,030,616 个 image roles**，其中既有 core 的 **1,267,816 个**完成；“full TEST”只描述已完成组内的图片覆盖，不代表全部 732 组完成。扩展 ROI 的 600 个批准输出位置在 `.67/.221` 均未出现输出目录；不能因模型完成而推定导出完成。权重、数据集、session stores 不属于普通 Git 结果交付。

## 6. 原始结果归档：私有百度网盘

大文件使用 **ByPy / 百度网盘**，不使用 GitHub Release。实际私有目录：

```text
/apps/bypy/IRAOD-New/results/completed_snapshot_20260912T014556Z
```

[artifact_manifest.json](artifact_manifest.json) 是完整归档清单，记录每个文件的大小、源 SHA256、源 MD5、百度返回的 MD5/大小、实际路径和验证时间。[validation_evidence.json](validation_evidence.json) 保留 676 个原生指标/类别 AP/图片数与冻结 CSV 的一致性，以及 1,278,342 条逐文件记录检查。百度列表的 `md5` 字段是混淆值，不能作为真实 MD5；这里使用 PCS `meta.block_list` 的单项值，与源文件完整 MD5 严格相等。不是百度提供了 SHA256，也不是仅凭大小声称校验通过。

**92 个文件、48,650,476,805 字节已全部上传并逐一验证**；最终目录清单与预定文件集合完全一致，0 个未完成上传。

| 网盘文件类型 | 数量 |
|---|---:|
| 独立结果归档（含 39 个 ROI tar） | 63 |
| 逐文件索引 / 排除清单 | 9 / 9 |
| 原始归档注册清单 | 9 |
| 署名通知 / 完整许可 | 2 |

网盘中包含原生数值及训练/失败/选择依据、676 份预测、132 个完成组的 1,267,816 份 ROI NPZ、3,520 张可视化 PNG、24 组 embedding，以及压缩逐文件索引和排除清单。每个 `results-*.tar[.gz]` 可独立解包；`files-*.jsonl.gz` 给出源主机/路径、归档成员、大小、SHA256 和逻辑单元绑定。`excluded-*.jsonl.gz` 中 `separate_bundle` 表示已在另一归档内，不是全局缺失。归档源文件保留在原主机，未上传模型权重、独立数据集/输入图片、凭据或 session stores。

只有现有网盘账号所有者可访问；没有公开分享链接或提取码。已授权且登录同一账号的 ByPy 客户端使用 **相对 app 根目录**的路径，不能重复加 `/apps/bypy`：

```bash
bypy list IRAOD-New/results/completed_snapshot_20260912T014556Z
bypy downfile IRAOD-New/results/completed_snapshot_20260912T014556Z/assets-67-roi.json ./assets-67-roi.json
```

其他归档文件按清单中的 `bypy_path` 下载。公开 Git 仓库仅保存小型表格、紧凑清单、验证器和许可说明；未经账号所有者授权的读者不能直接下载私有 payload，因此这是可校验的私有归档交付，不是公开下载的完整数据发布。错误创建的 GitHub draft Release 已删除，不再作为当前存储入口。

可视化归档附有 [RESULT_ASSET_NOTICE.md](RESULT_ASSET_NOTICE.md) 及完整 [CC-BY-NC-4.0.txt](CC-BY-NC-4.0.txt)。保留上游来源、署名、非商业限制和修改说明；数据集衍生产物许可不替代各代码项目许可。许可文件中的官方 GitHub 来源链接是溯源依据，不是结果存储链接。
