# 第二检测器选型审计：原生 OBB 的 DETR 系检测器（DIOR-R / RSAR，严格 SFOD）

日期：2026-11-01。数字都来自下面列出的 README、论文、配置、训练日志和本分支的实测；推断出来的部分标 `未验证`。
本文件由选型审计（只读调研）加上第一阶段移植的实测（§5、§6）组成。

## 0. 结论与已定决策

**检测器：ARS-DETR，用 STAR-MMRotate 里的 mmrotate 0.3.4 移植版**（`yangxue0827/STAR-MMRotate` @ `465c63f`，Apache-2.0）。
具体是 RSAR benchmark 用的不带 DN 的 AR-CSL 变体（`ARSDETR` / `ARSDeformableDETRHead`）。代码已作为插件包 `ars_detr/` vendor 进本仓库，注册到已安装的 mmrotate 0.3.4 registry，不改 iraod env。

决定性理由：

1. **它是唯一与现有栈版本对齐的原生 OBB DETR。** STAR-MMRotate 的 `mmrotate/version.py` 是 `0.3.4`；mmcv 范围 `1.5.3–1.8.0`、mmdet 范围 `>=2.25.1, <3.0.0` 来自 `mmrotate/__init__.py:24-34` 的断言（mmdet 上下界在 `:33-34`），`requirements/mminstall.txt` 只写了 `mmcv-full>=1.5.0`。现有 env 是 mmrotate 0.3.4 / mmcv-full 1.7.2 / mmdet 2.28.0，都在范围内。RSAR 官方 ARS-DETR 训练日志的 env 头也是 `MMCV: 1.7.2`、`MMRotate: 0.3.4+48167d6`、PyTorch 1.13.1。
2. **RSAR 有官方结果和公开 checkpoint**，并且已在本仓库的 env 里复现（§5）。RSAR README 报告 ARS-DETR 31.56（标 `1x*`，实为 36e）。README 没有写明指标；审稿时按 RSAR 仓库 `configs/_base_/datasets/rsar.py:70` 的 `iou_thrs=[0.5..0.95]` 推断为 AP50:95，并标为 `未验证`。§5 的端口检查在 RSAR test 上测得 AP50:95 = 31.56，与 README 一致到小数点后两位，所以现在确认：31.56 是 test 集上的 AP50:95。DIOR-R 有论文结果：Table XI，R-50，3x=36e，AP50 66.12 / AP75 45.81 / AP50:95 43.89；论文用的是 trainval 训练（`未验证`）。
3. **角度约定与现有栈一致。** `ARCSLCoder` 断言 `angle_version == 'le90'`，与现栈 `angle_version = 'le90'` 相同。
4. **模块只依赖 mmcv 1.x / mmdet 2.x 的 API**，没有自编译 CUDA 扩展。推理是 top-k（`max_per_img`，默认 300），没有 NMS。`simple_test` 的输出格式（逐类 `(n, 6)` 数组，`cx, cy, w, h, a, score`）与现有消费方相同（单测见 `ars_detr/tests/test_port.py`）。

**DETR 组的 backbone 和源模型（2026-11-01 牵头方决策）：**

- Oriented R-CNN 组的两个冻结源模型是**随机初始化的 OrthoNet-50**：`rsar_source/` 和 `dior_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/train/resolved_config.py` 里 OrthoNet 都是 `init_cfg=None`、`load_from=None`。ImageNet 初始化的 OrthoNet（`attn_init='identity'`）只在分支 `sogc-orthonet-phase1`（`8b42687`，2026-10-26）上，晚于源模型训练。所以"与 R-CNN 源相同的 backbone 和 ImageNet 初始化"本身做不到。
- 因此 **DETR 组用 R-50 + torchvision ImageNet 初始化**（ARS-DETR 官方设置），两个数据集相同。
- **RSAR 源 = 公开的 `arcsl_detr_r50_rsar_epoch_36.pth`**（最后一个 epoch，只用 RSAR train 训练，证据见 §5），按官方 pipeline / 输入尺度使用，不再训练 RSAR 源。
- **DIOR-R 源**：R-50 ImageNet，只用 DIOR `ImageSets/train.txt`（`val.txt` 的 corruption 版本是自适应图像，绝不用于源训练），输入尺度与 Oriented R-CNN DIOR 源相同（800×800，同样的翻转增强），官方优化器（AdamW 1e-4，总 batch 8），72 epoch，取最后一个 epoch，不按 val 选模型。配置：`configs/ars_detr/arcsl_detr_r50_dior.py`。
- **DETR 组是独立的公平性组，不与 Oriented R-CNN 组放在同一张表里排名。** 两组的 backbone、初始化和源模型都不同，只能各自组内比较。
- 评测：两组都用本仓库同一个旋转 VOC AP50 评测器（`test.py --eval mAP`，即 `DOTADataset` / `DIORDataset.evaluate`），另外报 AP50:95（`tools/eval_map50_95.py`，即 `test.py --map50-95`，同一评测器在 IoU 0.50:0.05:0.95 上取平均）。这个工具来自 PR #44（`feat/lods-obb`），PR #45 依赖它先合入。

**备选是 O2-RTDETR**（ai4rs，TGRS 2026）。它精度最高，DIOR-R 和 RSAR 都有公开 checkpoint，但基于 mmrotate 1.x / mmengine / mmcv 2.x，整套 SFOD 代码都要重写（见 §4）。

## 1. 现有栈（只读核对）

| 项 | 值 | 证据 |
|---|---|---|
| env | `/home/zechuan/miniforge3/envs/iraod` | `queue_dru20260919/common.sh:3` (`PY=...`) |
| 版本 | py 3.10.21, torch 2.0.1+cu118, torchvision 0.15.2, mmcv-full 1.7.2, mmdet 2.28.0, mmrotate 0.3.4, numpy 1.26.4, scipy 1.15.3 | 在 gpu-67 上执行 `pip list` |
| GPU | gpu-67: 8× NVIDIA A100-SXM-64GB | `nvidia-smi` |
| R-CNN 组源训练 | Oriented R-CNN + OrthoNet-50（随机初始化），le90，800×800，100e；DIOR 用 `ImageSets/train.txt`（5,862 张），RSAR 用 train（78,837 张） | 两个 `resolved_config.py`；DIOR 与 RSAR 文件计数 |
| 与二阶段耦合的代码 | `sfod/rotated_unbiased_teacher.py:48,192-193,323-343`（rpn+roi forward_train）；`sfod/rotated_semi_two_stage.py:117-127`（simple_test）；VLST 的 `_roi_forward_with_vlst`（远端）用 `bbox_roi_extractor` 加 `shared_fcs` 得到 1024-d 正样本 RoI 特征；IRG/LPLD/SF-UT 的 `ProposalAlignedTeacher`（远端 `sfod/extensions/proposal_teacher.py:28-34,79-115`）在 teacher 和 student 之间共享 RPN proposal；DRU 的 historical student loss 挂在 `roi_head.bbox_head`（`sfod/rotated_dru.py:53-56,75`） | grep |

## 2. 候选对比

DIOR-R 的 AP50 是 VOC 口径；RSAR README 的 "mAP" 是 AP50:95（见 §0、§5）。

| 候选 | 代码 / License / 最近提交 | 框架与依赖 | 角度 | DIOR-R 结果（来源） | RSAR 结果（来源） | 公开 ckpt DIOR-R / RSAR | 源训练配方 / VRAM | 与现栈兼容性 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| **ARS-DETR**（TGRS 2024） | `httle/ARS-DETR`，Apache-2.0，2023-10-16；STAR-MMRotate 移植版 Apache-2.0，2025-02-03 | 原仓：mmrotate 0.1.0，mmcv 1.4.5–1.5.0；STAR 移植版：mmrotate 0.3.4，mmcv 1.5.3–1.8.0，mmdet 2.25.1–<3.0（`mmrotate/__init__.py:24-34`） | le90（断言） | 66.12 AP50（论文 Table XI，R-50，36e，bs2，AdamW 1e-4；训练集为 trainval，`未验证`） | 31.56 AP50:95（RSAR README，标 `1x*`，实为 36e；§5 端口检查确认是 test 集 AP50:95）；官方日志 val AP50 最后一个 epoch（ep36）为 0.640，最佳 ep33 为 0.650；本仓库评测器 test AP50 61.16 | 无公开 DIOR-R ckpt（`未验证`）/ 有，RSAR 作者的 Google Drive | RSAR 日志：4×RTX 3090，2 img/GPU，8,964 MB/GPU，0.423 s/iter（中位数），9,850 iter/epoch，36e | **同版本**，已 vendor 进现栈 | **采用** |
| O2-RTDETR（TGRS 2026） | `wokaikaixinxin/ai4rs/projects/rotated_rtdetr`，Apache-2.0，2026-10-06 | mmrotate 1.0.0rc1（ai4rs），mmcv 2.0.0rc4–2.2.0，mmdet 3.0–3.4，推荐 torch 2.4 | 长边 [0,π)（`width_longer=True, start_angle=0`） | R18vd 67.00 / R34vd 68.67 / **R50vd 72.26**（72e 配置，**ckpt 取的是第 42 epoch**） | R18vd **73.31 AP50 / 39.54 AP50:95**（36e，bs16） | 有 / 有（ModelScope） | DIOR R50vd 日志：2×P100，bs8，7.3 GB 中位（峰值 9.9），0.812 s/iter，1,466 iter/ep；RSAR R18vd：2×2080Ti，bs16，6.8 GB，0.655 s/iter | 需要单独的 mmengine env 并重写 SFOD | 备选 |
| RHINO（WACV 2025） | `SIAnalytics/RHINO`，**CC-BY-NC**，2024-07-26；ai4rs 也有移植 | mmrotate 1.0.0rc1，mmcv 2.0.0，mmdet 3.0.0，mmengine 0.7.3 | 长边 [0,π) | R50 **71.79** / Swin-T 72.67（arXiv 2305.07598 Table 5，800×800） | 无 | 无 / 无（只有 DOTA ckpt） | 2×V100/A100，总 bs8，36e；DOTA 在 2×A40 上 30 h，16–20 GB | 需要单独 env；非商用许可 | 不选 |
| OrientedFormer（TGRS 2024） | `wokaikaixinxin/OrientedFormer`，Apache-2.0，2026-09-27 | mmrotate 1.x，mmcv 2.0.0rc4–2.1.0，mmdet 3.0–3.1 | le90 | R50 67.28 / Swin-T 68.84（12e，2×2080Ti，bs4） | 69.80 AP50 / 37.48 AP50:95（ai4rs，1x，ckpt 为 epoch_11） | 有（HF）/ 有（ModelScope） | RSAR 日志：2×2080Ti，bs4，5.1 GB，0.568 s/iter，19,710 iter/ep | 需要单独 env；属 query-based Sparse 系，带 `dqs_cfg nms_rotated` | 不选 |
| Rotated Deformable DETR（ai4rs） | ai4rs，Apache-2.0 | 同 ai4rs | le90 | 无 | 66.20 AP50 / 33.05 AP50:95（3x） | 无 / 有 | 2×2080Ti，bs4 | 需要单独 env；属于朴素基线 | 不选 |
| Rotated DINO（ai4rs） | ai4rs，Apache-2.0 | 同 ai4rs | 长边 | 65.78（12e） | 无 | 有 / 无 | 2×P100，bs8，11.0 GB 中位（峰值 15.1），1.347 s/iter | 需要单独 env | 不选 |
| AO2-DETR（TCSVT） | `Ixiaohuihuihui/AO2-DETR`，Apache-2.0 | mmrotate 0.1.0，mmcv 1.4.5–1.5.0 | — | 无（论文用 DOTA / SKU110K-R / HRSC） | 无 | 无 | — | README 写明**完整 APR 模块代码已丢失**，仓库只有 Deformable DETR 基线 | 排除 |
| RO²-DETR（ISPRS JPRS 2025） | `DangMinmin/RO2-DETR`，无 license | — | — | — | — | — | — | 仓库只有 4 个文件（README 和 1 个 config），没有模型代码 | 排除 |
| Oriented-DETR（Point-Axis，ECCV 2024） | `zhaozeyang108/Oriented-DETR`，**无 license**，2024-12-12 | 独立 DINO 系代码，torch 1.12 | point-axis 表示 | 有 DIOR config，数值未核对（`未验证`） | 无 | 有 DIOR 训练说明，ckpt 未核对 | 8 卡示例 | 框架不同；无 license | 排除 |
| RiO-DETR（ECCV 2026） / v2 | `RicePasteM/RiO-DETR`，Apache-2.0；v2 为 **AGPL-3.0** | 独立 RT-DETR 系代码，torch 2.2.2 | — | 官方仓**因 IP 原因未放出论文组件**，只放了 RT-DETRv2-OBB 基线：S 73.75 … X 76.52；v2：N 72.17 … X 78.30 | 无 | 有 / 无 | — | 框架完全不同；v2 训练时用 DINOv3 teacher，会引入额外信息源 | 排除 |
| EMO2-DETR（TGRS 2023）、D2Q-DETR（arXiv 2303.00542）、O²DETR（arXiv 2106.03146） | 没有找到公开代码（GitHub 搜索无结果，web 检索只有论文） | — | — | — | — | — | — | — | 排除（"无代码"为检索结论，`未验证`穷尽） |
| detrex-obb / RT-DETR_OBB | 前者 4★，detrex；后者 README 只有 46 字节 | — | — | — | — | — | — | 没有可复现结果 | 排除 |

## 3. 各方法移植到 ARS-DETR 的忠实度

覆盖 `experiments/comparison/protocol.md` 的全部必需方法（A–F、IRG、LPLD、SFUT、DRU、AASFOD、SFYOLO），外加 PETS 和 LODS。第一批（A–F、PETS）已移植并过门控，第二批的最终判定和移植状态见 §10。

三条对所有方法都成立的约束：

- **伪标签分数阈值事先固定，不调。** 与 Oriented R-CNN 组同一规则：UT 系方法 `score = 0.7`（`configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar1.py:5`），R-CNN 组的端口用方法官方阈值的地方（PETS 的 0.5、LODS 的 0.8）DETR 端口用同一个值。绝不在 `val.txt` 或 RSAR val 上调阈值。DETR 每张图固定输出 top-300、无 NMS，所以阈值过滤是必需的第一步。
- **`use_bbox_reg=False` 必须清零全部回归损失。** 现有代码只把名字里带 `bbox` 的损失乘 0（`sfod/rotated_unbiased_teacher.py:296-301`）。ARS-DETR head 另外还输出 `loss_piou`、`loss_angle`、`d{i}.loss_iou`、`d{i}.loss_angle`、`enc_loss_piou`、`enc_loss_angle`（`enc_loss_bbox` 带 `bbox`），照搬会让 IoU 和角度损失在伪标签上继续回传。DETR 的 SFOD 端口必须在伪标签上清零所有回归项（box L1、IoU/PIoU、角度，含 `enc_` 和 `d{i}.` 前缀），只保留分类损失。Hungarian 匹配本身仍用框代价（`ARS_HungarianAssigner`：FocalLossCost + BBoxL1Cost + IoUCost(giou) + CrossEntropyLossCost 角度），这是 DETR 的固有部分。
- **挂接点缺失时必须报错。** 凡是按属性名查找 RoI head 的代码在 DETR 上会拿到 `None`，不能静默跳过（DRU 是现成的例子，见下表）。

| 方法 | 现有 OBB 端口的挂接点 | ARS-DETR 上的判定 | 理由 / 需要的适配 |
|---|---|---|---|
| A Source Only | 源模型直接测目标域 | **忠实** | 同一评测器；源模型换成 DETR 组的源（§0） |
| B ST / Mean Teacher | `UnbiasedTeacher`，rpn+roi `forward_train`、`simple_test` | **适配** | 需要单阶段版 teacher-student 包装；teacher 走 `simple_test`（top-k，无 NMS），伪框作为 gt 进 Hungarian 匹配；阈值 0.7；伪标签上只留分类损失（见上） |
| C CLIP-CGA | 标签层面：`sfod/cga.py` 的 `refine_test(results, img_metas)`，输入是 teacher 的 OBB 检测 | **忠实** | 与检测器无关；只是输入分布变了（固定 top-300、低分框多），先按 0.7 过滤 |
| D SARCLIP-CGA | 同 C | **忠实** | 同 C |
| E VLST | 特征层面：1024-d RoI fc 特征，正样本是分配到伪 GT 的 RoI（`_roi_forward_with_vlst`） | **适配** | 改用 decoder 末层 query 特征（256-d），正样本是 Hungarian 匹配到伪 GT 的 query；投影头输入维度 1024→256；一对一匹配下正样本数比 RoI 采样少，损失标度会变 |
| F CGA+VLST | C + E | **适配** | 继承 E 的适配 |
| IRG | `ProposalAlignedTeacher`：teacher 的 RPN proposal 同时喂给 teacher/student 的 RoI head，取分类器前特征建实例关系图，并用 `fc_cls` | **适配（结构性）** | DETR 没有共享 proposal：要把 teacher 的 query / reference point 喂给 student decoder 才能得到对齐的实例特征；图建在 256-d query 特征上，分类器换成 `cls_branches[-1]` |
| LPLD | 同上；在共享 proposal 上取低置信 teacher 预测，对 student 做 KL 蒸馏 | **适配（结构性）** | 同 IRG 的 query 对齐改造；"低置信 proposal"改成 teacher 低于阈值、且不与高置信伪框重叠的 query；sigmoid focal 分数与 softmax 分布不同，KL 要换成逐类 BCE，忠实度下降 |
| SFUT（SF-UT，arXiv 2407.07586） | `SFUTOBB`：UT + 完整伪框回归 + 逐步 EMA，无辅助损失 | **适配（包装层）** | 方法本身与检测器无关；用 B 的单阶段包装即可。注意 SF-UT 按论文**保留**伪框回归，不受上面"清零回归损失"约束（与 R-CNN 组端口一致） |
| DRU | `DRUUnbiasedTeacher`：Dynamic Retraining-Updating + Historical Student Loss；HSL 通过 `roi_head.bbox_head.forward` 抓分类分数（`sfod/rotated_dru.py:53-56,75`） | **不可直接移植，需重设计** | 在 DETR 上 `roi_head` 不存在，`_capture_student_cls` 返回 `(fn(), None)`，HSL 被静默跳过，DRU 退化成普通 UT。DETR 端口必须改为抓 `cls_branches[-1]` 的 sigmoid 输出并重写 HSL，或在挂接点缺失时直接报错；绝不能静默退化。DRU 的重训/更新调度与检测器无关 |
| AASFOD | 标准做法：每个 seed 先按 teacher 检测把目标图分成 source-similar / dissimilar（TSD），再做对齐 + FNS 两阶段（`train_aasfod`） | **适配** | 划分只消费 teacher 检测（需先按阈值过滤 top-300），对齐作用在 backbone / neck 特征上，都与检测器无关；具体挂接点以远端端口为准（`未验证`） |
| SFYOLO | TAM（图像层面的目标域增强）+ 两轮 teacher-student | **适配** | TAM 是图像层面，与检测器无关；teacher-student 部分用 B 的单阶段包装；挂接点以远端端口为准（`未验证`） |
| PETS | `PETSUnbiasedTeacher`：静态/动态 teacher 周期交换 + 两个 teacher 检测框的 WBF 共识（分数 > 0.5，IoU 0.55） | **适配** | 交换是 state_dict 互换，与架构无关；共识在检测框层面做，DETR 的 top-k 重复框会被 WBF 合并；阈值沿用 R-CNN 组端口的官方值 |
| LODS | teacher 的 RPN proposal `rois_t` 共享给 student RoI head，做 CE + 实例图 GW；另有图像风格增强 SEM 和 backbone C4 全局 GW（远端 `sfod/rotated_lods.py:174-183`） | **适配（结构性）** | SEM 和全局 GW 与检测器无关；共享 proposal 的 CE 和实例 GW 需要与 IRG 相同的 query 对齐改造 |

代码移植量：vendor 了 STAR-MMRotate 15 个 Python 文件（§5），全部注册到 mmrotate 0.3.4 的 registry。DETR 版 teacher-student 包装和 VLST query 钩子见 §10；本表是选型阶段的初判，第二批以 §10.4 为准。

## 4. 为什么不选 1.x 系（O2-RTDETR / RHINO / OrientedFormer）

- 现有 SFOD 代码（`UnbiasedTeacher`、`StrictSourceFreeDOTADataset`、CGA 钩子、VLST、DRU、PETS）都写在 mmdet 2.x 的 `EpochBasedRunner`、`DataContainer` 和 registry 上。换到 mmengine 就要整栈重写，两组会落在两套框架里，基线行为是否一致又多一个混杂因素。
- ai4rs 提供的 checkpoint 有按测试集挑 epoch 的迹象：O2-RTDETR R50vd 用的是第 42 epoch（共 72），OrientedFormer RSAR 标为 "best ckpt" epoch_11。挑最佳 epoch 就等于用测试集做模型选择，所以即使选它们也必须按"最后一个 epoch"的规则重训，公开 ckpt 不能直接当干净源模型。
- RHINO 是 CC-BY-NC 许可，DIOR-R 和 RSAR 都没有 ckpt。

## 5. 移植与端口检查（实测）

**Vendor 范围**（`ars_detr/`，STAR-MMRotate @ `465c63f`，保留上游版权头，`THIRD_PARTY_NOTICES.md` 已加 Apache-2.0 说明）：

- `core/bbox/assigners/ars_hungarian_assigner.py`（`ARS_HungarianAssigner`；代价项 FocalLossCost / BBoxL1Cost / IoUCost / CrossEntropyLossCost 是 mmdet 2.28 自带的，按官方日志里的配置构建）
- `core/bbox/match_costs/`（STAR 自己的 match cost registry 与旋转代价）
- `core/bbox/coder/arcsl_angle_coder.py`（`ARCSLCoder`）
- `models/utils/builder.py`（`ROTATED_TRANSFORMER`）、`rotated_transformer.py`、`ars_rotated_transformer.py`
- `models/dense_heads/rotated_detr_head.py`、`ars_detr_head.py`
- `models/detectors/ars_detr.py`、`rotated_detr.py`、`rotated_detr_crop.py`、`single_stage_crop.py`、`single_stage_img_split_bridge_tools.py`
- 损失：配置里的 FocalLoss / L1Loss / GIoULoss / CrossEntropyLoss 都是 mmdet 2.28 自带的，STAR 的 `models/losses` 里没有这个模型需要的文件，所以没有 vendor。

唯一的本地改动是 import 路径：STAR 独有的模块指向本包，与上游 mmrotate 0.3.4 相同的模块（`RotatedBaseDetector`、`FeatureRefineModule`、各 builder）指向已安装的 `mmrotate`。

**配置**：`configs/ars_detr/arcsl_detr_r50_rsar.py` 原样取自 RSAR 仓库 `configs/star/`；它引用的 `csl_detr_r50_rsar_fix.py` 不在 RSAR 仓库里，按 `csl_detr_r50_rsar.py` 和官方日志 `arcsl_detr_r50_rsar.json` 头部的 `config` 重建。与日志逐项一致：训练 RResize 1024、三向翻转 0.25×3；测试 800；AdamW lr 1e-4、wd 1e-5、backbone / sampling_offsets / reference_points 的 lr_mult 0.1；grad clip 35；warmup linear 500 iter、ratio 1/3；step [32]；36e；4 GPU × 2 img。本地只换了数据路径和数据集类（本仓库的 `DOTADataset`，test 指向带标注的 `test/annfiles`）。

**RSAR 源 checkpoint 的数据证据**：官方日志内嵌配置的 `data.train` 是 `RSARv2_fix/train/annfiles/` + `train/images/`；val 只用于每 epoch 的 `evaluation`（`save_best='auto'`），test 的 `ann_file` 指向 `test/images/`（无标注）。没有 val 或 test 图像进入训练。我们取的是最后一个 epoch 的 `epoch_36.pth`，不是 best。

| 项 | 值 |
|---|---|
| checkpoint | `/mnt/shared/zechuan/iraod_artifacts/ars_detr/official_rsar/arcsl_detr_r50_rsar_epoch_36.pth`（gpu-67），sha256 `1927e1cf11244cc53dac3b99d10bccbda19d0808a0ab97684133fbb57b8ff185` |
| 官方日志 | 同目录 `arcsl_detr_r50_rsar.json`，sha256 `cf023eff536f021c98af90e37bca5aee63b14a04400807fb5b7d342ca3cd6090` |
| 评测 | iraod env（torch 2.0.1），RSAR clean test 8,538 张，`test.py --eval mAP` 加上当时本分支的 `tools/eval_ap50_95.py`（之后已删除，改用 PR #44 的 `tools/eval_map50_95.py`；reviewer 按代码比对确认，在这个配置下两者计算的是同一个量，未重算）。gpu-67 上的 `portcheck.sh` 和结果文件保持原样，作为实际运行的记录 |
| AP50（本仓库 VOC 评测器） | **61.16** |
| AP50:95 | **31.56**（AP55 57.96 / AP60 53.87 / AP65 48.29 / AP70 40.27 / AP75 28.96 / AP80 15.56 / AP85 7.00 / AP90 2.54 / AP95 0.04） |
| 对照 | README 31.56；官方日志最后一个 epoch val AP50 0.640（最佳 ep33 0.650） |
| 结果目录 | `/mnt/shared/zechuan/iraod_artifacts/ars_detr/portcheck_rsar_test/`（`results.pkl`、`ap50_95.json`、`run.log`） |

**结论：移植正确。** AP50:95 与 README 的 31.56 一致到小数点后两位。test AP50 61.16 比官方日志 val AP50 64.0 低 2.9 个点，同一个 checkpoint，差别来自 val 和 test 两个 split 不同。训练时的 torch 1.13.1 换成本 env 的 2.0.1 没有带来可见偏差。checkpoint 的 `meta` 是 epoch 36、iter 354,780，即每 epoch 9,855 iter；乘以总 batch 8 约 78,840 张，正好是 RSAR train 的 78,837 张。这与"只用 train 训练"一致；如果 train+val 一起训练，会是 87,304 张。

上游实现中两处与常规写法不同，均保持原样（官方 checkpoint 就是这样训出来的）：`ARS_HungarianAssigner` 用 `BBoxL1Cost(box_format='xywh')` 时，预测框以 xyxy 传入、gt 被转成 cxcywh 后再比较；`_get_bboxes_single` 用图像宽度同时缩放 x 和 y（只对方形输入正确，RSAR / DIOR 的 800 输入都是方形）。另外 `RotatedDETR.forward_train` 每一步都把 C3 特征拷到 CPU 做 NaN 检查，会拖慢训练，下面的实测已经包含这部分开销。

## 6. 源训练成本（DIOR-R 实测）

本节的实测替换了原审计里按 3090 日志换算的成本表（那张表全部是 `未验证` 的换算）。

**实测**：gpu-67 单张 A100-SXM-64GB，iraod env，`configs/ars_detr/arcsl_detr_r50_dior.py`（800×800，DIOR `train.txt`），代码 `22cf270`，seed 42，fp32。每个配置跑到目标迭代后停止。显存是 mmcv 日志里的 `memory`（`torch.cuda.max_memory_allocated`），s/iter 是去掉第一个日志区间后的中位数。日志在 `/mnt/shared/zechuan/iraod_artifacts/ars_detr/smoke_dior/{spg2,spg8}/`。

| 每卡 batch | 迭代数 | 显存 | s/iter | 图像/s | 总 loss（首个 → 最后一个区间） | 其中 loss_piou / loss_angle |
|---|---|---|---|---|---|---|
| 2 | 110 | 5,689 MB | 0.304 | 6.6 | 57.48 → 44.77 | 4.19 → 2.98 / 1.48 → 1.24 |
| 8 | 45 | 20,770 MB | 0.995 | 8.0 | 59.03 → 45.88 | 4.59 → 3.44 / 1.47 → 1.35 |

所有损失项都是有限值，并且在 warmup（lr 从 3e-5 升到 5e-5）期间下降。

**DIOR-R 72e 的成本**：5,862 × 72 = 422,064 张图像，总 batch 8 时为 52,758 iter。

| 拓扑（总 batch 8） | 墙钟 | GPU·h | 依据 |
|---|---|---|---|
| 1×A100，每卡 8 | ≈14.6 h | ≈14.6 | 实测 0.995 s/iter |
| 4×A100，每卡 2（官方拓扑） | ≈4.5 h | ≈17.8 | 实测单卡 0.304 s/iter，假设 DDP 线性扩展（`未验证`） |
| 1×4090D，每卡 8 | ≈15 h | ≈15 | `未验证`：没有能跑的 4090D（gpu-68 / 134 / 136 没有 DIOR 图像，gpu-138 GPU 全忙）；fleet 里的 4090D 都是 48 GB，20.8 GB 放得下。按 RSAR 官方日志推算，这个模型每步受 CPU 侧开销（每步的 NaN 检查拷贝、CPU 上的 Hungarian 匹配）影响大，4090D 每步耗时估计与 A100 接近 |
| 4×4090D，每卡 2 | ≈4.5–5 h | ≈18–20 | `未验证`，同上 |

推荐 1×A100、每卡 8：总 batch 正好等于官方的 8，单进程不涉及 DDP，GPU·h 最少。backbone 的 BN 是冻结的，neck 用 GN，所以每卡 batch 不影响模型语义。

DIOR-R 源训练命令（gpu-67，部署目录 `/mnt/SSD2_8TB/zechuan/IRAOD-New-ars-detr-20261101`，`<g>` 为 agent-dispatch 认领到的 GPU）：

```bash
cd /mnt/SSD2_8TB/zechuan/IRAOD-New-ars-detr-20261101
export PYTHONPATH=$PWD PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 IRAOD_RUNTIME_READY=1 \
  CONDA_PREFIX=/home/zechuan/miniforge3/envs/iraod LD_LIBRARY_PATH=/home/zechuan/miniforge3/envs/iraod/lib \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<g> CGA_SCORER=none CGA_BACKEND=none CGA_FILTER_MODE=none
mkdir -p /mnt/shared/zechuan/iraod_artifacts/ars_detr/dior_source
nohup /home/zechuan/miniforge3/envs/iraod/bin/python train.py configs/ars_detr/arcsl_detr_r50_dior.py \
  --work-dir /mnt/shared/zechuan/iraod_artifacts/ars_detr/dior_source/seed42-r50-1gpu-spg8 --seed 42 \
  --cfg-options data.samples_per_gpu=8 data.workers_per_gpu=4 \
  > /mnt/shared/zechuan/iraod_artifacts/ars_detr/dior_source/seed42-r50-1gpu-spg8.out 2>&1 &
```

源模型取 `epoch_72.pth`。配置里唯一的训练内评测是第 72 个 epoch 在 `test.txt` 上的 AP50，不参与选模型。

**RSAR**：采用公开 checkpoint（§0、§5），不训练。作为参考，官方 36e 是 4×3090、每卡 2、1024 输入，日志中位 0.423 s/iter × 9,855 iter × 36 epoch ≈ 41.7 h 墙钟，约 167 个 3090 GPU·h。

DETR 的 SFOD 自适应阶段要跑 teacher 前向和 student 前后向，现 ST 配置的 `samples_per_gpu=32` 对 DETR 不可行。不同方法之间要按"看到的图像数"对齐预算，而不是按每卡 batch。

## 7. 主要风险

1. **小数据收敛。** DIOR 源只用 `train.txt` 的 5,862 张，论文 66.12 用的是 trainval 的 11,725 张（`未验证`），所以 DETR 源精度可能明显低于论文值。已按 72e 设置。
2. **SFOD 包装的语义。** 回归损失清零、阈值固定、DRU 不得静默退化，见 §3。
3. **变体不一致。** RSAR 的 "ARS-DETR" 实际是 AR-CSL 加 ARS transformer、不带 DN；DIOR-R 论文数是完整的 ARS-DETR（DN+ARW+ARM，`未验证`）。主实验用不带 DN 的变体：它有 RSAR ckpt 可做验证，teacher-student 也不用处理 DN 分支。
4. **两组不可横比。** DETR 组用 R-50 ImageNet，R-CNN 组用随机初始化 OrthoNet-50（§0），两组结果只能组内比较。
5. **绝对精度偏低。** ARS-DETR 低于 O2-RTDETR（72.26）和 RHINO（71.79），审稿时可能被质疑。应对是把它定位为"架构泛化检验"，而不是 SOTA 对比。

## 8. 未验证清单

- DIOR-R 论文 66.12 用的是 trainval，以及对应的是哪个 ARS-DETR 变体。
- ARS-DETR 没有公开的 DIOR-R ckpt：httle README 只放了 DOTA ckpt，STAR-MMRotate 的 HF 仓需要登录。
- AASFOD、SFYOLO 在远端端口里的具体挂接点（§3 表）。
- Oriented-DETR（point-axis）的 DIOR 数值；EMO2-DETR / D2Q-DETR / O²DETR "无公开代码"是否穷尽。
- §6 里 4090D 的换算和多卡扩展效率。

## 9. 来源

- ARS-DETR：https://github.com/httle/ARS-DETR ；论文 https://arxiv.org/abs/2303.04989 （HTML 版 Table XI）
- STAR-MMRotate：https://github.com/yangxue0827/STAR-MMRotate （`mmrotate/version.py`、`mmrotate/__init__.py:24-34`、`requirements/mminstall.txt`）
- RSAR：https://github.com/zhasion/RSAR （README benchmark 表与脚注；`configs/star/*.py`；`configs/_base_/datasets/rsar.py:70`；Google Drive 上的 ARS-DETR ckpt 与 json 日志）
- RHINO：https://github.com/SIAnalytics/RHINO ；https://arxiv.org/abs/2305.07598 （Table 5、Table 9）
- ai4rs（O2-RTDETR / OrientedFormer / Rotated Deformable DETR / Rotated DINO / RHINO 移植）：https://github.com/wokaikaixinxin/ai4rs ，各 `projects/*/README.md` 及 ModelScope 日志
- OrientedFormer：https://github.com/wokaikaixinxin/OrientedFormer
- O2-RTDETR：https://github.com/wokaikaixinxin/O2-RT-DETR ；https://arxiv.org/abs/2603.15497
- AO2-DETR：https://github.com/Ixiaohuihuihui/AO2-DETR ；https://arxiv.org/abs/2205.12785
- RO²-DETR：https://github.com/DangMinmin/RO2-DETR
- Oriented-DETR（point-axis）：https://github.com/zhaozeyang108/Oriented-DETR
- RiO-DETR：https://github.com/RicePasteM/RiO-DETR ；https://github.com/RicePasteM/RiO-DETRv2 ；https://arxiv.org/abs/2603.09411
- EMO2-DETR：https://pure.bit.edu.cn/en/publications/emo2-detr-efficient-matching-oriented-object-detection-with-trans ；D2Q-DETR：https://www.arxiv.org/pdf/2303.00542 ；O²DETR：https://arxiv.org/pdf/2106.03146

## 10. 第二阶段：严格 SFOD 方法移植到 ARS-DETR（分支 `feat/detr-sfod`）

代码：`sfod/detr_sfod.py`（`DETRMeanTeacher` = B，`DETRPETSTeacher` = PETS，`DETRVLSTTeacher` = E/F，`ARSDETR_CGA` = 所有臂的 EMA/静态 teacher 类型），配置 `configs/ars_detr/sfod/detr_{st,clip_cga,sarclip_cga,vlst,vlst_cga,pets}_{rsar,dior}_strict.py`，单元格脚本 `experiments/comparison/detr_sfod_cell.sh`，CPU 测试 `ars_detr/tests/test_detr_sfod.py`。
从别的分支原样带入、未改动的文件：`sfod/pets_dynamics.py`、`sfod/rotated_pets.py`（`feat/pets-obb`），`sfod/semantic_teacher/`、`sfod/cga.py` 与 `sarclip_adapter.py` 的 strict 版本（R-CNN 组 D/F 部署代码 `IRAOD-New-strict-af` = `0f98a48`）。

### 10.1 DETR 组共享自适应协议（启动任何目标域运行之前固定，未在目标 GT 上调）

| 项 | 值 | 依据 |
|---|---|---|
| 数据流 | R-CNN 组的 strict 数据流：只用 corruption val 图像（`StrictSourceFreeDOTADataset`，目标 GT 一律为空，非空即报错），同一组 weak / strong 视图（共享 RRandomFlip；weak = RResize 800；strong = 原分辨率 + ColorJitter / 灰度 / 高斯模糊），`weight_l = 0` | R-CNN 组 `*_st_baseline_rsar_orthonet_strict.py` |
| 拓扑 / batch | 1 GPU × 每卡 8 张，总 batch 8（= 源模型训练的总 batch） | 实测 B 在 RSAR 上 teacher + student 峰值 33.1 GB（源 spg8 20.8 GB 的 1.6×），A100-64GB 放得下；同一 GPU 放不下两份训练 |
| 预算 | 1 个 epoch 目标图像：RSAR 8,467 张 → 1,059 iter；DIOR-R 5,863 张 → 733 iter | 与 R-CNN 组相同的图像数 |
| 优化器 | 源模型的 AdamW lr 1e-4、wd 1e-5、backbone / sampling_offsets / reference_points lr_mult 0.1、grad clip 35，所有方法相同 | 源配置 |
| lr 策略 | R-CNN 组自适应用的策略：线性 warm-up 100 iter（ratio 0.001），单 epoch 内不衰减 | R-CNN 组 UT 配置 |
| EMA | B / C / D / E / F：每步动量 0.998（UT 官方值），最后一个 iter 的 EMA 作为结果；PETS：0.9996，首次交换后 0.999 | UT / PETS 官方值 |
| 伪标签 | teacher 的 top-300 输出，分数阈值事先固定：UT 系方法（B–F）0.7；有官方阈值的端口沿用官方值（PETS 0.5；LODS 若移植则 0.8） | 与 R-CNN 组同规则 |
| 伪标签上的损失 | 只训练分类：所有层（最后一层、`d0`–`d4.`、`enc_`）的 `loss_bbox`（L1）、`loss_iou` / `loss_piou`（GIoU）、`loss_angle`（AR-CSL）乘 0；保留 `loss_cls`、`d{i}.loss_cls`、`enc_loss_cls` 和 VLST 损失 | `weight_pseudo_label_losses`；Hungarian 匹配仍用框代价，这是 DETR 的固有部分 |
| 评测 | 最后一个 iter 的 EMA，在 corruption TEST 上用本仓库同一个旋转 VOC AP50 评测器（`test.py --eval mAP`），`--out predictions.pkl` | 与 R-CNN 组相同 |
| seed | 42 / 43 / 44 | |
| PETS 交换点 | 官方 10 个 epoch、warm-up 2、period 1，按比例映射到单 epoch：RSAR 在第 211/317/423/529/635/741/847/953 iter 之前交换，DIOR 在 146/219/293/366/439/513/586/659 之前 | 与 R-CNN 组 PETS 端口同一规则 |

### 10.2 与 R-CNN 组实现的差异（DETR 特有）

1. **回归损失清零**：现有 `use_bbox_reg=False` 只清零名字带 `bbox` 的键（`sfod/rotated_unbiased_teacher.py:296-301`）。DETR 端口按后缀清零上表全部回归项，测试逐键检查值为 0、且这些项反传到任何参数的梯度都为 0。
2. **CGA（C/D/F）**：标签层面的仲裁不变，`refine_test` 直接作用在 teacher 的 OBB 检测上，环境变量与 R-CNN 组 C、D/F 完全一致（C：CLIP RN50x64、`an aerial image of a {}`、`legacy`；D/F：冻结的 base SARCLIP、`veto_soft`、0.7 / 0.1 / 0.9 / 0.7）。唯一的改动是先丢掉分数 < 0.7 的 top-300 框再送 CGA，这对伪标签集合是**精确等价**的：`veto_soft` 只会把分数乘以 ≤ 1 的系数或置 0；`legacy` 只在 CLIP top-1 ≠ 检测标签时混合，此时检测标签的概率 ≤ 0.5，混合后 ≤ 0.7·s + 0.15 < 0.7（s < 0.7）。代码对其他模式直接报错。测试用确定性的假打分器对两种模式比较"全量 CGA 后阈值"与"先过滤再 CGA"的伪标签，逐框相等。CGA 失败一律报错，没有回退到原始 teacher。
3. **VLST（E/F）**：学生特征从 1024-d RoI fc 特征换成 decoder 最后一层的 query 特征（`inter_states[-1]`，256-d，取最后一层分类分支的输入）；正样本是最后一层 Hungarian 匹配到伪 GT 的 query（与该层分类损失用同一次匹配：用相同输入重跑一次 `get_targets`），不带 DN（no-DN 变体）。投影头 256 → 256 → 512，其余超参与 R-CNN 组 E/F 相同（权重 0.1、温度 0.07、原型动量 0.9、文本/视觉 α 0.5、正样本分数 = 准入阈值 0.7）。一对一匹配下每个伪 GT 只有一个正样本（R-CNN 组每个伪 GT 可有多个 IoU ≥ 0.5 的 RoI），所以损失的样本数比 R-CNN 组少。测试检查：特征 256-d、正样本数 = 伪 GT 数且标签一致、梯度到达投影头、decoder 和 backbone、不经过最后一层分类分支、冻结的 VLM（文本原型）没有梯度、投影头不进入 EMA。视觉原型更新前先丢掉分数 < 0.7 的框再裁剪（`update_visual_prototypes` 本来就只用 ≥ 0.7 的框，结果相同，只少做裁剪）。原型更新出错时报错，不像 R-CNN 组那样打印后跳过。
4. **PETS**：交换、两个 teacher 的旋转 WBF 共识和推理与 R-CNN 端口共用同一份代码（`PETSUnbiasedTeacher` 的 `set_pets_schedule` / `_pets_before_iter` / `inference_unlabeled` 直接复用），只换检测器。DETR 的 top-300 里重复框较多，由 WBF 合并（测试检查共识输出非空且不多于输入）。
5. **空伪标签**：上游 `_get_target_single` 在一张图没有 GT 时崩溃（mmdet 把空的 `pos_gt_bboxes` reshape 成 (0, 4)，再取 `[:, :-1]` 得到 (0, 3)）。已在 `ars_detr/models/dense_heads/ars_detr_head.py` 改为直接索引 `gt_bboxes`，对非空 GT 结果不变。这是 vendor 代码的第二处本地改动。
6. **`batch_input_shape`**：DETR head 读这个 meta 字段，而 teacher / student 视图绕过了 `BaseDetector.forward_*`，所以在 `forward_train_semi` 里补上。
7. **DRU 守卫**：`DRUUnbiasedTeacher._capture_student_cls` 在没有 `roi_head.bbox_head` 时改为抛出 `NotImplementedError`，不再静默退化成 UT（测试覆盖）。

R-CNN 组与 DETR 组共有、本端口没有改的一点：CGA 按 `img_meta['filename']` 裁剪原图，但 teacher 框处于共享 RRandomFlip 之后的坐标系，CGA（`sfod/cga.py` 的 `_crop_patches`）不做翻转还原（VLST 会还原）。翻转概率 0.75，所以大多数训练图上 CGA 裁到的位置不对。为了与 R-CNN 组的 D/F 保持一致，这里保留原样，见 §10.7。

### 10.3 门控单元格（RSAR chaff，seed 42）

单卡 A100（gpu-67 GPU 6，agent-dispatch 认领 `gpu:gpu-67:6`），RSAR train → chaff，seed 42，1,059 iter（SFYOLO 为 2 epoch），结果为最后一个 iter 的 EMA（`iter_1060_ema.pth`，mmcv 的文件名是最终 iter + 1）在 chaff TEST（8,538 张）上的 VOC AP50。所有记录到的 loss 都是有限值（逐个检查了每个 log 区间）。输出根目录 `gpu-67:/mnt/shared/zechuan/iraod_artifacts/ars_detr/sfod/rsar/<方法>_s42/chaff/`。

| 方法 | AP50 | 相对 A | 最终 pseudo/图（累计） | EMA | 代码 |
|---|---|---|---|---|---|
| A Source Only | **42.12** | — | — | 源模型 `arcsl_detr_r50_rsar_epoch_36.pth` | — |
| B ST / Mean Teacher | 45.36 | +3.24 | 0.73 | `B_s42/chaff/work/iter_1060_ema.pth` | `a79aa92` |
| C CLIP-CGA | 23.61 | −18.51 | 0.073 | `C_s42/.../iter_1060_ema.pth` | `6ebd39c` |
| D SARCLIP-CGA | 15.94 | −26.18 | 0.011 | `D_s42/.../iter_1060_ema.pth` | `6ebd39c` |
| E VLST | 45.96 | +3.84 | 0.76 | `E_s42/.../iter_1060_ema.pth` | `67e3c80` |
| F CGA + VLST | 20.88 | −21.24 | 0.011 | `F_s42/.../iter_1060_ema.pth` | `a79aa92` 代码 + `6ebd39c` 配置 |
| PETS | **55.58** | +13.46 | 0.82 | `PETS_s42/.../iter_1060_ema.pth` | `6ebd39c` |
| SFUT（第二批） | 49.97 | +7.85 | 1.04 | `SFUT_s42/.../iter_1060_ema.pth` | `67e3c80` |
| SFYOLO（第二批，2 epoch） | 37.71 | −4.41 | 0.60 | `SFYOLO_s42/.../iter_2119_ema.pth` | `67e3c80` |

代码列是运行时部署目录 `.deploy_commit` 记录的提交；分支 rebase 到 `03f92d4` 之后它们分别对应 `728364b`、`cf1e474`、`09174b1`（rebase 只改了文档，代码相同）。`a79aa92` → `6ebd39c` → `67e3c80` 之间对 A–F、PETS 的计算没有任何改动（只修了配置加载顺序的问题，并加了 SFUT/SFYOLO 和一处行为不变的重构）。F 的第一次运行（`F_s42.invalid_cga_env_off/`）因为下面的配置问题实际没有开 CGA，已作废，没有进入上表；这里的 F 是修复后重跑的结果，日志里有 `[CGA] filter mode=veto_soft`。配置问题：mmcv 先执行子配置、后执行 `_base_`，原来基配置里的环境变量重置会把 C/D/F 的 CGA 关掉；现在基配置只做 `setdefault`，并有逐方法的配置测试。

**C/D/F 的塌缩是 DETR 组里发现的架构交互，不是移植缺陷**（牵头方决定按原样保留，不做事后修改：看过目标 TEST 结果后再选修法就等于在目标 GT 上做选择）。证据：

1. 与 R-CNN 组的 CGA 行为一致：`sfod/cga.py` 和 `sarclip_adapter.py` 与 R-CNN 组 D/F 的部署代码（`/home/zechuan/IRAOD-New-strict-af`）逐字节相同（sha256 一致）。在 32 张 chaff val 图上，用真实 SARCLIP 比较 R-CNN 的调用方式（`refine_test` 作用于 teacher 全部输出，即 `OrientedRCNN_CGA.simple_test(with_cga=True)`）和本端口的调用方式（先按 0.7 过滤，再 `refine_test`），32/32 张图的准入伪标签逐框相同（32 个原始伪框里保留 2 个）。
2. 机制：SARCLIP 在 chaff 上几乎总是和检测器标签不一致。400 张 chaff val 图上，veto_soft 只保留 teacher ≥ 0.7 框的 3–5%（翻转图 293 张：344 → 17，保留 4.9%；未翻转图 107 张：113 → 3，保留 2.7%）。翻转图和未翻转图的保留率差不多，所以 CGA 不还原翻转（§10.2 末段）不是主因。训练中伪标签降到约 0.01 个/图（D、F），只训分类的 DETR student 实际上只拿到"全是背景"的监督：所有 query 都被推向背景，teacher 分数随 EMA 一起下降，伪标签更少，循环下去（C/D/F 最后一个区间的总 loss 约 3e-4）。C 的 `legacy` 规则在不一致时同样把分数压到 0.7 以下（0.7·s + 0.3·p_label < 0.7），保留稍多（0.07 个/图）。
3. 对照 R-CNN 组同一单元格（RSAR chaff seed 42，`/mnt/shared/zechuan/iraod_artifacts/comparison/rsar/chaff/seed_42/f2145b4/methods/`）：SARCLIP 在那里同样只有约 8% 一致（D 训练日志 `diag_window`：2,842 个框里一致 236 个，argmax 大多是 harbor），但 R-CNN 的 D/F 没有塌缩（A 47.19 / D 45.93 / F 44.95）。区别在于 R-CNN 的 RoI 头每张图按固定比例采样正负样本（512 个 RoI、正样本 25%），伪 GT 很少时 RPN/RoI 的负样本损失也有上限；DETR 的一对一匹配加 focal loss 在零伪 GT 的图上把全部 300 个 query 当负样本，没有这层平衡。

### 10.4 第二批方法的最终判定

| 方法 | 判定 | 理由 | 动作 |
|---|---|---|---|
| IRG | **适配（结构性），不移植** | 原方法的实例关系图建在 teacher 与 student **同一批 RPN proposal** 的 RoI 特征上，并用两边的 `fc_cls`（`ProposalAlignedTeacher`）。DETR 没有共享 proposal；要对齐只能把 teacher 的 query / reference point 塞进 student decoder，这改变了 student 的前向，不再是原机制 | 不移植 |
| LPLD | **适配（结构性），不移植** | 依赖共享 RPN proposal 上的低置信 teacher 预测和 softmax KL；DETR 是逐类 sigmoid focal、没有共享 proposal，KL 只能换成逐类 BCE、proposal 只能换成 query | 不移植 |
| SFUT（SF-UT） | **忠实** | 机制与检测器无关：teacher 伪框（共享阈值 0.7）+ **保留**全部伪框损失（含回归）+ teacher 在前向中冻结、每次优化器 step 之后以 0.9996 做 EMA（R-CNN 端口 `AfterOptimizerTeacherHook`）。R-CNN 端口里的 `ProposalAlignedTeacher` 只是共用骨架，SFUT 不用它的辅助损失 | 已移植：`DETRSFUTTeacher` + `DETRAfterOptimizerTeacherHook`，配置 `detr_sfut_{rsar,dior}_strict.py` |
| DRU | **适配（需重设计），不移植** | 重训 / 动态更新与检测器无关，但 Historical Student Loss 定义在 RoI 头 softmax 分类分数的均值上（`sfod/rotated_dru.py`）。DETR 只能改成最后一层 query 的 sigmoid 分数均值，分布和语义都变了 | 不移植；`_capture_student_cls` 在没有 RoI 头时报错 |
| AASFOD | **适配（需重设计），不移植** | TSD 划分用 RoI 头共享 FC 上的 20 次 MC-dropout 方差（`sfod/extensions/aasfod_mechanisms.py`），在 DETR 上只能换成 decoder FFN 的 dropout，不是原定义；R-CNN 组自己的 AASFOD 协议（TSD 与否）也还未定 | 不移植 |
| SFYOLO | **忠实** | TAM（图像层面的风格自编码器）+ Mean Teacher + 学生稳定化（SSM）都与检测器无关；R-CNN 端口（`SFYOLOOBB`，`84a1aeb`）建在 SFUT 之上：strong 视图先过该域的 TAM（seed 42 的完整拟合，对所有检测器 seed 共用），参数 EMA 0.999（优化器 step 后），第 2 个 epoch 开始时 student ← 0.5·student + 0.5·teacher，单独的 2 epoch 预算 | 已移植：`DETRSFYOLOTeacher`（TAM 增广、参数 EMA、SSM 直接复用 `SFYOLOOBB` 的函数），复用 R-CNN 组的 TAM 文件 `comparison/xaf_student_quant_20260908/tam_runs_84a1aeb/<数据集>/<域>/seed_42/tam.pth`；载入时只把 VGG 编码器路径换成本机副本（`third_party/vgg16_oxford/`，文件名需一致），其余校验与 `load_completed_tam` 相同。单 epoch 内不衰减的 lr 策略（step=[1]）在 2 epoch 下会在第 2 个 epoch 乘 0.1，与 R-CNN 端口继承到的配置相同 |
| LODS | **适配（结构性），不移植** | SEM 和 backbone C4 全局 GW 与检测器无关，但 CE 与实例图 GW 用 teacher 的 RPN proposal `rois_t` 共享给 student RoI 头，问题同 IRG | 不移植 |

### 10.5 CPU 测试

```
cd <deploy dir> && PYTHONPATH=$PWD IRAOD_RUNTIME_READY=1 CUDA_VISIBLE_DEVICES= \
  python -W ignore -m unittest ars_detr.tests.test_detr_sfod ars_detr.tests.test_port
Ran 16 tests in 20.654s
OK
```

覆盖：伪标签上回归损失清零（值与梯度）、strict 前向 + EMA 更新 + 拒绝目标 GT、空伪标签、PETS 交换 / EMA / 共识跑在 DETR 输出上、CGA 预过滤的精确等价（`legacy`、`veto_soft`）与非法模式报错、VLST 维度 / 正样本 / 梯度路径 / 翻转还原、SFUT 保留回归且 teacher 只在优化器 step 后移动、SFYOLO 的 TAM 视图 / 有限损失 / SSM（用真实的 chaff TAM）、每个方法配置加载后的 CGA 环境变量、DRU 守卫。

### 10.6 命令模板

部署：新目录 `/mnt/SSD2_8TB/zechuan/IRAOD-New-detr-sfod-20261101`（gpu-67），从本分支 rsync，并从 `gpu-67:/home/zechuan/IRAOD-New-dru-20260919` 拷入被 gitignore 的 `mmdet_extension/models/backbones/orthonet.py` 和 `mmdet_extension/models/utils/`；`.deploy_commit` 记录提交号。先在 agent-dispatch 上认领 GPU。

```bash
# METHOD = A | B | C | D | E | F | PETS；DATASET = rsar | dior；GPU = 认领到的单卡
bash experiments/comparison/detr_sfod_cell.sh METHOD DATASET DOMAIN SEED GPU
# 训练（脚本内部）：
python train.py configs/ars_detr/sfod/detr_<name>_<dataset>_strict.py --work-dir <root>/work \
  --seed SEED --deterministic --no-validate \
  --cfg-options corrupt=DOMAIN load_from=SRC model.ema_ckpt=SRC checkpoint_config.max_keep_ckpts=1 \
    data.train.img_prefix=<corruption val images>
# 评测（A 用源 ckpt，其余用最大 iter 的 iter_*_ema.pth）：
python test.py configs/ars_detr/arcsl_detr_r50_<dataset>.py <ema> --eval mAP --out <root>/eval_test/predictions.pkl \
  --work-dir <root>/eval_test --cfg-options data.samples_per_gpu=8 data.workers_per_gpu=4 \
    data.test.ann_file=<test ann> data.test.img_prefix=<corruption test images>
```

输出：`/mnt/shared/zechuan/iraod_artifacts/ars_detr/sfod/<dataset>/<METHOD>_s<SEED>/<DOMAIN>/`（`train.log`、`work/`、`train_done`、`eval_test/{eval.log,eval_status,predictions.pkl}`）。DIOR-R 的单元格要等源模型 `dior_source/seed42-r50-1gpu-spg8/epoch_72.pth` 训练完成。

### 10.7 未决问题

1. CGA 不还原翻转（§10.2 末段）。这是 R-CNN 组 C/D/F 也有的问题；修它会同时改变两组的 C/D/F，需要牵头方决定是否两组一起修、一起重跑。
2. strong 视图沿用 R-CNN 组的"原分辨率"（RSAR 大多是 256×256），而 DETR 源模型是在 1024 上训练、800 上测试的；teacher 的 weak 视图是 800。是否另给 DETR 组的 strong 视图加 RResize，属于协议变更，没有做。
3. 上游 `_get_bboxes_single` 用图像宽度同时缩放 x、y（§5），对 RSAR 里少数非方形图（抽样约 0.5%）会让 teacher 框的 y 坐标偏移；源评测也一样，保留原样。
