# Full-TEST vis + EigenCAM (72 cells)

Seed-42 A–F only. RSAR 8 domains × 6 methods + DIOR 4 domains × 6 methods.
Do not launch until the owner says start. `fulltest_72.sh` defaults to `DRY_RUN=1`.

## Storage

Canonical root on `.67` shared disk:

```text
$ART=/mnt/shared/zechuan/iraod_artifacts
$ART/comparison/show_dir_fulltest/<DS>/<domain>/<A-F>/*.png
$ART/comparison/gradcam_fulltest/<DS>/<domain>/<A-F>/*.png
```

On `.183` / `.134` the same tree is `/home/zechuan/iraod_artifacts/...` (host_binding map). Write there, then rsync back to `.67` if that host is not on `/mnt/shared`.

| 产物 | 图数 | 体积（按 250KB/PNG） |
|---|---:|---:|
| vis 全 TEST | 48×8538 + 24×11738 = 691,536 | ~173 GB |
| EigenCAM 全 TEST | 同上 | ~173 GB |
| 合计 | 1,383,072 | ~350 GB |

每格还写 `run.log`。先确认 `$ART` 所在盘有 ≥400 GB 空闲。

## Upload（机器间，不经本机）

Code（本目录脚本）→ 各机 `CODE` checkout：

```bash
rsync -az --relative \
  tools/analysis/cam_overlay.py \
  tools/analysis/gradcam_obb.py \
  experiments/comparison/fulltest_72.sh \
  gpu.67:/mnt/SSD2_8TB/zechuan/IRAOD-New-strict-af/
```

`.183` / `.134` 若没有这份 code，同样 rsync 到该机现用 IRAOD checkout，然后 `CODE=` 指向它。

Ckpt / 数据：`.67` 上 `checkpoints.json` 的 72 条路径已在 `$ART`。其它机器缺文件时，机器间 rsync 只补缺失：

```bash
# example: .67 → .183, resume missing
rsync -a --ignore-existing \
  /mnt/shared/zechuan/iraod_artifacts/comparison/rsar/ \
  gpu.183:/home/zechuan/iraod_artifacts/comparison/rsar/
rsync -a --ignore-existing \
  /mnt/shared/zechuan/iraod_data/RSAR/ \
  gpu.183:/home/zechuan/iraod_data/RSAR/
```

DIOR 同源：`dior_source/`、`comparison/dior/`、`dior_splits/`、`iraod_data/DIOR/`。

回传图（仅当执行机不是 `.67` shared）：

```bash
rsync -a gpu.183:/home/zechuan/iraod_artifacts/comparison/show_dir_fulltest/ \
  /mnt/shared/zechuan/iraod_artifacts/comparison/show_dir_fulltest/
```

## 执行划分

历史授权：`.67` GPU 4–7；`.183` GPU 1–9（0 不用）；`.134` GPU 4–7。共享机先 `nvidia-smi`。

| 机器 | SHARD | GPUS | 格数 | 内容 |
|---|---|---|---:|---|
| `.67` | `1/3` | `4,5,6,7` | 24 | 单元格 1,4,7,… |
| `.183` | `2/3` | `1,2,3,4,5,6,7,8,9` | 24 | 2,5,8,… |
| `.134` | `3/3` | `4,5,6,7` | 24 | 3,6,9,… |

打印（默认）：

```bash
MODE=vis DRY_RUN=1 SHARD=1/3 GPUS=4,5,6,7 bash experiments/comparison/fulltest_72.sh
MODE=gradcam DRY_RUN=1 SHARD=1/3 GPUS=4,5,6,7 bash experiments/comparison/fulltest_72.sh
```

冒烟：CAM 可限 8 张。vis 走 `test.py`，没有张数上限，冒烟请手跑一格或先 CAM。

```bash
MODE=gradcam DRY_RUN=0 SKIP_DONE=1 MAX_IMAGES=8 SHARD=1/1 GPUS=4 \
  bash experiments/comparison/fulltest_72.sh
```

全量 vis/CAM 不要设 `MAX_IMAGES`。`SKIP_DONE=1`：vis 按 PNG 数 ≥ 8538/11738 跳过；gradcam 把 `--skip-done` 传给 `gradcam_obb.py`。

## 时间（未实测，按 4090/A100 单卡 vis 0.25s/图、CAM 0.60s/图）

| 格 | vis | CAM |
|---|---|---|
| RSAR 一格 8538 图 | ~36 min | ~85 min |
| DIOR 一格 11738 图 | ~49 min | ~117 min |
| 72 格 GPU-小时 | ~48 h | ~115 h |

三机并行（4+9+4 GPU，按上表 shard，忽略排队）：

| 阶段 | 墙钟（估） |
|---|---|
| vis | 3–6 h |
| CAM | 8–14 h |
| vis+CAM | 12–20 h |

`.67` 只用 4 卡、其它机不跑：vis ~12 h，CAM ~29 h。冒烟 72×8 张约 20 分钟量级。

## 入口

- vis: `$CODE/test.py --eval mAP --show-dir ... --show-score-thr 0.3`
- CAM: `$CODE/tools/analysis/gradcam_obb.py`（EigenCAM on neck 最大分辨率特征，再画框）

例外与 vis 循环相同：RSAR chaff 用 `f2145b4`；chaff/F 用 `preflight_ddp2`。清单没有 DIOR-17 和 IRG 等端口方法。
