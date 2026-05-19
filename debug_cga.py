# /home/storageSDA1/liaojr/SFOD-RS/debug_cga_one.py
# 单图调试：调用 sfod.cga 里的 SARCLIP 版 CGA 做重打分

import os
import sar_clip
print("sar_clip at:", sar_clip.__file__)

import numpy as np
from sfod.cga import CGA   # 确保已替换为 SARCLIP 版本的 cga.py
from safetensors import safe_open


# 1) 类别（务必与 cfg & bbox_head.num_classes 顺序一致）
CLASSES = ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor']
cls_to_id = {c: i for i, c in enumerate(CLASSES)}

# 2) 两行 DOTA 标注（四点+类别+diff）
poly_lines = [
    "61.0 252.0 208.0 99.0 226.0 116.0 78.0 269.0 harbor 0",
    "81.0 7.0 118.0 41.0 33.0 131.0 -4.0 97.0 harbor 0"
]

def poly_line_to_xyxy_and_label(line):
    parts = line.strip().split()
    coords = list(map(float, parts[:8]))   # x1 y1 x2 y2 x3 y3 x4 y4
    cls = parts[8]
    xs = coords[0::2]; ys = coords[1::2]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)  # 取AABB包围框
    return [x1, y1, x2, y2], cls_to_id[cls]

# 3) 组装 CGA 入参
boxes, scores, labels = [], [], []
for line in poly_lines:
    b, lid = poly_line_to_xyxy_and_label(line)
    boxes.append(b)
    labels.append(lid)
    scores.append(0.9)  # 初始分数，CGA会重打分

boxes  = np.array(boxes,  dtype=np.float32)
scores = np.array(scores, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# 4) 图片路径（与标注对应）
img_path = "/home/storageSDA1/liaojr/dataset/RSAR/train/images/0000002.png"

# 5) 初始化 SARCLIP 版 CGA 并前向
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED_PATH = "/home/storageSDA1/Dataset/SARCLIP/ViT-B-32/vit_b_32_model.safetensors"
DEFAULT_CACHE_DIR = "/home/storageSDA1/Dataset/SARCLIP/ViT-B-32"

PRETRAINED_PATH = os.environ.get("SARCLIP_PRETRAINED", DEFAULT_PRETRAINED_PATH)
CACHE_DIR = os.environ.get("SARCLIP_CACHE_DIR", DEFAULT_CACHE_DIR)
assert os.path.exists(PRETRAINED_PATH)

model_candidates = [
    os.environ.get("SARCLIP_MODEL", DEFAULT_MODEL),
    "ViT-B-32",
    "RN50",
    "ViT-L-14",
]
seen = set()
cga = None
loaded_model = None
for model_name in model_candidates:
    if model_name in seen:
        continue
    seen.add(model_name)
    try:
        cga = CGA(
            class_names=CLASSES,
            model=model_name,
            pretrained=PRETRAINED_PATH,
            cache_dir=CACHE_DIR,
            precision=os.environ.get("SARCLIP_PRECISION", "fp32"),
            templates=("A SAR image of a {}", "This SAR patch shows a {}"),
            tau=100.0,            # 温度，可试 50~150
            expand_ratio=0.4,     # 框扩张，可试 0.2~0.6
            force_grayscale=False, # 若是单通道SAR且想强制灰度->3通道，可设 True
            backend="sarclip",
        )
        loaded_model = model_name
        print("loaded SARCLIP_MODEL:", loaded_model)
        break
    except Exception as e:
        msg = str(e).splitlines()[0] if str(e) else repr(e)
        print(f"failed SARCLIP_MODEL={model_name}: {type(e).__name__}: {msg}")

if cga is None:
    with safe_open(PRETRAINED_PATH, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        stage_depths = {}
        for key in keys:
            parts = key.split(".")
            if len(parts) >= 4 and parts[0] == "stages" and parts[2] == "blocks":
                stage_idx = int(parts[1])
                block_idx = int(parts[3])
                stage_depths.setdefault(stage_idx, set()).add(block_idx)
        stage_depths = {k: len(v) for k, v in stage_depths.items()}
        stem_shape = tuple(f.get_tensor("stem.0.weight").shape) if "stem.0.weight" in keys else None
        head_shape = tuple(f.get_tensor("head.norm.weight").shape) if "head.norm.weight" in keys else None

    print("checkpoint summary:")
    print("  num_keys:", len(keys))
    print("  stage_depths:", stage_depths)
    print("  stem.0.weight:", stem_shape)
    print("  head.norm.weight:", head_shape)
    print("  prefix sample:", sorted({k.split(".")[0] for k in keys})[:10])
    print("likely structure: ConvNeXt-Small-like image tower (3, 3, 27, 3) with 96/192/384/768 channels")
    print("note: this checkpoint has no text tower keys, so it is not a full SARCLIP CLIP checkpoint for zero-shot classification")
    raise RuntimeError("No SARCLIP model candidate could load model.safetensors")

logits, patches = cga(img_path, boxes, scores, labels)

print("logits shape:", logits.shape)     # [N, num_classes]
print("top-1 idx:", logits.argmax(axis=1))
print("prob for GT cls:", [float(logits[i, labels[i]]) for i in range(len(labels))])
