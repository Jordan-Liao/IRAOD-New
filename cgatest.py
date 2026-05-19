# /home/storageSDA1/liaojr/SFOD-RS/debug_cga_one.py
#"/home/storageSDA1/liaojr/dataset/RSAR/train/images/0000002.png"
# debug_cga_from_poly.py
import numpy as np
from PIL import Image
from sfod.cga import CGA   # 按你项目实际导入路径修改
# ——— 1) 准备类别，与cga.py里的CLASSES一致 ———
CLASSES = ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor']
cls_to_id = {c:i for i,c in enumerate(CLASSES)}

# ——— 2) 你的两行DOTA标注（四点+类别+diff）———
poly_lines = [
    "61.0 252.0 208.0 99.0 226.0 116.0 78.0 269.0 harbor 0",
    "81.0 7.0 118.0 41.0 33.0 131.0 -4.0 97.0 harbor 0"
]

def poly_line_to_xyxy_and_label(line):
    parts = line.strip().split()
    coords = list(map(float, parts[:8]))           # x1 y1 ... x4 y4
    cls = parts[8]                                 # 类别名
    # diff = parts[9]  # 最后一个通常是困难/忽略标记，用不到
    xs = coords[0::2]; ys = coords[1::2]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)   # 直接取包围的轴对齐框
    return [x1, y1, x2, y2], cls_to_id[cls]

# ——— 3) 组装CGA需要的入参：boxes(轴对齐)、scores(可先给0.9)、labels(id) ———
boxes = []
scores = []
labels = []
for line in poly_lines:
    b, lid = poly_line_to_xyxy_and_label(line)
    boxes.append(b)
    labels.append(lid)
    scores.append(0.9)   # 随便给个分数，CGA会做重打分

boxes  = np.array(boxes,  dtype=np.float32)
scores = np.array(scores, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# ——— 4) 路径换成你的真实图像（与标注对应的那张）———
img_path = "/home/storageSDA1/liaojr/dataset/RSAR/train/images/0000002.png"  # TODO: 换成真实路径

# ——— 5) 调用CGA（在CGA.__init__和__call__里打断点）———
cga = CGA(CLASSES, model='RN50x64')
logits, patches = cga(img_path, boxes, scores, labels)

print("logits shape:", logits.shape)     # [N, num_classes]
print("top-1 idx:", logits.argmax(axis=1))
print("prob for GT cls:", [logits[i, labels[i]] for i in range(len(labels))])
