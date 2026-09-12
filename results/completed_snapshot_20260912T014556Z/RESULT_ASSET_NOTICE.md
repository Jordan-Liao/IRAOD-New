# Result asset attribution and redistribution terms

This snapshot is published for **noncommercial research**. Dataset-derived
visualizations and other licensed dataset material remain subject to
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
The accompanying `CC-BY-NC-4.0.txt` is the full legal text supplied by RSAR.
These asset-specific terms are distinct from repository code licenses. No
commercial-use permission, ownership of upstream images, or upstream endorsement
is claimed. Preserve this notice, source links, attribution, modification notice,
license and disclaimer when redistributing these assets.

## RSAR-derived assets

Paths containing `/RSAR/` or `/rsar/` identify RSAR-derived results.

- Xin Zhang, Xue Yang, Yuxuan Li, Jian Yang, Ming-Ming Cheng, Xiang Li.
  *RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark*. CVPR, 2025.
- Dataset and source: <https://github.com/zhasion/RSAR>.
- Pinned license grant:
  <https://github.com/zhasion/RSAR/blob/6594e685de1e592bd66bff5451763380d0d36c53/README.md#license>.
- Pinned full legal text:
  <https://github.com/zhasion/RSAR/blob/6594e685de1e592bd66bff5451763380d0d36c53/LICENSE>.
- RSAR credits **SARDet-100K** as its underlying dataset:
  Yuxuan Li, Xiang Li, Weijie Li, Qibin Hou, Li Liu, Ming-Ming Cheng, Jian Yang.
  *SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR
  Object Detection*. NeurIPS, 2024.
- Upstream source and CC BY-NC 4.0 grant:
  <https://github.com/zcablii/SARDet_100K/blob/b289569419dd47a57240a46666fe2a69227f29f4/README.md#license>.

## DIOR/DIOR-R-derived assets

Paths containing `/DIOR/` or `/dior/` identify DIOR/DIOR-R-derived results.

- Ke Li, Gang Wan, Gong Cheng, Liqiu Meng, Junwei Han. *Object detection in
  optical remote sensing images: a survey and a new benchmark*. ISPRS Journal
  of Photogrammetry and Remote Sensing, 159:296-307, 2020.
- Gong Cheng, Jiabao Wang, Ke Li, Xingxing Xie, Chunbo Lang, Yanqing Yao,
  Junwei Han. *Anchor-free Oriented Proposal Generator for Object Detection*.
  IEEE Transactions on Geoscience and Remote Sensing, 2022.
- Official author dataset page: <https://gcheng-nwpu.github.io/#Datasets>.
- Pinned author declaration:
  <https://github.com/gcheng-nwpu/gcheng-nwpu.github.io/blob/121d17417c9effbe6e66bff2c5cde243ae87e1ad/index.html>.
  Its "DIOR and DIOR-R datasets" section states that DIOR-R shares the same
  images with DIOR and **both datasets are freely available under CC BY-NC 4.0**.

## Modifications and scope

The IRAOD experiment contributors generated the released result derivatives:
corrupted-domain image variants, detector box/label/score overlays, visualization
rendering and any displayed crop/resize; model predictions, ROI feature
extraction and sampled embedding plots are experimental derivatives, not
upstream original annotations or official benchmark claims. Original clean
images shown in overlays still belong to their upstream licensors.

No raw dataset distribution or model checkpoints are included. The PNGs are
the existing completed visualizations at the frozen scientific cutoff, not new
renderings. `files-*.jsonl.gz` binds each result to its original path, bytes and
checksum. The package creation time is later than the scientific cutoff and
does not change the experiment selection.

CC BY-NC 4.0 sections 2(a)(1)(A)-(B) permit noncommercial sharing of the licensed
and adapted material, subject to section 3 attribution conditions. The
licensors provide the material as-is under the license's disclaimer of
warranties and limitation of liability. Rights not granted by the license,
including applicable privacy, publicity, patent and trademark rights, are not
expanded by this publication.
