# Third-party notices

## StripNet backbone

The implementation in `mmdet_extension/models/backbones/stripnet.py` is
adapted from the official [Strip R-CNN repository](https://github.com/YXB-NKU/Strip-R-CNN),
copyright (c) 2022 MCG-NKU.

The upstream work is licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
It may not be used for commercial purposes under that license.

Local changes comprise registry/import compatibility changes, formatting,
argument validation, removal of unused classification-only helpers, and
documentation. The StripNet detection compute graph and parameter names are
preserved.

## ARS-DETR (STAR-MMRotate)

The package `ars_detr/` vendors ARS-DETR modules from
[STAR-MMRotate](https://github.com/yangxue0827/STAR-MMRotate) (commit
`465c63f4c4ab3660968c938ede46990645560933`), an MMRotate 0.3.4 fork that ports
[ARS-DETR](https://github.com/httle/ARS-DETR). Copyright (c) OpenMMLab and the
STAR-MMRotate / ARS-DETR authors, licensed under the
[Apache License, Version 2.0](ars_detr/LICENSE); upstream copyright headers are
kept in each file.

Local changes: imports of STAR-only modules were redirected to this package and
imports of unchanged MMRotate 0.3.4 modules to the installed `mmrotate`, so the
modules register into the installed registries without modifying the env. The
model code is otherwise unchanged.

## RSAR ARS-DETR configs

`configs/ars_detr/arcsl_detr_r50_rsar.py` is copied from the
[RSAR repository](https://github.com/zhasion/RSAR) (`configs/star/`, commit
`6594e685de1e592bd66bff5451763380d0d36c53`), and `csl_detr_r50_rsar_fix.py` is
rebuilt from that repository's `configs/star/csl_detr_r50_rsar.py` and the
config embedded in its official training log `arcsl_detr_r50_rsar.json`. The
RSAR repository is licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
It may not be used for commercial purposes under that license. Local changes
are data paths and this repository's dataset classes.
