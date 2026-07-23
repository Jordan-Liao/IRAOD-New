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
