"""DETR group, method C on DIOR-R (method part = detr_clip_cga_rsar_strict)."""
_base_ = './detr_st_dior_strict.py'

import os

list(map(lambda k: os.environ.pop(k, None),
         ['SARCLIP_LORA', 'SARCLIP_PRETRAINED', 'SARCLIP_DIR', 'SARCLIP_MODEL',
          'SARCLIP_CACHE_DIR', 'SARCLIP_PRECISION']))
os.environ.update(CGA_SCORER='clip', CGA_BACKEND='clip', CGA_STRICT='1',
                  CGA_CLIP_MODEL='RN50x64',
                  CGA_TEMPLATES='an aerial image of a {}',
                  CGA_FILTER_MODE='legacy')
work_dir = './work_dirs/detr_clip_cga_dior_strict_${corrupt}'
