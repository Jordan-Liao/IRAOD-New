"""DETR group, method F on DIOR-R (method part = detr_vlst_cga_rsar_strict)."""
_base_ = './detr_vlst_dior_strict.py'

import os

os.environ.update(CGA_SCORER='sarclip', CGA_BACKEND='sarclip', CGA_STRICT='1',
                  CGA_FILTER_MODE='veto_soft', CGA_DROP_SCORE='0.0',
                  CGA_FILTER_LOG_EVERY='500', CGA_VETO_PRED_THR='0.7',
                  CGA_VETO_LABEL_THR='0.1', CGA_PROTECT_DET_SCORE='0.9',
                  CGA_BLEND_DET_WEIGHT='0.7')
work_dir = './work_dirs/detr_vlst_cga_dior_strict_${corrupt}'
