"""DETR group, method D (SARCLIP-CGA) on RSAR, strict.

CGA settings = the R-CNN group's strict D: frozen base SARCLIP
(SARCLIP_PRETRAINED, no LoRA; strict CGA raises otherwise), veto_soft.
"""
_base_ = './detr_st_rsar_strict.py'

import os

os.environ.update(CGA_SCORER='sarclip', CGA_BACKEND='sarclip', CGA_STRICT='1',
                  CGA_FILTER_MODE='veto_soft', CGA_DROP_SCORE='0.0',
                  CGA_FILTER_LOG_EVERY='500', CGA_VETO_PRED_THR='0.7',
                  CGA_VETO_LABEL_THR='0.1', CGA_PROTECT_DET_SCORE='0.9',
                  CGA_BLEND_DET_WEIGHT='0.7')
work_dir = './work_dirs/detr_sarclip_cga_strict_${corrupt}'
