"""DIOR binding for the same MDP-OBB source-preserving implementation."""

_base_ = './lpld_dior.py'

load_from = '/myfile/pretrain/oriented_rcnn_orthonet_dior_epoch_100.pth'
model = dict(type='MDPOBB', ema_ckpt=load_from)
