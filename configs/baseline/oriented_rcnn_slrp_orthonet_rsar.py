"""RSAR OrthoNet baseline with the separable low-rank projection enabled.

SLRP sits immediately after the stem's maxpool (``stages=(-1,)``). That position
is measured, not assumed: ``work_dirs/depth_profile_400`` scores one set of
observables at every tap from the input tensor to C5, and the post-maxpool tap
holds the peak separability for 5 of the 7 RSAR corruptions -- including
``noise_suppression`` at 0.863 on ``separable_residual_fraction``, the very
quantity SLRP is built on, against 0.596 at C2 and 0.501 at C4. The same profile
shows that signature decaying to 0.02 of its input-domain strength by C4, so a
module placed there cannot act on it.

SLRP is zero-initialized, so ``load_from`` may point at a plain OrthoNet
checkpoint: the backbone reproduces its pretrained forward pass exactly until
the gate is trained. ``ortho_vector`` is excluded from weight decay because the
forward pass L2-normalizes it, which makes decay on it pure norm decay with no
effect on the function (the supplied RSAR checkpoint has it collapsed to
||v|| ~ 0.02-0.09 in stages 2-4 for exactly this reason).
"""

_base_ = './oriented_rcnn_orthonet_rsar.py'

model = dict(
    backbone=dict(
        slrp_cfg=dict(
            enabled=True,
            stages=(-1,),  # -1 is the post-maxpool tap; 0..3 are C2..C5
            window=7,
            hidden_channels=16,
            gamma=0.5,
            sparse_multiplier=1.0)))

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={'ortho_vector': dict(decay_mult=0.0)}))
