# MDP-OBB CPU candidate

This is a preprint-guided OBB port candidate, not a validated comparison result.
It preserves the existing source detector and does not authorize GPU training.
The previous 540-cell matrix and published metrics are unchanged.

**GPU acceptance update:** the first `RSAR/chaff/42` attempt completed265 updates
but failed numerical acceptance. Update60 was the last finite logged point;
update70 had NaN losses, and both final checkpoints are invalid. See
[the preserved outcome](mdp_chaff42_gpu_acceptance.json). CPU checks and source
loading passed, but did not establish stable real-data training. No result is
promoted, no other cell or identical retry is started, and no numerical
mitigation is selected without a causal replay.
Only ten-update scalar logs and invalid final checkpoints were retained.
There is no finite intermediate state, exact failed augmented batch/RNG or
per-step gradient trace; finite checking occurred only after training.
Consequently the originating operation is not identified, and an instrumented
first-invalid capture requires a separate bounded execution decision.

One subsequent diagnostic has been authorized, capped at300 allocated GPU
seconds and80 updates. `mdp_first_invalid.py` wraps the existing native entry
without changing the model, batch, schedule, seed or data order. It keeps a
rolling CPU pre-forward model/optimizer/RNG snapshot and the current batch
references, checks loss/gradients/updated state, and enables targeted
forward/anomaly observation from update55. A failure saves the actual batch,
pre-state, available operands and gradients on the remote host. No full-budget
model result is produced; reaching80 finite updates is not a repair claim.
The finite helper is reused unchanged from the existing d668 implementation.

The corrected diagnostic reached80 finite updates and did not reproduce the
original NaN. It did not save a replay payload at the finite limit, so it is
neither a valid full-budget model nor evidence of a repair. The same recorded
algorithm, GPU, training arguments/configuration and environment selectors were
used, but matched native log windows already diverge within updates1-10, before
anomaly observation starts at55. See
[the diagnostic outcome](mdp_chaff42_diagnostic_outcome.json). Actual initial
auxiliary state, augmented-input/RNG and GPU execution parity were not captured
for both runs; the cause remains insufficient evidence. No further start is
authorized, and no scientific parameter has been changed to make this result
look successful.

## References and fixed interpretation

- Liu et al., arXiv:2401.17916v1, *Source-free Domain Adaptive Object Detection
  in Remote Sensing Images*, Eqs. (5), (8)-(20), Algorithm 1 and Sec. IV.B.
- Published identity: DOI 10.1080/10095020.2024.2378920.
- Official implementation: `weix-liu/AFSP` at
  `02ec5dae2d1904130a06202e6fcd53667aa847f7`, including `lib.tar`.

The publisher version was unavailable during source verification. This port is
explicitly tied to the inspected preprint and disclosed implementation choices,
not a claim of reproducing the published HBB scores.

## Data and detector flow

`MDPOBB` extends the existing image-only proposal teacher. Its RSAR/DIOR configs
inherit the same OrthoNet-50/FPN detector, rotated RPN/ROI heads, target-image
binding, seed/subset rules, batch32, LR0.02 and one-epoch candidate budget as the
existing ports. Runtime paths and fixed source checkpoints must still be bound
before an authorized run; config defaults are not deployment evidence.

An even local batch is split into two N/2 groups as in Algorithm 1. The frozen
teacher sees all weak-view images. Its ordinary detections provide pseudo boxes
at the existing threshold0.7; the shared finite, positive-width/height admission
mask remains active. The existing shared-geometry mapper rescales xywh into each
strong view without discarding angles. Mixing uses equal image weights and the
concatenated pseudo-box/label sets. It never accesses target annotations.

The Student runs two branches: the mixed N/2 images and AFSP applied to the first
N/2 strong images. Both retain all native RPN/ROI classification and regression
losses. The synthetic mixed-image metadata records both original filenames,
the pair's valid canvas and padding extents. It does not mark another pair's
larger collated padding as valid image area.

## AFSP

AFSP acts on the first residual-stage output C2 before subsequent backbone
stages, not independently on arbitrary FPN levels. Temporary hooks are removed
even if forward raises. The ordinary detector evaluation path never invokes it.

The style network follows the official two1x1-convolution architecture. Spatial
statistics use the population standard deviation from the preprint; numerical
epsilons1e-6 and1e-8 follow the official normalization pattern. A0.5 blend combines
learned and original channel statistics. L1 magnitude is preserved per image,
which extends the official batch1 computation without coupling unrelated images.
Dual gradient reversal gives adversarial gradients to AFSP parameters while
preserving the ordinary gradient direction upstream.

## Prototype feature distillation

PFD uses original stride16 C4, not the final classifier vector or a new detector
head. Teacher features are untransformed. The Student transformation follows the
official Conv3x3/BN/ReLU structure, with padding1 to preserve the C4 coordinate
frame for rotated pooling. The official padding-free transform shrinks that map.

The actual ROI extractor/classifier calls supply aligned RoIs and predicted
classes: all teacher inference proposals and the Student's actual training RoIs.
Native `RoIAlignRotated` uses7x7 output, sampling2, stride16 and clockwise le90
geometry. Pooling is chunked without capping or dropping proposals. Background
is a prototype class, following the official detector's class convention.
Per-class sums/counts are globally reduced with an autograd-aware sum for DDP.

The preprint Eq. (16) is implemented literally:

    global_i = 0.7 * local_i + 0.3 * local_(i-1)

Previous local prototypes are detached independent snapshots. They are not
aliases overwritten by the current batch and are not a recursive EMA. Empty
classes have zero local history for that iteration; alignment is evaluated only
where both current class means have observations. The class-support count is
reported, and a no-overlap loss remains graph-connected rather than silently
skipping either detection branch.

The PFD reduction follows the L2 norm sum in preprint Eq. (17), not the official
script's mean squared error:

    loss = loss_msp + loss_afsp + 0.5 * sum_class ||global_student - global_teacher||_2

This reduction, equal0.5/0.5 MSP (instead of the script's0.25/0.5 overlap), and
independent local history are intentional, disclosed preprint-guided choices.
No TEST score was used to choose them.

## EMA and source loading

Sec. IV.B explicitly states retention0.9 and teacher updates once after each
epoch. This agrees with the official script; Algorithm 1's `ema_period` is not
evidence that its experimental setup used per-iteration updates. The existing
`EpochFinalTeacherHook` runs before the checkpoint hook. No iteration-start EMA
is performed.

The Student's source-load guard checks every original detector key and shape,
rejects unexpected keys, and checks or initializes the identical plain-detector
teacher. Only new `mdp.*` state may be absent when loading a source checkpoint.
This is not a blanket `strict=False` success claim. Supplied teacher checkpoint
loading is strict. Full MDP state includes both local-prototype histories;
resuming adaptation also requires a separately restored corresponding EMA.
Exact optimizer/RNG/paired-EMA resume is not claimed by this CPU candidate.

Auxiliary parameters are Student-only. Teacher gradients stay disabled and
teacher state is not changed during either forward/backward branch.

## CPU checks and remaining execution gate

The targeted tests cover equal mixing, label/angle preservation, nonzero
adversarial gradients and their signs, per-image normalization, finite constant
features, two-step prototype history, two-rank global-prototype math, image-only
loading, native rotated pooling, native RPN/ROI losses, source-load rejection and
unchanged ordinary evaluation.

```sh
python -m unittest discover -s tools/tests -p 'test_mdp_losses.py' -v
python -m unittest discover -s tools/tests -p 'test_mdp_integration.py' -v
```

Tensor tests require PyTorch only. Integration uses the existing native
MMCV/MMRotate environment, not an environment with missing `mmcv._ext`.
No old environment is upgraded to accommodate the legacy AFSP repository.
The current [CPU evidence](mdp_obb_cpu_validation.json) records 21 passing checks
in the existing Host67 native environment, plus exact Student/teacher loading of
all396 original state entries from each real RSAR and DIOR source checkpoint.
Only11 new `mdp.*` entries are auxiliary. No actual-source model forward, target
training, GPU capacity check or benchmark score is claimed by the source-load
check. Those remain separate from the synthetic integration fixtures.
