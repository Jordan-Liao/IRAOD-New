"""Hook that freezes everything except SLRP.

SLRP calibration trains 1200 parameters against the real detection loss while the
detector stays exactly at its pretrained values. Freezing has to cover the neck
and heads too, which ``frozen_stages`` does not reach, and it has to keep every
normalization layer in eval mode so running statistics do not drift -- otherwise
the "pretrained detector" the gate is calibrated against changes underneath it.
"""

from mmcv.runner import HOOKS, Hook
from mmcv.utils.parrots_wrapper import _BatchNorm, _InstanceNorm


@HOOKS.register_module()
class FreezeExceptSLRPHook(Hook):
    """Disable grad everywhere but SLRP, and hold norm layers in eval mode.

    Args:
        trainable_markers (tuple[str]): substrings identifying the parameters
            that stay trainable.
        strict (bool): if True, raise when no parameter matches, which catches a
            config that silently trains nothing.
    """

    def __init__(self, trainable_markers=('slrp',), strict=True):
        if isinstance(trainable_markers, str):
            trainable_markers = (trainable_markers,)
        if not trainable_markers:
            raise ValueError('trainable_markers must not be empty')
        self.trainable_markers = tuple(trainable_markers)
        self.strict = bool(strict)
        self._reported = False

    def _is_trainable(self, name):
        return any(marker in name for marker in self.trainable_markers)

    def before_run(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') \
            else runner.model
        trainable, frozen = [], 0
        for name, parameter in model.named_parameters():
            if self._is_trainable(name):
                parameter.requires_grad_(True)
                trainable.append((name, parameter.numel()))
            else:
                parameter.requires_grad_(False)
                frozen += parameter.numel()
        if self.strict and not trainable:
            raise RuntimeError(
                'no parameter matched '
                f'{self.trainable_markers}; nothing would be trained')
        total = sum(count for _, count in trainable)
        runner.logger.info(
            f'FreezeExceptSLRPHook: training {total} parameters in '
            f'{len(trainable)} tensors, froze {frozen}')
        for name, count in trainable:
            runner.logger.info(f'  trainable: {name} ({count})')

    def _freeze_norm(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') \
            else runner.model
        for name, module in model.named_modules():
            if self._is_trainable(name):
                continue
            if isinstance(module, (_BatchNorm, _InstanceNorm)):
                module.eval()

    def before_train_epoch(self, runner):
        self._freeze_norm(runner)

    def before_train_iter(self, runner):
        # Re-asserted every iteration: the runner puts the whole model back into
        # train mode at each epoch boundary, and a single train-mode forward is
        # enough to move the running statistics.
        self._freeze_norm(runner)
