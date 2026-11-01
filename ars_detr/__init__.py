# ARS-DETR (AR-CSL, no denoising) vendored from STAR-MMRotate (Apache-2.0),
# see LICENSE here and THIRD_PARTY_NOTICES.md. Importing this package registers
# the modules into the installed mmrotate 0.3.4 / mmdet 2.x registries.
from .core.bbox.match_costs import *  # noqa: F401,F403
from .core.bbox.assigners.ars_hungarian_assigner import ARS_HungarianAssigner  # noqa: F401
from .core.bbox.coder.arcsl_angle_coder import ARCSLCoder  # noqa: F401
from .models.utils.rotated_transformer import RotatedDeformableDetrTransformer  # noqa: F401
from .models.utils.ars_rotated_transformer import ARSRotatedDeformableDetrTransformer  # noqa: F401
from .models.dense_heads.ars_detr_head import ARSDeformableDETRHead  # noqa: F401
from .models.detectors.ars_detr import ARSDETR  # noqa: F401
