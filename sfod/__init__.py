from .compat import patch_mmrotate_multiclass_nms_rotated
patch_mmrotate_multiclass_nms_rotated()

from .semi_dior_dataset import SemiDIORDataset
from .semi_dota_dataset import DOTADataset, SemiDOTADataset

from .dior import DIORDataset
from .dense_teacher_rand_aug import *

from .rotated_semi_two_stage import SemiTwoStageDetector
from .rotated_semi_base import SemiBaseDetector
from .rotated_unbiased_teacher import UnbiasedTeacher

from .oriented_rcnn_cga import OrientedRCNN_CGA
