# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from ppd_sharpdepth.patchrefiner.estimator.registry import MODELS

def build_model(cfg):
    """Build backbone."""
    return MODELS.build(cfg)