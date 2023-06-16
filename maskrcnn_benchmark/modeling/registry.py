# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from maskrcnn_benchmark.utils.registry import Registry

BACKBONES = Registry()
LANGUAGE_BACKBONES = Registry()
MMSS_HEADS = Registry()
RPN_HEADS = Registry()
ROI_BOX_FEATURE_EXTRACTORS = Registry()
ROI_BOX_PREDICTOR = Registry()
ROI_KEYPOINT_FEATURE_EXTRACTORS = Registry()
ROI_KEYPOINT_PREDICTOR = Registry()
ROI_MASK_FEATURE_EXTRACTORS = Registry()
ROI_MASK_PREDICTOR = Registry()
