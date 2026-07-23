#!/usr/bin/env python

"""Action Flow V2 uses the standard SmolVLA processors.

Segmentation labels are regular observation columns in the merged LeRobot
dataset, so every generic processor preserves and moves them with the batch.
"""

from ..smolvla.processor_smolvla import make_smolvla_pre_post_processors

make_action_flow_v2_pre_post_processors = make_smolvla_pre_post_processors

__all__ = ["make_action_flow_v2_pre_post_processors"]
