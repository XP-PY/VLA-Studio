#!/usr/bin/env python

"""Action Flow V2.1 uses SmolVLA normalization and tokenization processors."""

from ..smolvla.processor_smolvla import make_smolvla_pre_post_processors

make_action_flow_v2_1_pre_post_processors = make_smolvla_pre_post_processors

__all__ = ["make_action_flow_v2_1_pre_post_processors"]
