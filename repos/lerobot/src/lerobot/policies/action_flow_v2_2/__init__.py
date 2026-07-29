#!/usr/bin/env python

"""Action Flow V2.2 configuration and data contracts.

The policy model and processor are intentionally introduced in later development
stages. Stage 1 exposes only the stable configuration and dataset interfaces.
"""

from .configuration_action_flow_v2_2 import ActionFlowV22Config
from .data_contracts_action_flow_v2_2 import (
    BOX_COORDINATE_NAMES,
    BOX_FEATURE_KEYS,
    GROUNDING_TOKEN_LOSS_WEIGHT,
    GROUNDING_TOKEN_MASK,
    GROUNDING_TOKENS,
    TARGET_BOXES,
    TARGET_BOX_LOSS_WEIGHT,
    TARGET_BOX_VISIBILITY,
    TARGET_ENTITY_IDS,
    TARGET_ENTITY_ROLES,
    TARGET_ENTITY_VALID,
    ActionFlowV22DatasetMetadata,
    BatchFieldSpec,
    TargetBoxRecord,
    TargetRole,
    TargetVisibility,
    make_model_batch_contract,
)

__all__ = [
    "BOX_COORDINATE_NAMES",
    "BOX_FEATURE_KEYS",
    "GROUNDING_TOKEN_LOSS_WEIGHT",
    "GROUNDING_TOKEN_MASK",
    "GROUNDING_TOKENS",
    "TARGET_BOXES",
    "TARGET_BOX_LOSS_WEIGHT",
    "TARGET_BOX_VISIBILITY",
    "TARGET_ENTITY_IDS",
    "TARGET_ENTITY_ROLES",
    "TARGET_ENTITY_VALID",
    "ActionFlowV22Config",
    "ActionFlowV22DatasetMetadata",
    "BatchFieldSpec",
    "TargetBoxRecord",
    "TargetRole",
    "TargetVisibility",
    "make_model_batch_contract",
]
