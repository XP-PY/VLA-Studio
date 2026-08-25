#!/usr/bin/env python

"""Model-batch keys used by Action Flow V2.4."""

PLAN_TOKENS = "observation.afv24_plan_tokens"
PLAN_TOKEN_MASK = "observation.afv24_plan_token_mask"
PLAN_TOKEN_LOSS_WEIGHT = "observation.afv24_plan_token_loss_weight"

BOX_TOKENS = "observation.afv24_box_tokens"
BOX_TOKEN_MASK = "observation.afv24_box_token_mask"
BOX_TOKEN_LOSS_WEIGHT = "observation.afv24_box_token_loss_weight"

FAST_TOKENS = "afv24_fast_tokens"
FAST_TOKEN_MASK = "afv24_fast_token_mask"
FAST_TOKEN_LOSS_WEIGHT = "afv24_fast_token_loss_weight"

SEGMENT_ACTION_MASK = "segment_action_mask"

PLANNING_STAGE = "planning.stage"
PLANNING_TARGET_ENTITY = "planning.target_entity"
PLANNING_FRAMES_REMAINING = "planning.frames_remaining"
PLANNING_BOX_KEYFRAME = "planning.box_keyframe"

PLANNING_FEATURE_KEYS = (
    PLANNING_STAGE,
    PLANNING_TARGET_ENTITY,
    PLANNING_FRAMES_REMAINING,
    PLANNING_BOX_KEYFRAME,
)

__all__ = [
    "BOX_TOKENS",
    "BOX_TOKEN_LOSS_WEIGHT",
    "BOX_TOKEN_MASK",
    "FAST_TOKENS",
    "FAST_TOKEN_LOSS_WEIGHT",
    "FAST_TOKEN_MASK",
    "PLAN_TOKENS",
    "PLAN_TOKEN_LOSS_WEIGHT",
    "PLAN_TOKEN_MASK",
    "PLANNING_BOX_KEYFRAME",
    "PLANNING_FEATURE_KEYS",
    "PLANNING_FRAMES_REMAINING",
    "PLANNING_STAGE",
    "PLANNING_TARGET_ENTITY",
    "SEGMENT_ACTION_MASK",
]
