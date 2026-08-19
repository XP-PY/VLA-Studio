#!/usr/bin/env python

"""Planning and model-batch contracts for Action Flow V2.3."""

from __future__ import annotations

from enum import IntEnum

PLAN_TOKENS = "observation.plan_tokens"
PLAN_TOKEN_MASK = "observation.plan_token_mask"
PLAN_TOKEN_LOSS_WEIGHT = "observation.plan_token_loss_weight"
ACTION_TOKEN_LOSS_WEIGHT = "action_token_loss_weight"

PLANNING_STAGE = "planning.stage"
PLANNING_TARGET_ENTITY = "planning.target_entity"
PLANNING_FRAMES_REMAINING = "planning.frames_remaining"
PLANNING_BOX_KEYFRAME = "planning.box_keyframe"

PLANNING_FEATURE_KEYS = (
    PLANNING_STAGE,
    PLANNING_TARGET_ENTITY,
    PLANNING_BOX_KEYFRAME,
)


class PlanningStage(IntEnum):
    """Structured next-step stage predicted by the VLM planning branch."""

    GRASP = 1
    PUT = 2
    COMPLETE = 3


__all__ = [
    "ACTION_TOKEN_LOSS_WEIGHT",
    "PLAN_TOKEN_LOSS_WEIGHT",
    "PLAN_TOKEN_MASK",
    "PLAN_TOKENS",
    "PLANNING_BOX_KEYFRAME",
    "PLANNING_FEATURE_KEYS",
    "PLANNING_FRAMES_REMAINING",
    "PLANNING_STAGE",
    "PLANNING_TARGET_ENTITY",
    "PlanningStage",
]
