#!/usr/bin/env python

"""Configuration for the planning-conditioned Action Flow V2.3 policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.configs import PreTrainedConfig

from ..action_flow_v2_2.configuration_action_flow_v2_2 import ActionFlowV22Config
from .data_contracts_action_flow_v2_3 import PLANNING_FEATURE_KEYS

DEFAULT_PLANNING_TOKENS = (
    "<plan>",
    "</plan>",
    "<grasp>",
    "<put>",
    "<complete>",
)


@PreTrainedConfig.register_subclass("action_flow_v2_3")
@dataclass
class ActionFlowV23Config(ActionFlowV22Config):
    """AFV2.2 plus explicit next-step planning and plan-conditioned control.

    AFV2.2 token IDs remain unchanged. Planning tokens are appended after the
    FAST range, which allows an existing AFV2.2 tokenizer artifact to be
    extended without moving any grounding or action token.
    """

    tokenizer_artifact_path: str = "artifacts/tokenizers/action_flow_v2_3_smolvlm_fast"
    checkpoint_tokenizer_dirname: str = "action_flow_v2_3_tokenizer"

    planning_tokens: tuple[str, ...] = DEFAULT_PLANNING_TOKENS
    max_plan_tokens: int = 32
    plan_loss_weight: float = 1.0
    box_loss_weight: float = 1.0
    fast_loss_weight: float = 0.5
    flow_loss_weight: float = 1.0

    segment_aware_action_loss: bool = True
    keyframe_sampling_fraction: float = 0.25
    keyframe_feature_name: str = "planning.box_keyframe"
    generate_grounding_during_rollout: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.planning_tokens or len(set(self.planning_tokens)) != len(
            self.planning_tokens
        ):
            raise ValueError("`planning_tokens` must be non-empty and unique.")
        if any(
            not token or not token.startswith("<") or not token.endswith(">")
            for token in self.planning_tokens
        ):
            raise ValueError("Every planning token must be an angle-bracket token.")
        overlap = set(self.planning_tokens).intersection(self.structural_tokens)
        if overlap:
            raise ValueError(f"Planning tokens overlap AFV2.2 structural tokens: {overlap}.")
        required = {"<plan>", "</plan>", "<grasp>", "<put>", "<complete>"}
        if not required.issubset(self.planning_tokens):
            raise ValueError(f"Planning tokens are missing {sorted(required - set(self.planning_tokens))}.")
        if self.max_plan_tokens < 3:
            raise ValueError("`max_plan_tokens` must be at least 3.")
        if self.plan_loss_weight < 0:
            raise ValueError("`plan_loss_weight` must be non-negative.")
        if self.train_expert_only and self.plan_loss_weight > 0:
            raise ValueError(
                "`train_expert_only=True` is incompatible with a nonzero planning loss."
            )
        if not 0.0 <= self.keyframe_sampling_fraction < 1.0:
            raise ValueError("`keyframe_sampling_fraction` must be in [0, 1).")
        if not self.keyframe_feature_name:
            raise ValueError("`keyframe_feature_name` must be non-empty.")
        if (
            self.plan_loss_weight
            == self.box_loss_weight
            == self.fast_loss_weight
            == self.flow_loss_weight
            == 0
        ):
            raise ValueError("At least one Action Flow V2.3 loss must be enabled.")

    @property
    def action_flow_v2_2_vocab_size(self) -> int:
        """Vocabulary size before the appended planning-token range."""

        return self.fast_token_start_id + self.fast_token_count

    @property
    def planning_token_start_id(self) -> int:
        return self.action_flow_v2_2_vocab_size

    @property
    def required_vlm_vocab_size(self) -> int:
        return self.planning_token_start_id + len(self.planning_tokens)

    def structural_token_id(self, token: str) -> int:
        """Return the stable ID of an AFV2.2 structural or V2.3 planning token."""

        if token in self.planning_tokens:
            return self.planning_token_start_id + self.planning_tokens.index(token)
        return super().structural_token_id(token)

    def token_contract(self) -> dict[str, Any]:
        """Return the complete AFV2.3 token allocation."""

        contract = super().token_contract()
        contract["extended_vlm_vocab_size"] = self.extended_vlm_vocab_size
        contract["fast_token_range"] = [
            self.fast_token_start_id,
            self.action_flow_v2_2_vocab_size,
        ]
        contract["planning_token_ids"] = {
            token: self.structural_token_id(token) for token in self.planning_tokens
        }
        return contract

    def architecture_contract(self) -> dict[str, Any]:
        """Return the V2.3 branch and knowledge-insulation contract."""

        contract = super().architecture_contract()
        contract.update(
            {
                "autoregressive_targets": ["next_step_plan", "target_boxes", "fast_actions"],
                "expert_context": "detached_prefix_and_plan_kv",
                "segment_aware_action_loss": self.segment_aware_action_loss,
            }
        )
        return contract

    def set_dataset_feature_metadata(self, features: dict[str, dict[str, Any]]) -> None:
        """Validate AFV2.2 features plus the frame-level planning labels."""

        super().set_dataset_feature_metadata(features)
        missing = [key for key in PLANNING_FEATURE_KEYS if key not in features]
        if missing:
            raise ValueError(f"Action Flow V2.3 dataset is missing planning features: {missing}.")


__all__ = ["ActionFlowV23Config", "DEFAULT_PLANNING_TOKENS"]
