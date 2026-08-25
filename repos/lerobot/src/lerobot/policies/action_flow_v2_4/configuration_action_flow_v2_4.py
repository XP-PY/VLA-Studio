#!/usr/bin/env python

"""Configuration for the PI0.5-based Action Flow V2.4 policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import PreTrainedConfig

from ..action_flow_v2_2.data_contracts_action_flow_v2_2 import BOX_FEATURE_KEYS, TARGET_BOXES
from ..pi05.configuration_pi05 import PI05Config
from .data_contracts_action_flow_v2_4 import PLANNING_FEATURE_KEYS


@PreTrainedConfig.register_subclass("action_flow_v2_4")
@dataclass
class ActionFlowV24Config(PI05Config):
    """PI0.5 flow matching with autoregressive Plan, box, and FAST objectives."""

    dtype: str = "bfloat16"
    n_action_steps: int = 1
    use_relative_actions: bool = True
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])

    pi05_pretrained_path: str | None = "lerobot/pi05_base"
    target_tokenizer_name: str = "google/paligemma-3b-pt-224"
    camera_names: tuple[str, ...] = ("wrist", "top")
    entity_names: tuple[str, ...] = ()
    max_box_entities: int = 8
    coordinate_token_count: int = 1_024
    max_plan_tokens: int = 32
    max_box_tokens: int = 192
    fast_tokenizer_name: str = "artifacts/tokenizers/rm65_fast_h30"
    max_fast_tokens: int = 256
    fast_skip_tokens: int = 128
    fast_vocab_size: int = 2_048
    paligemma_vocab_size: int = 257_152

    plan_loss_weight: float = 1.0
    box_loss_weight: float = 1.0
    fast_loss_weight: float = 1.0
    flow_loss_weight: float = 1.0
    restrict_fast_vocabulary: bool = True
    segment_aware_action_loss: bool = True
    keyframe_sampling_fraction: float = 0.05
    keyframe_feature_name: str = "planning.box_keyframe"
    generate_boxes_during_rollout: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pi05_pretrained_path is not None and not self.pi05_pretrained_path.strip():
            raise ValueError("`pi05_pretrained_path` must be None or a non-empty path/repo ID.")
        if not self.target_tokenizer_name:
            raise ValueError("`target_tokenizer_name` must be non-empty.")
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("`camera_names` must be non-empty and unique.")
        if self.max_box_entities <= 0:
            raise ValueError("`max_box_entities` must be positive.")
        if self.coordinate_token_count != 1_024:
            raise ValueError("PaliGemma grounding requires exactly 1,024 built-in location tokens.")
        if self.max_plan_tokens < 4 or self.max_box_tokens < 8:
            raise ValueError("Plan and box token budgets are too small.")
        if not self.fast_tokenizer_name:
            raise ValueError("`fast_tokenizer_name` must be non-empty.")
        if self.max_fast_tokens < 8:
            raise ValueError("`max_fast_tokens` must be at least 8.")
        if self.fast_skip_tokens < 0 or self.fast_vocab_size <= 0:
            raise ValueError("FAST skip/token vocabulary sizes must be valid.")
        if self.fast_token_start_id < 0:
            raise ValueError("The configured FAST range does not fit the PaliGemma vocabulary.")
        for name, value in (
            ("plan_loss_weight", self.plan_loss_weight),
            ("box_loss_weight", self.box_loss_weight),
            ("fast_loss_weight", self.fast_loss_weight),
            ("flow_loss_weight", self.flow_loss_weight),
        ):
            if value < 0:
                raise ValueError(f"`{name}` must be non-negative.")
        if (
            self.plan_loss_weight
            == self.box_loss_weight
            == self.fast_loss_weight
            == self.flow_loss_weight
            == 0
        ):
            raise ValueError("At least one Action Flow V2.4 loss must be enabled.")
        if self.train_expert_only and (
            self.plan_loss_weight > 0
            or self.box_loss_weight > 0
            or self.fast_loss_weight > 0
        ):
            raise ValueError(
                "`train_expert_only=True` is incompatible with Plan, box, or FAST supervision."
            )
        if not 0.0 <= self.keyframe_sampling_fraction < 1.0:
            raise ValueError("`keyframe_sampling_fraction` must be in [0, 1).")
        if not self.keyframe_feature_name:
            raise ValueError("`keyframe_feature_name` must be non-empty.")

    def set_dataset_feature_metadata(self, features: dict[str, dict[str, Any]]) -> None:
        """Validate planning/box labels and persist their camera/entity order."""

        missing = [key for key in (*BOX_FEATURE_KEYS, *PLANNING_FEATURE_KEYS) if key not in features]
        if missing:
            raise ValueError(f"Action Flow V2.4 dataset is missing features: {missing}.")

        box_metadata = features[TARGET_BOXES]
        camera_names = tuple(box_metadata.get("camera_names", ()))
        entity_names = tuple(box_metadata.get("entity_names", ()))
        if not camera_names or not entity_names:
            raise ValueError("Target-box metadata must define camera_names and entity_names.")
        if self.camera_names and tuple(self.camera_names) != camera_names:
            raise ValueError(
                f"Configured cameras {self.camera_names} do not match dataset cameras {camera_names}."
            )
        if tuple(box_metadata.get("shape", ()))[:2] != (len(camera_names), self.max_box_entities):
            raise ValueError(
                "Target-box dimensions do not match camera_names/max_box_entities: "
                f"{box_metadata.get('shape')}."
            )
        self.camera_names = camera_names
        self.entity_names = entity_names

    def architecture_contract(self) -> dict[str, Any]:
        return {
            "base_policy": "pi05",
            "vlm_backbone": "paligemma_2b",
            "action_expert": "gemma_300m_flow_matching",
            "autoregressive_targets": [
                "current_plan",
                "stage_target_boxes",
                "stage_local_fast_actions",
            ],
            "auxiliary_action_target": "stage_local_fast_tokens",
            "autoregressive_attention": "plan_then_parallel_boxes_and_fast",
            "expert_context": "shared_prefix_plus_teacher_forced_or_predicted_plan",
            "box_context": "shared_prefix_plus_plan",
            "flow_gradient_updates_vlm": True,
            "segment_aware_action_loss": self.segment_aware_action_loss,
        }

    @property
    def fast_token_start_id(self) -> int:
        return self.paligemma_vocab_size - self.fast_skip_tokens - self.fast_vocab_size

    @property
    def fast_token_stop_id(self) -> int:
        return self.paligemma_vocab_size - self.fast_skip_tokens


__all__ = ["ActionFlowV24Config"]
