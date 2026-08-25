#!/usr/bin/env python

"""Pre/post processors for planning-conditioned Action Flow V2.3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from ..action_flow_v2_2.data_contracts_action_flow_v2_2 import (
    GROUNDING_TOKEN_LOSS_WEIGHT,
    GROUNDING_TOKEN_MASK,
    GROUNDING_TOKENS,
    TARGET_BOX_BINS,
    TARGET_BOX_LOSS_WEIGHT,
    TARGET_BOX_VISIBILITY,
    TARGET_ENTITY_IDS,
    TARGET_ENTITY_ROLES,
    TARGET_ENTITY_VALID,
)
from ..action_flow_v2_2.processor_action_flow_v2_2 import (
    ActionFlowV22LatestStateProcessorStep,
    _remove_observation_time_axis,
    validate_relative_action_stats,
)
from .configuration_action_flow_v2_3 import ActionFlowV23Config
from .data_contracts_action_flow_v2_3 import (
    ACTION_TOKEN_LOSS_WEIGHT,
    PLAN_TOKEN_LOSS_WEIGHT,
    PLAN_TOKEN_MASK,
    PLAN_TOKENS,
    PLANNING_FRAMES_REMAINING,
    PLANNING_STAGE,
    PLANNING_TARGET_ENTITY,
    SEGMENT_ACTION_MASK,
)
from .tokenization_action_flow_v2_3 import ActionFlowV23TargetTokenizer, VLM_TOKENIZER_DIR

TARGET_TOKENIZER_PROCESSOR_NAME = "action_flow_v2_3_target_tokenizer_processor"


def resolve_tokenizer_artifact_path(
    config: ActionFlowV23Config,
    *,
    pretrained_path: str | Path | None = None,
) -> Path:
    """Resolve a standalone or checkpoint-bundled V2.3 tokenizer artifact."""

    checkpoint_source = pretrained_path or config.pretrained_path
    checkpoint_path = Path(checkpoint_source).expanduser() if checkpoint_source else None
    if checkpoint_path is not None and checkpoint_path.is_dir():
        bundled = checkpoint_path / config.checkpoint_tokenizer_dirname
        if bundled.is_dir():
            return bundled.resolve()
    elif checkpoint_source is not None and not checkpoint_path.is_absolute():
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=str(checkpoint_source),
                allow_patterns=[f"{config.checkpoint_tokenizer_dirname}/**"],
            )
        )
        bundled = snapshot / config.checkpoint_tokenizer_dirname
        if bundled.is_dir():
            return bundled.resolve()

    artifact = Path(config.tokenizer_artifact_path).expanduser()
    if artifact.is_dir():
        return artifact.resolve()
    raise FileNotFoundError(
        f"Action Flow V2.3 tokenizer artifact was not found at {artifact} or in "
        f"checkpoint {checkpoint_path}. Run the V2.3 tokenizer preparation script first."
    )


def _target_tokenizer_config_kwargs(config: ActionFlowV23Config) -> dict[str, Any]:
    return {
        "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps,
        "camera_names": list(config.camera_names),
        "max_cameras": config.max_cameras,
        "max_box_entities": config.max_box_entities,
        "max_grounding_tokens": config.max_grounding_tokens,
        "max_fast_action_tokens": config.max_fast_action_tokens,
        "max_plan_tokens": config.max_plan_tokens,
        "expected_state_dim": config.expected_state_dim,
        "expected_action_dim": config.expected_action_dim,
        "expected_relative_action_dim": config.expected_relative_action_dim,
        "base_vlm_vocab_size": config.base_vlm_vocab_size,
        "structural_tokens": list(config.structural_tokens),
        "planning_tokens": list(config.planning_tokens),
        "coordinate_token_count": config.coordinate_token_count,
        "fast_token_count": config.fast_token_count,
        "location_token_prefix": config.location_token_prefix,
        "location_token_suffix": config.location_token_suffix,
        "location_token_width": config.location_token_width,
        "fast_token_prefix": config.fast_token_prefix,
        "fast_token_suffix": config.fast_token_suffix,
        "fast_token_width": config.fast_token_width,
        "extended_vlm_vocab_size": config.extended_vlm_vocab_size,
        "segment_aware_action_loss": config.segment_aware_action_loss,
    }


def _string_batch(value: Any, *, name: str, batch_size: int) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = [str(item) for item in value]
    else:
        raise TypeError(f"`{name}` must be a string or a batch of strings.")
    if len(values) != batch_size:
        raise ValueError(f"`{name}` has {len(values)} values for batch size {batch_size}.")
    return values


def _integer_batch(value: Any, *, name: str, batch_size: int) -> list[int]:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    values = value.reshape(-1).detach().cpu().tolist()
    if len(values) != batch_size:
        raise ValueError(f"`{name}` has {len(values)} values for batch size {batch_size}.")
    return [int(item) for item in values]


@dataclass
@ProcessorStepRegistry.register(name=TARGET_TOKENIZER_PROCESSOR_NAME)
class ActionFlowV23TargetTokenizerProcessorStep(ProcessorStep):
    """Serialize plans, sparse boxes, FAST actions, and stage masks."""

    tokenizer_artifact_path: str
    config_kwargs: dict[str, Any]
    target_tokenizer: ActionFlowV23TargetTokenizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        config = ActionFlowV23Config(device="cpu", **self.config_kwargs)
        self.target_tokenizer = ActionFlowV23TargetTokenizer.from_pretrained(
            self.tokenizer_artifact_path,
            config=config,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, torch.Tensor) or action.ndim != 3:
            raise ValueError(
                "Action Flow V2.3 expects actions shaped "
                f"(batch_size, chunk_size, action_dim), got {getattr(action, 'shape', None)}."
            )
        observation = transition.get(TransitionKey.OBSERVATION)
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        if not isinstance(observation, dict):
            raise ValueError("Action Flow V2.3 training requires an observation dictionary.")
        box_keys = (
            TARGET_BOX_BINS,
            TARGET_BOX_VISIBILITY,
            TARGET_BOX_LOSS_WEIGHT,
            TARGET_ENTITY_IDS,
            TARGET_ENTITY_ROLES,
            TARGET_ENTITY_VALID,
        )
        missing = [key for key in box_keys if key not in observation]
        missing.extend(
            key
            for key in (PLANNING_STAGE, PLANNING_TARGET_ENTITY, PLANNING_FRAMES_REMAINING)
            if key not in complementary
        )
        if missing:
            raise KeyError(f"Action Flow V2.3 training data is missing {missing}.")

        expected_ndims = {
            TARGET_BOX_BINS: 3,
            TARGET_BOX_VISIBILITY: 2,
            TARGET_BOX_LOSS_WEIGHT: 2,
            TARGET_ENTITY_IDS: 1,
            TARGET_ENTITY_ROLES: 1,
            TARGET_ENTITY_VALID: 1,
        }
        labels = {
            key: _remove_observation_time_axis(
                observation[key], expected_sample_ndim=ndim, name=key
            )
            for key, ndim in expected_ndims.items()
        }
        batch_size = action.shape[0]
        stages = _string_batch(
            complementary[PLANNING_STAGE], name=PLANNING_STAGE, batch_size=batch_size
        )
        targets = _string_batch(
            complementary[PLANNING_TARGET_ENTITY],
            name=PLANNING_TARGET_ENTITY,
            batch_size=batch_size,
        )
        remaining = _integer_batch(
            complementary[PLANNING_FRAMES_REMAINING],
            name=PLANNING_FRAMES_REMAINING,
            batch_size=batch_size,
        )
        serialized = [
            self.target_tokenizer.serialize_sample(
                normalized_relative_actions=action[index],
                target_box_bins=labels[TARGET_BOX_BINS][index],
                target_box_visibility=labels[TARGET_BOX_VISIBILITY][index],
                target_box_loss_weight=labels[TARGET_BOX_LOSS_WEIGHT][index],
                target_entity_ids=labels[TARGET_ENTITY_IDS][index],
                target_entity_roles=labels[TARGET_ENTITY_ROLES][index],
                target_entity_valid=labels[TARGET_ENTITY_VALID][index],
                planning_stage=stages[index],
                planning_target_entity=targets[index],
                planning_frames_remaining=remaining[index],
            )
            for index in range(batch_size)
        ]

        new_transition = transition.copy()
        new_observation = dict(observation)
        for key, attribute in (
            (PLAN_TOKENS, "plan_tokens"),
            (PLAN_TOKEN_MASK, "plan_token_mask"),
            (PLAN_TOKEN_LOSS_WEIGHT, "plan_token_loss_weight"),
            (GROUNDING_TOKENS, "grounding_tokens"),
            (GROUNDING_TOKEN_MASK, "grounding_token_mask"),
            (GROUNDING_TOKEN_LOSS_WEIGHT, "grounding_token_loss_weight"),
        ):
            new_observation[key] = torch.stack([getattr(item, attribute) for item in serialized])
        new_transition[TransitionKey.OBSERVATION] = new_observation

        complementary[ACTION_TOKENS] = torch.stack([item.action_tokens for item in serialized])
        complementary[ACTION_TOKEN_MASK] = torch.stack(
            [item.action_token_mask for item in serialized]
        )
        complementary[ACTION_TOKEN_LOSS_WEIGHT] = torch.stack(
            [item.action_token_loss_weight for item in serialized]
        )
        complementary[SEGMENT_ACTION_MASK] = torch.stack(
            [item.segment_action_mask for item in serialized]
        )
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "tokenizer_artifact_path": self.tokenizer_artifact_path,
            "config_kwargs": self.config_kwargs,
        }

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_action_flow_v2_3_pre_post_processors(
    config: ActionFlowV23Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create the relative-action, planning-target, and robot-action pipelines."""

    validate_relative_action_stats(config, dataset_stats)
    artifact_path = resolve_tokenizer_artifact_path(config)
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        action_names=config.action_feature_names,
    )
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        ActionFlowV22LatestStateProcessorStep(),
        relative_step,
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=str(artifact_path / VLM_TOKENIZER_DIR),
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        ActionFlowV23TargetTokenizerProcessorStep(
            tokenizer_artifact_path=str(artifact_path),
            config_kwargs=_target_tokenizer_config_kwargs(config),
        ),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        AbsoluteActionsProcessorStep(
            enabled=config.use_relative_actions,
            relative_step=relative_step,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline(
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


__all__ = [
    "ActionFlowV23TargetTokenizerProcessorStep",
    "TARGET_TOKENIZER_PROCESSOR_NAME",
    "make_action_flow_v2_3_pre_post_processors",
    "resolve_tokenizer_artifact_path",
]
