#!/usr/bin/env python

"""Input/output processors for Action Flow V2.2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_action_flow_v2_2 import ActionFlowV22Config
from .data_contracts_action_flow_v2_2 import (
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
from .tokenization_action_flow_v2_2 import (
    ActionFlowV22TargetTokenizer,
    VLM_TOKENIZER_DIR,
)

TARGET_TOKENIZER_PROCESSOR_NAME = "action_flow_v2_2_target_tokenizer_processor"
LATEST_STATE_PROCESSOR_NAME = "action_flow_v2_2_latest_state_processor"


def resolve_tokenizer_artifact_path(
    config: ActionFlowV22Config,
    *,
    pretrained_path: str | Path | None = None,
) -> Path:
    """Resolve a standalone or checkpoint-bundled V2.2 tokenizer artifact."""

    checkpoint_source = pretrained_path or config.pretrained_path
    checkpoint_path = (
        Path(checkpoint_source).expanduser() if checkpoint_source is not None else None
    )
    if checkpoint_path is not None and checkpoint_path.is_dir():
        bundled = checkpoint_path / config.checkpoint_tokenizer_dirname
        if bundled.is_dir():
            return bundled.resolve()
    elif checkpoint_source is not None and not checkpoint_path.is_absolute():
        from huggingface_hub import snapshot_download

        snapshot_path = Path(
            snapshot_download(
                repo_id=str(checkpoint_source),
                allow_patterns=[f"{config.checkpoint_tokenizer_dirname}/**"],
            )
        )
        bundled = snapshot_path / config.checkpoint_tokenizer_dirname
        if bundled.is_dir():
            return bundled.resolve()

    artifact_path = Path(config.tokenizer_artifact_path).expanduser()
    if artifact_path.is_dir():
        return artifact_path.resolve()
    raise FileNotFoundError(
        "Action Flow V2.2 tokenizer artifact was not found. Checked "
        f"{artifact_path} and checkpoint directory {checkpoint_path}. "
        "Run `examples/action_flow_v2/prepare_action_flow_v2_2_tokenizer.py` first."
    )


def _target_tokenizer_config_kwargs(config: ActionFlowV22Config) -> dict[str, Any]:
    """Return the JSON-safe config subset required by target serialization."""

    return {
        "chunk_size": config.chunk_size,
        "camera_names": list(config.camera_names),
        "max_cameras": config.max_cameras,
        "max_box_entities": config.max_box_entities,
        "max_grounding_tokens": config.max_grounding_tokens,
        "max_fast_action_tokens": config.max_fast_action_tokens,
        "expected_state_dim": config.expected_state_dim,
        "expected_action_dim": config.expected_action_dim,
        "expected_relative_action_dim": config.expected_relative_action_dim,
        "base_vlm_vocab_size": config.base_vlm_vocab_size,
        "structural_tokens": list(config.structural_tokens),
        "coordinate_token_count": config.coordinate_token_count,
        "fast_token_count": config.fast_token_count,
        "location_token_prefix": config.location_token_prefix,
        "location_token_suffix": config.location_token_suffix,
        "location_token_width": config.location_token_width,
        "fast_token_prefix": config.fast_token_prefix,
        "fast_token_suffix": config.fast_token_suffix,
        "fast_token_width": config.fast_token_width,
        "extended_vlm_vocab_size": config.extended_vlm_vocab_size,
    }


def _remove_observation_time_axis(
    value: torch.Tensor,
    *,
    expected_sample_ndim: int,
    name: str,
) -> torch.Tensor:
    """Return a batched label tensor after removing LeRobot's singleton time axis."""

    expected_batch_ndim = expected_sample_ndim + 1
    if value.ndim == expected_batch_ndim + 1 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != expected_batch_ndim:
        raise ValueError(
            f"`{name}` has shape {tuple(value.shape)}; expected a batched tensor with "
            f"{expected_sample_ndim} sample dimensions."
        )
    return value


@dataclass
@ProcessorStepRegistry.register(name=LATEST_STATE_PROCESSOR_NAME)
class ActionFlowV22LatestStateProcessorStep(ProcessorStep):
    """Remove LeRobot's singleton observation-time axis before relative conversion."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict) or OBS_STATE not in observation:
            return transition
        state = observation[OBS_STATE]
        if not isinstance(state, torch.Tensor):
            return transition
        if state.ndim >= 3 and state.shape[1] == 1:
            new_transition = transition.copy()
            new_observation = dict(observation)
            new_observation[OBS_STATE] = state[:, 0]
            new_transition[TransitionKey.OBSERVATION] = new_observation
            return new_transition
        return transition

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name=TARGET_TOKENIZER_PROCESSOR_NAME)
class ActionFlowV22TargetTokenizerProcessorStep(ProcessorStep):
    """Serialize box labels and normalized relative actions for joint V2.2 training."""

    tokenizer_artifact_path: str
    config_kwargs: dict[str, Any]
    target_tokenizer: ActionFlowV22TargetTokenizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        target_config = ActionFlowV22Config(device="cpu", **self.config_kwargs)
        self.target_tokenizer = ActionFlowV22TargetTokenizer.from_pretrained(
            self.tokenizer_artifact_path,
            config=target_config,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, torch.Tensor) or action.ndim != 3:
            raise ValueError(
                "Action Flow V2.2 target serialization expects normalized actions shaped "
                f"(batch_size, chunk_size, action_dim), got {type(action)} "
                f"{getattr(action, 'shape', None)}."
            )

        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            raise ValueError("Action Flow V2.2 training requires an observation dictionary.")
        required = (
            TARGET_BOX_BINS,
            TARGET_BOX_VISIBILITY,
            TARGET_BOX_LOSS_WEIGHT,
            TARGET_ENTITY_IDS,
            TARGET_ENTITY_ROLES,
            TARGET_ENTITY_VALID,
        )
        missing = [key for key in required if key not in observation]
        if missing:
            raise KeyError(
                "Action Flow V2.2 training requires structured target-box labels; "
                f"missing {missing}."
            )

        labels = {
            TARGET_BOX_BINS: _remove_observation_time_axis(
                observation[TARGET_BOX_BINS],
                expected_sample_ndim=3,
                name=TARGET_BOX_BINS,
            ),
            TARGET_BOX_VISIBILITY: _remove_observation_time_axis(
                observation[TARGET_BOX_VISIBILITY],
                expected_sample_ndim=2,
                name=TARGET_BOX_VISIBILITY,
            ),
            TARGET_BOX_LOSS_WEIGHT: _remove_observation_time_axis(
                observation[TARGET_BOX_LOSS_WEIGHT],
                expected_sample_ndim=2,
                name=TARGET_BOX_LOSS_WEIGHT,
            ),
            TARGET_ENTITY_IDS: _remove_observation_time_axis(
                observation[TARGET_ENTITY_IDS],
                expected_sample_ndim=1,
                name=TARGET_ENTITY_IDS,
            ),
            TARGET_ENTITY_ROLES: _remove_observation_time_axis(
                observation[TARGET_ENTITY_ROLES],
                expected_sample_ndim=1,
                name=TARGET_ENTITY_ROLES,
            ),
            TARGET_ENTITY_VALID: _remove_observation_time_axis(
                observation[TARGET_ENTITY_VALID],
                expected_sample_ndim=1,
                name=TARGET_ENTITY_VALID,
            ),
        }
        if any(value.shape[0] != action.shape[0] for value in labels.values()):
            raise ValueError("Target-box labels and action chunks have different batch sizes.")

        serialized = [
            self.target_tokenizer.serialize_sample(
                normalized_relative_actions=action[index],
                target_box_bins=labels[TARGET_BOX_BINS][index],
                target_box_visibility=labels[TARGET_BOX_VISIBILITY][index],
                target_box_loss_weight=labels[TARGET_BOX_LOSS_WEIGHT][index],
                target_entity_ids=labels[TARGET_ENTITY_IDS][index],
                target_entity_roles=labels[TARGET_ENTITY_ROLES][index],
                target_entity_valid=labels[TARGET_ENTITY_VALID][index],
            )
            for index in range(action.shape[0])
        ]

        new_transition = transition.copy()
        new_observation = dict(observation)
        new_observation[GROUNDING_TOKENS] = torch.stack(
            [sample.grounding_tokens for sample in serialized]
        )
        new_observation[GROUNDING_TOKEN_MASK] = torch.stack(
            [sample.grounding_token_mask for sample in serialized]
        )
        new_observation[GROUNDING_TOKEN_LOSS_WEIGHT] = torch.stack(
            [sample.grounding_token_loss_weight for sample in serialized]
        )
        new_transition[TransitionKey.OBSERVATION] = new_observation

        complementary_data = dict(
            new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        )
        complementary_data[ACTION_TOKENS] = torch.stack(
            [sample.action_tokens for sample in serialized]
        )
        complementary_data[ACTION_TOKEN_MASK] = torch.stack(
            [sample.action_token_mask for sample in serialized]
        )
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
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


def validate_relative_action_stats(
    config: ActionFlowV22Config,
    dataset_stats: dict[str, dict[str, Any]] | None,
) -> None:
    """Validate statistics consumed after absolute-to-relative conversion.

    This validates shape and numerical integrity. Relative semantics are
    recorded separately by the dataset preparation script because they cannot
    be inferred from aggregate values alone.
    """

    if dataset_stats is None or ACTION not in dataset_stats:
        raise ValueError(
            "Action Flow V2.2 requires relative-action statistics for `action`. "
            "Run `examples/action_flow_v2/prepare_action_flow_v2_2_relative_stats.py` "
            "on the training dataset first."
        )
    if config.use_relative_actions and not config.action_feature_names:
        raise ValueError(
            "Action Flow V2.2 requires action names from dataset metadata before "
            "constructing relative-action processors."
        )

    action_stats = dataset_stats[ACTION]
    required = ("mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99")
    missing = [name for name in required if name not in action_stats]
    if missing:
        raise ValueError(f"Relative-action statistics are missing: {missing}.")
    for name in required:
        value = np.asarray(action_stats[name])
        if value.shape != (config.expected_action_dim,):
            raise ValueError(
                f"Action statistic `{name}` has shape {value.shape}; expected "
                f"{(config.expected_action_dim,)}."
            )
        if not np.isfinite(value).all():
            raise ValueError(f"Action statistic `{name}` contains non-finite values.")
    if np.any(np.asarray(action_stats["std"]) < 0):
        raise ValueError("Action standard deviations must be non-negative.")


def make_action_flow_v2_2_pre_post_processors(
    config: ActionFlowV22Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create the complete Action Flow V2.2 processor pair.

    The training path is:

    ``absolute action -> relative joints -> normalization -> model``

    The rollout path reverses it:

    ``model -> unnormalization -> absolute joints -> CPU robot action``

    The six RM65 joints are relative to the raw current state cached before
    normalization. Dimensions matching ``relative_exclude_joints`` remain
    absolute; the default exclusion is ``gripper``.

    During training, the normalized relative action chunk is serialized by FAST
    while structured target boxes are serialized into teacher-forcing tokens.
    Both token targets are produced before device transfer because FAST uses
    NumPy/SciPy internally. During rollout no action labels are present, so target
    serialization is skipped.
    """

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
        ActionFlowV22TargetTokenizerProcessorStep(
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
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


__all__ = [
    "ActionFlowV22TargetTokenizerProcessorStep",
    "ActionFlowV22LatestStateProcessorStep",
    "LATEST_STATE_PROCESSOR_NAME",
    "TARGET_TOKENIZER_PROCESSOR_NAME",
    "make_action_flow_v2_2_pre_post_processors",
    "resolve_tokenizer_artifact_path",
    "validate_relative_action_stats",
]
