#!/usr/bin/env python

"""Pre/post-processing for the PI0.5-based Action Flow V2.4 policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
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
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from ..action_flow_v2_2.data_contracts_action_flow_v2_2 import (
    TARGET_BOXES,
    TARGET_BOX_LOSS_WEIGHT,
    TARGET_BOX_VISIBILITY,
    TARGET_ENTITY_IDS,
    TARGET_ENTITY_ROLES,
    TARGET_ENTITY_VALID,
)
from ..action_flow_v2_2.processor_action_flow_v2_2 import (
    _remove_observation_time_axis,
)
from ..pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from .configuration_action_flow_v2_4 import ActionFlowV24Config
from .data_contracts_action_flow_v2_4 import (
    BOX_TOKENS,
    BOX_TOKEN_LOSS_WEIGHT,
    BOX_TOKEN_MASK,
    FAST_TOKENS,
    FAST_TOKEN_LOSS_WEIGHT,
    FAST_TOKEN_MASK,
    PLAN_TOKENS,
    PLAN_TOKEN_LOSS_WEIGHT,
    PLAN_TOKEN_MASK,
    PLANNING_FRAMES_REMAINING,
    PLANNING_STAGE,
    PLANNING_TARGET_ENTITY,
    SEGMENT_ACTION_MASK,
)
from .tokenization_action_flow_v2_4 import (
    ActionFlowV24FastTokenizer,
    ActionFlowV24TargetTokenizer,
)

TARGET_TOKENIZER_PROCESSOR_NAME = "action_flow_v2_4_target_tokenizer_processor"


def validate_action_stats(
    config: ActionFlowV24Config,
    dataset_stats: dict[str, dict[str, Any]] | None,
) -> None:
    """Validate the relative-action statistics used by PI0.5 normalization."""

    if dataset_stats is None or ACTION not in dataset_stats:
        raise ValueError(
            "Action Flow V2.4 requires relative-action statistics for `action`."
        )
    if config.use_relative_actions and not config.action_feature_names:
        raise ValueError(
            "Action Flow V2.4 requires action names from dataset metadata before "
            "constructing relative-action processors."
        )
    if ACTION not in config.output_features:
        raise ValueError("Action Flow V2.4 output features do not define `action`.")

    action_dim = config.output_features[ACTION].shape[0]
    action_stats = dataset_stats[ACTION]
    required = ("mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99")
    missing = [name for name in required if name not in action_stats]
    if missing:
        raise ValueError(f"Relative-action statistics are missing: {missing}.")
    for name in required:
        value = np.asarray(action_stats[name])
        if value.shape != (action_dim,):
            raise ValueError(
                f"Action statistic `{name}` has shape {value.shape}; expected {(action_dim,)}."
            )
        if not np.isfinite(value).all():
            raise ValueError(f"Action statistic `{name}` contains non-finite values.")
    if np.any(np.asarray(action_stats["std"]) < 0):
        raise ValueError("Action standard deviations must be non-negative.")


def _string_batch(value: Any, *, name: str, batch_size: int) -> list[str]:
    if isinstance(value, str):
        result = [value]
    elif isinstance(value, (list, tuple)):
        result = [str(item) for item in value]
    else:
        raise TypeError(f"`{name}` must be a string or batch of strings.")
    if len(result) != batch_size:
        raise ValueError(f"`{name}` has {len(result)} values for batch size {batch_size}.")
    return result


def _integer_batch(value: Any, *, name: str, batch_size: int) -> list[int]:
    values = torch.as_tensor(value).reshape(-1).detach().cpu().tolist()
    if len(values) != batch_size:
        raise ValueError(f"`{name}` has {len(values)} values for batch size {batch_size}.")
    return [int(item) for item in values]


def _target_config_kwargs(config: ActionFlowV24Config) -> dict[str, Any]:
    return {
        "device": "cpu",
        "pi05_pretrained_path": None,
        "target_tokenizer_name": config.target_tokenizer_name,
        "camera_names": tuple(config.camera_names),
        "entity_names": tuple(config.entity_names),
        "max_box_entities": config.max_box_entities,
        "coordinate_token_count": config.coordinate_token_count,
        "max_plan_tokens": config.max_plan_tokens,
        "max_box_tokens": config.max_box_tokens,
        "fast_tokenizer_name": config.fast_tokenizer_name,
        "max_fast_tokens": config.max_fast_tokens,
        "fast_skip_tokens": config.fast_skip_tokens,
        "fast_vocab_size": config.fast_vocab_size,
        "paligemma_vocab_size": config.paligemma_vocab_size,
        "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps,
        "segment_aware_action_loss": config.segment_aware_action_loss,
    }


@dataclass
@ProcessorStepRegistry.register(name=TARGET_TOKENIZER_PROCESSOR_NAME)
class ActionFlowV24TargetTokenizerProcessorStep(ProcessorStep):
    """Convert reviewed Plan/box labels and stage-local actions into token targets."""

    config_kwargs: dict[str, Any]
    target_tokenizer: ActionFlowV24TargetTokenizer = field(init=False, repr=False)
    fast_tokenizer: ActionFlowV24FastTokenizer | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        config = ActionFlowV24Config(**self.config_kwargs)
        self.target_tokenizer = ActionFlowV24TargetTokenizer(config)

    def _get_fast_tokenizer(self) -> ActionFlowV24FastTokenizer:
        if self.fast_tokenizer is None:
            config = ActionFlowV24Config(**self.config_kwargs)
            self.fast_tokenizer = ActionFlowV24FastTokenizer(
                config,
                paligemma_tokenizer=self.target_tokenizer.vlm_tokenizer,
            )
        return self.fast_tokenizer

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, torch.Tensor) or action.ndim != 3:
            raise ValueError(
                "Action Flow V2.4 expects actions shaped (batch_size, chunk_size, action_dim), "
                f"got {getattr(action, 'shape', None)}."
            )
        observation = transition.get(TransitionKey.OBSERVATION)
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        if not isinstance(observation, dict):
            raise ValueError("Action Flow V2.4 requires an observation dictionary.")

        box_ndims = {
            TARGET_BOXES: 3,
            TARGET_BOX_VISIBILITY: 2,
            TARGET_BOX_LOSS_WEIGHT: 2,
            TARGET_ENTITY_IDS: 1,
            TARGET_ENTITY_ROLES: 1,
            TARGET_ENTITY_VALID: 1,
        }
        missing = [key for key in box_ndims if key not in observation]
        missing.extend(
            key
            for key in (PLANNING_STAGE, PLANNING_TARGET_ENTITY, PLANNING_FRAMES_REMAINING)
            if key not in complementary
        )
        if missing:
            raise KeyError(f"Action Flow V2.4 training data is missing {missing}.")

        labels = {
            key: _remove_observation_time_axis(
                observation[key], expected_sample_ndim=ndim, name=key
            )
            for key, ndim in box_ndims.items()
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
        plans = [
            self.target_tokenizer.serialize_plan(stages[index], targets[index])
            for index in range(batch_size)
        ]
        boxes = [
            self.target_tokenizer.serialize_boxes(
                target_boxes=labels[TARGET_BOXES][index],
                target_box_visibility=labels[TARGET_BOX_VISIBILITY][index],
                target_box_loss_weight=labels[TARGET_BOX_LOSS_WEIGHT][index],
                target_entity_ids=labels[TARGET_ENTITY_IDS][index],
                target_entity_roles=labels[TARGET_ENTITY_ROLES][index],
                target_entity_valid=labels[TARGET_ENTITY_VALID][index],
            )
            for index in range(batch_size)
        ]
        horizons = [max(1, min(value, action.shape[1])) for value in remaining]
        fast_tokenizer = self._get_fast_tokenizer()
        fast_targets = [
            fast_tokenizer.serialize(action[index, : horizons[index]])
            for index in range(batch_size)
        ]

        new_transition = transition.copy()
        new_observation = dict(observation)
        for key, values, attribute in (
            (PLAN_TOKENS, plans, "token_ids"),
            (PLAN_TOKEN_MASK, plans, "token_mask"),
            (PLAN_TOKEN_LOSS_WEIGHT, plans, "loss_weight"),
            (BOX_TOKENS, boxes, "token_ids"),
            (BOX_TOKEN_MASK, boxes, "token_mask"),
            (BOX_TOKEN_LOSS_WEIGHT, boxes, "loss_weight"),
        ):
            new_observation[key] = torch.stack([getattr(item, attribute) for item in values])
        new_transition[TransitionKey.OBSERVATION] = new_observation

        complementary[FAST_TOKENS] = torch.stack(
            [target.token_ids for target in fast_targets]
        )
        complementary[FAST_TOKEN_MASK] = torch.stack(
            [target.token_mask for target in fast_targets]
        )
        complementary[FAST_TOKEN_LOSS_WEIGHT] = torch.stack(
            [target.loss_weight for target in fast_targets]
        )
        step_indices = torch.arange(action.shape[1]).unsqueeze(0)
        horizon = torch.tensor(horizons, dtype=torch.long)
        complementary[SEGMENT_ACTION_MASK] = step_indices < horizon.unsqueeze(1)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"config_kwargs": self.config_kwargs}

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_action_flow_v2_4_pre_post_processors(
    config: ActionFlowV24Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build PI0.5 preprocessing plus Plan/box target serialization."""

    validate_action_stats(config, dataset_stats)
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        action_names=config.action_feature_names,
    )
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        relative_step,
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name=config.target_tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        ActionFlowV24TargetTokenizerProcessorStep(config_kwargs=_target_config_kwargs(config)),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline(steps=input_steps, name=POLICY_PREPROCESSOR_DEFAULT_NAME),
        PolicyProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


__all__ = [
    "ActionFlowV24TargetTokenizerProcessorStep",
    "TARGET_TOKENIZER_PROCESSOR_NAME",
    "make_action_flow_v2_4_pre_post_processors",
    "validate_action_stats",
]
