#!/usr/bin/env python

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.action_flow_v2_2.configuration_action_flow_v2_2 import (
    ActionFlowV22Config,
)
from lerobot.policies.action_flow_v2_2.processor_action_flow_v2_2 import (
    make_action_flow_v2_2_pre_post_processors,
    validate_relative_action_stats,
)
from lerobot.policies.action_flow_v2_2.data_contracts_action_flow_v2_2 import (
    TARGET_BOX_BINS,
    TARGET_BOX_LOSS_WEIGHT,
    TARGET_BOX_VISIBILITY,
    TARGET_ENTITY_IDS,
    TARGET_ENTITY_ROLES,
    TARGET_ENTITY_VALID,
)
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.utils.constants import ACTION, OBS_STATE

ACTION_NAMES = [
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
    "gripper.pos",
]


def make_config() -> ActionFlowV22Config:
    return ActionFlowV22Config(
        device="cpu",
        chunk_size=3,
        n_action_steps=3,
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (7,)),
        },
        output_features={
            ACTION: PolicyFeature(FeatureType.ACTION, (7,)),
        },
        action_feature_names=ACTION_NAMES,
    )


def feature_stats(mean: np.ndarray, std: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": mean,
        "std": std,
        "min": mean - 3 * std,
        "max": mean + 3 * std,
        "q01": mean - 2 * std,
        "q10": mean - std,
        "q50": mean,
        "q90": mean + std,
        "q99": mean + 2 * std,
        "count": np.array([1000]),
    }


def make_stats() -> dict[str, dict[str, np.ndarray]]:
    return {
        ACTION: feature_stats(
            np.array([0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 500.0], dtype=np.float32),
            np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 200.0], dtype=np.float32),
        ),
        OBS_STATE: feature_stats(
            np.array(
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 500.0],
                dtype=np.float32,
            ),
            np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 200.0], dtype=np.float32),
        ),
    }


def make_absolute_batch() -> dict[str, torch.Tensor]:
    state = torch.tensor(
        [
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 450.0],
            [-5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 700.0],
        ]
    )
    offsets = torch.tensor(
        [
            [[0.5, 1.0, -1.0, 2.0, -2.0, 0.25, 100.0],
             [1.0, 1.5, -0.5, 2.5, -1.5, 0.50, 200.0],
             [1.5, 2.0, 0.0, 3.0, -1.0, 0.75, 300.0]],
            [[-0.5, -1.0, 1.0, -2.0, 2.0, -0.25, 800.0],
             [-1.0, -1.5, 0.5, -2.5, 1.5, -0.50, 700.0],
             [-1.5, -2.0, 0.0, -3.0, 1.0, -0.75, 600.0]],
        ]
    )
    action = offsets.clone()
    action[..., :6] += state[:, None, :6]
    box_bins = torch.zeros(2, 2, 8, 4, dtype=torch.long)
    box_bins[:, :, 0] = torch.tensor([10, 20, 30, 40])
    visibility = torch.zeros(2, 2, 8, dtype=torch.uint8)
    visibility[:, :, 0] = 1
    loss_weight = torch.zeros(2, 2, 8)
    loss_weight[:, :, 0] = 1
    entity_ids = torch.full((2, 8), -1, dtype=torch.long)
    entity_ids[:, 0] = 6  # tomato
    roles = torch.zeros(2, 8, dtype=torch.long)
    roles[:, 0] = 1  # source
    valid = torch.zeros(2, 8, dtype=torch.uint8)
    valid[:, 0] = 1
    return {
        OBS_STATE: state,
        ACTION: action,
        TARGET_BOX_BINS: box_bins,
        TARGET_BOX_VISIBILITY: visibility,
        TARGET_BOX_LOSS_WEIGHT: loss_weight,
        TARGET_ENTITY_IDS: entity_ids,
        TARGET_ENTITY_ROLES: roles,
        TARGET_ENTITY_VALID: valid,
        "task": ["Grasp the tomato.", "Grasp the tomato."],
    }


def test_stage3_pipeline_uses_six_relative_joints_and_absolute_gripper():
    config = make_config()
    stats = make_stats()
    preprocessor, postprocessor = make_action_flow_v2_2_pre_post_processors(
        config,
        stats,
    )
    batch = make_absolute_batch()
    original_action = batch[ACTION].clone()

    processed = preprocessor(batch)
    action_mean = torch.tensor(stats[ACTION]["mean"])
    action_std = torch.tensor(stats[ACTION]["std"])
    expected_relative = original_action.clone()
    expected_relative[..., :6] -= batch[OBS_STATE][:, None, :6]
    expected_normalized = (expected_relative - action_mean) / action_std

    torch.testing.assert_close(processed[ACTION], expected_normalized)
    recovered = postprocessor(processed[ACTION])
    torch.testing.assert_close(recovered, original_action)

    relative_step = next(
        step for step in preprocessor.steps if isinstance(step, RelativeActionsProcessorStep)
    )
    absolute_step = next(
        step for step in postprocessor.steps if isinstance(step, AbsoluteActionsProcessorStep)
    )
    assert relative_step._build_mask(7) == [True, True, True, True, True, True, False]
    assert absolute_step.relative_step is relative_step


def test_factory_builds_stage3_processors():
    preprocessor, postprocessor = make_pre_post_processors(
        make_config(),
        dataset_stats=make_stats(),
    )

    assert any(isinstance(step, RelativeActionsProcessorStep) for step in preprocessor.steps)
    assert any(isinstance(step, AbsoluteActionsProcessorStep) for step in postprocessor.steps)


def test_inference_observation_caches_state_without_action_labels():
    stats = make_stats()
    preprocessor, postprocessor = make_pre_post_processors(
        make_config(),
        dataset_stats=stats,
    )
    state = make_absolute_batch()[OBS_STATE]
    preprocessor(
        {
            OBS_STATE: state,
            "task": ["Grasp the tomato.", "Grasp the tomato."],
        }
    )

    relative_action = torch.tensor(
        [
            [[0.5, -1.0, 2.0, -0.25, 4.0, -3.0, 300.0]],
            [[-0.5, 1.0, -2.0, 0.25, -4.0, 3.0, 800.0]],
        ]
    )
    action_mean = torch.tensor(stats[ACTION]["mean"])
    action_std = torch.tensor(stats[ACTION]["std"])
    normalized_action = (relative_action - action_mean) / action_std

    absolute_action = postprocessor(normalized_action)
    expected = relative_action.clone()
    expected[..., :6] += state[:, None, :6]

    torch.testing.assert_close(absolute_action, expected)


def test_saved_processors_reload_and_reconnect_relative_state(tmp_path):
    config = make_config()
    stats = make_stats()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        dataset_stats=stats,
    )
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(tmp_path),
    )
    batch = make_absolute_batch()
    original_action = batch[ACTION].clone()
    processed = loaded_preprocessor(batch)
    recovered = loaded_postprocessor(processed[ACTION])

    torch.testing.assert_close(recovered, original_action)


def test_stats_validation_rejects_missing_or_wrong_action_stats():
    config = make_config()
    with pytest.raises(ValueError, match="requires relative-action statistics"):
        validate_relative_action_stats(config, None)

    stats = make_stats()
    stats[ACTION]["mean"] = np.zeros(6)
    with pytest.raises(ValueError, match="expected"):
        validate_relative_action_stats(config, stats)


def test_processor_requires_dataset_action_names():
    config = make_config()
    config.action_feature_names = None

    with pytest.raises(ValueError, match="requires action names"):
        make_action_flow_v2_2_pre_post_processors(config, make_stats())
