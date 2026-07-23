#!/usr/bin/env python

from __future__ import annotations

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.action_flow_v2.configuration_action_flow_v2 import (
    RELEVANCE_LOSS_WEIGHT,
    RELEVANCE_MASK,
    ActionFlowV2Config,
)
from lerobot.policies.action_flow_v2.modeling_action_flow_v2 import compute_relevance_loss
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.processor import batch_to_transition, transition_to_batch


def test_policy_registration():
    config = make_policy_config("action_flow_v2")

    assert isinstance(config, ActionFlowV2Config)
    assert get_policy_class("action_flow_v2").name == "action_flow_v2"


def test_merged_relevance_labels_survive_batch_conversion():
    batch = {
        "observation.state": torch.zeros(2, 7),
        RELEVANCE_MASK: torch.zeros(2, 2, 8, 8, dtype=torch.uint8),
        RELEVANCE_LOSS_WEIGHT: torch.tensor([[1.0, 1.0], [0.5, 0.0]]),
    }

    output = transition_to_batch(batch_to_transition(batch))

    assert output[RELEVANCE_MASK].shape == (2, 2, 8, 8)
    assert output[RELEVANCE_LOSS_WEIGHT].tolist() == [[1.0, 1.0], [0.5, 0.0]]


def make_dataset_features(camera_names: list[str]) -> dict:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": [f"joint_{index}" for index in range(7)],
        },
        **{
            f"observation.images.{camera}": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            }
            for camera in camera_names
        },
        RELEVANCE_MASK: {
            "dtype": "uint8",
            "shape": [len(camera_names), 8, 8],
            "names": ["camera", "height", "width"],
        },
        RELEVANCE_LOSS_WEIGHT: {
            "dtype": "float32",
            "shape": [len(camera_names)],
            "names": camera_names,
        },
    }


def test_dataset_metadata_excludes_relevance_labels_from_model_inputs():
    config = ActionFlowV2Config(device="cpu")
    config.input_features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
        "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
        RELEVANCE_MASK: PolicyFeature(FeatureType.STATE, (2, 8, 8)),
        RELEVANCE_LOSS_WEIGHT: PolicyFeature(FeatureType.STATE, (2,)),
    }

    config.set_dataset_feature_metadata(make_dataset_features(["wrist", "top"]))

    assert RELEVANCE_MASK not in config.input_features
    assert RELEVANCE_LOSS_WEIGHT not in config.input_features
    assert list(config.image_features) == [
        "observation.images.wrist",
        "observation.images.top",
    ]


def test_dataset_metadata_rejects_wrong_camera_order():
    config = ActionFlowV2Config(device="cpu")
    config.input_features = {
        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
        "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
    }

    with pytest.raises(ValueError, match="camera order"):
        config.set_dataset_feature_metadata(make_dataset_features(["top", "wrist"]))


def test_relevance_loss_ignores_zero_weight_camera():
    logits = torch.zeros(2, 2, 2, 2, requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 0] = 1
    targets[0, 1] = 1
    weights = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

    loss, metrics = compute_relevance_loss(logits, targets, weights)

    expected_bce = torch.log(torch.tensor(2.0))
    expected_dice = torch.tensor(1 / 3)
    torch.testing.assert_close(loss[0], expected_bce + expected_dice)
    torch.testing.assert_close(loss[1], torch.tensor(0.0))
    torch.testing.assert_close(metrics["relevance_bce_loss"], expected_bce)
    torch.testing.assert_close(metrics["relevance_dice_loss"], expected_dice)
    loss.mean().backward()
    assert logits.grad is not None
