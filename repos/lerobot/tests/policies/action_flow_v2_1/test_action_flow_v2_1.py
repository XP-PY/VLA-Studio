#!/usr/bin/env python

from __future__ import annotations

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.action_flow_v2_1.configuration_action_flow_v2_1 import (
    RELEVANCE_LOSS_WEIGHT,
    RELEVANCE_MASK,
    ActionFlowV21Config,
)
from lerobot.policies.action_flow_v2_1.modeling_action_flow_v2_1 import (
    DenseRelevanceDecoder,
    compute_dense_relevance_loss,
    prepare_relevance_targets,
)
from lerobot.policies.factory import get_policy_class, make_policy_config


def test_policy_registration():
    config = make_policy_config("action_flow_v2_1")

    assert isinstance(config, ActionFlowV21Config)
    assert get_policy_class("action_flow_v2_1").name == "action_flow_v2_1"


def test_dataset_metadata_accepts_dense_labels_and_excludes_them_from_inputs():
    config = ActionFlowV21Config(device="cpu")
    config.input_features = {
        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
        "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
        RELEVANCE_MASK: PolicyFeature(FeatureType.STATE, (2, 512, 512)),
        RELEVANCE_LOSS_WEIGHT: PolicyFeature(FeatureType.STATE, (2,)),
    }
    features = {
        RELEVANCE_MASK: {
            "dtype": "uint8",
            "shape": [2, 512, 512],
            "names": ["camera", "height", "width"],
        },
        RELEVANCE_LOSS_WEIGHT: {
            "dtype": "float32",
            "shape": [2],
            "names": ["wrist", "top"],
        },
    }

    config.set_dataset_feature_metadata(features)

    assert RELEVANCE_MASK not in config.input_features
    assert RELEVANCE_LOSS_WEIGHT not in config.input_features


def test_mask_preprocessing_matches_image_resize_and_top_padding():
    targets = torch.ones(1, 1, 1, 480, 640)
    weights = torch.ones(1, 1, 1)

    resized, prepared_weights = prepare_relevance_targets(
        targets,
        weights,
        target_width=512,
        target_height=512,
    )

    assert resized.shape == (1, 1, 512, 512)
    assert torch.count_nonzero(resized[:, :, :128]) == 0
    assert torch.all(resized[:, :, 128:] == 1)
    assert prepared_weights.shape == (1, 1)


def test_dense_decoder_produces_learned_target_resolution():
    decoder = DenseRelevanceDecoder(hidden_size=16, channels=(8, 4))
    tokens = torch.randn(2, 64, 16, requires_grad=True)

    logits = decoder(
        tokens,
        grid_height=8,
        grid_width=8,
        target_height=32,
        target_width=32,
    )

    assert logits.shape == (2, 32, 32)
    logits.mean().backward()
    assert tokens.grad is not None


def test_dense_loss_combines_bce_and_dice_without_resizing():
    logits = torch.zeros(2, 2, 16, 16, requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 0, 4:12, 4:12] = 1
    weights = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

    loss, metrics = compute_dense_relevance_loss(logits, targets, weights)

    assert loss.shape == (2,)
    assert loss[0] > 0
    assert loss[1] == 0
    assert set(metrics) == {
        "relevance_bce_loss",
        "relevance_dice_loss",
        "relevance_valid_camera_fraction",
    }
    loss.mean().backward()
    assert logits.grad is not None


def test_dense_loss_rejects_coarse_logits():
    with pytest.raises(ValueError, match="identical shapes"):
        compute_dense_relevance_loss(
            torch.zeros(1, 2, 8, 8),
            torch.zeros(1, 2, 512, 512),
            torch.ones(1, 2),
        )
