#!/usr/bin/env python

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.action_flow_v2_2.modeling_action_flow_v2_2 import (
    build_branched_attention_mask,
    build_joint_attention_mask,
    pack_autoregressive_targets,
    slice_prefix_key_values,
    weighted_token_cross_entropy,
)


def test_autoregressive_targets_are_compacted_into_independent_branches():
    packed = pack_autoregressive_targets(
        grounding_tokens=torch.tensor([[10, 11, 0, 0], [10, 21, 22, 0]]),
        grounding_mask=torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        ),
        grounding_loss_weight=torch.tensor(
            [[1.0, 0.5, 0.0, 0.0], [1.0, 1.0, 0.25, 0.0]]
        ),
        action_tokens=torch.tensor([[30, 31, 0], [30, 41, 42]]),
        action_mask=torch.tensor(
            [[True, True, False], [True, True, True]]
        ),
        pad_token_id=0,
        grounding_start_token_id=10,
        action_start_token_id=30,
    )

    torch.testing.assert_close(
        packed.grounding_token_ids,
        torch.tensor([[11, 0], [21, 22]]),
    )
    torch.testing.assert_close(
        packed.grounding_token_mask,
        torch.tensor([[True, False], [True, True]]),
    )
    torch.testing.assert_close(
        packed.grounding_loss_weight,
        torch.tensor([[0.5, 0.0], [1.0, 0.25]]),
    )
    torch.testing.assert_close(
        packed.action_token_ids,
        torch.tensor([[31, 0], [41, 42]]),
    )
    torch.testing.assert_close(
        packed.action_token_mask,
        torch.tensor([[True, False], [True, True]]),
    )


def test_joint_attention_prevents_context_leakage_and_keeps_targets_causal():
    attention_mask, position_ids = build_joint_attention_mask(
        prefix_pad_mask=torch.tensor([[True, True, True]]),
        prefix_attention_groups=torch.tensor([[False, False, True]]),
        target_mask=torch.tensor([[True, True, False]]),
    )

    assert attention_mask.shape == (1, 6, 6)
    assert not attention_mask[0, :3, 3:].any()
    assert attention_mask[0, 3, :4].all()
    assert not attention_mask[0, 3, 4]
    assert attention_mask[0, 4, :5].all()
    assert not attention_mask[0, 5].any()
    torch.testing.assert_close(position_ids, torch.tensor([[0, 1, 2, 3, 4, 4]]))


def test_branched_attention_keeps_grounding_and_fast_independent():
    attention_mask, position_ids = build_branched_attention_mask(
        prefix_pad_mask=torch.tensor([[True, True, True]]),
        prefix_attention_groups=torch.tensor([[False, False, True]]),
        grounding_mask=torch.tensor([[True, True, True]]),
        action_mask=torch.tensor([[True, True]]),
    )

    assert attention_mask.shape == (1, 8, 8)
    assert not attention_mask[0, :3, 3:].any()
    assert attention_mask[0, 3:6, :3].all()
    assert attention_mask[0, 6:8, :3].all()
    assert not attention_mask[0, 3:6, 6:8].any()
    assert not attention_mask[0, 6:8, 3:6].any()
    assert attention_mask[0, 3, 3]
    assert not attention_mask[0, 3, 4]
    assert attention_mask[0, 5, 3:6].all()
    assert attention_mask[0, 6, 6]
    assert not attention_mask[0, 6, 7]
    assert attention_mask[0, 7, 6:8].all()
    torch.testing.assert_close(position_ids, torch.tensor([[0, 1, 2, 3, 4, 5, 3, 4]]))


def test_weighted_token_losses_are_reduced_independently():
    logits = torch.tensor(
        [
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        ]
    )
    targets = torch.tensor([[0, 1, 0]])
    weights = torch.tensor([[1.0, 0.5, 0.0]])

    actual = weighted_token_cross_entropy(logits, targets, weights)
    token_losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    expected = (token_losses[:, 0] + 0.5 * token_losses[:, 1]) / 1.5

    torch.testing.assert_close(actual, expected)


def test_detached_prefix_cache_blocks_flow_gradient_to_vlm_side():
    vlm_parameter = torch.nn.Parameter(torch.tensor(2.0))
    expert_parameter = torch.nn.Parameter(torch.tensor(3.0))
    cache = {
        0: {
            "key_states": vlm_parameter * torch.ones(1, 5, 1, 1),
            "value_states": vlm_parameter * torch.ones(1, 5, 1, 1),
        }
    }

    prefix_cache = slice_prefix_key_values(cache, prefix_length=3, detach=True)
    flow_loss = expert_parameter * sum(
        value.sum() for value in prefix_cache[0].values()
    )
    flow_loss.backward()

    assert vlm_parameter.grad is None
    assert expert_parameter.grad is not None
    assert prefix_cache[0]["key_states"].shape[1] == 3
    assert not prefix_cache[0]["key_states"].requires_grad
