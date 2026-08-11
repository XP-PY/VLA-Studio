#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lerobot.policies.action_flow_v2_2.configuration_action_flow_v2_2 import ActionFlowV22Config
from lerobot.policies.action_flow_v2_3.configuration_action_flow_v2_3 import ActionFlowV23Config
from lerobot.policies.action_flow_v2_3.modeling_action_flow_v2_3 import (
    build_hierarchical_attention_mask,
    pack_autoregressive_targets_v2_3,
)
from lerobot.policies.action_flow_v2_3.tokenization_action_flow_v2_3 import (
    ActionFlowV23TargetTokenizer,
)


def test_planning_tokens_append_without_moving_v2_2_ids():
    v2_2 = ActionFlowV22Config(device="cpu")
    v2_3 = ActionFlowV23Config(device="cpu")

    assert v2_3.location_token_start_id == v2_2.location_token_start_id
    assert v2_3.fast_token_start_id == v2_2.fast_token_start_id
    assert v2_3.planning_token_start_id == v2_2.required_vlm_vocab_size
    assert v2_3.required_vlm_vocab_size == v2_2.required_vlm_vocab_size + 5


def test_hierarchical_attention_conditions_branches_on_plan_only():
    attention, positions = build_hierarchical_attention_mask(
        prefix_pad_mask=torch.tensor([[True, True, True]]),
        prefix_attention_groups=torch.tensor([[False, False, True]]),
        plan_mask=torch.tensor([[True, True, True]]),
        grounding_mask=torch.tensor([[True, True]]),
        action_mask=torch.tensor([[True, True]]),
    )

    # Layout: prefix [0:3], plan [3:6], grounding [6:8], action [8:10].
    assert attention.shape == (1, 10, 10)
    assert not attention[0, :3, 3:].any()
    assert attention[0, 3, :4].all()
    assert not attention[0, 3, 4]
    assert attention[0, 6:8, :6].all()
    assert attention[0, 8:10, :6].all()
    assert not attention[0, 6:8, 8:10].any()
    assert not attention[0, 8:10, 6:8].any()
    torch.testing.assert_close(
        positions, torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 6, 7]])
    )


def test_three_autoregressive_branches_remove_fixed_starts():
    packed = pack_autoregressive_targets_v2_3(
        plan_tokens=torch.tensor([[10, 11, 12, 0]]),
        plan_mask=torch.tensor([[True, True, True, False]]),
        plan_loss_weight=torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        grounding_tokens=torch.tensor([[20, 21, 0]]),
        grounding_mask=torch.tensor([[True, True, False]]),
        grounding_loss_weight=torch.tensor([[0.0, 0.0, 0.0]]),
        action_tokens=torch.tensor([[30, 31, 32]]),
        action_mask=torch.tensor([[True, True, True]]),
        action_loss_weight=torch.tensor([[1.0, 0.5, 0.5]]),
        pad_token_id=0,
        plan_start_token_id=10,
        grounding_start_token_id=20,
        action_start_token_id=30,
    )

    torch.testing.assert_close(packed.plan_token_ids, torch.tensor([[11, 12]]))
    torch.testing.assert_close(packed.grounding_token_ids, torch.tensor([[21]]))
    torch.testing.assert_close(packed.grounding_loss_weight, torch.tensor([[0.0]]))
    torch.testing.assert_close(packed.action_token_ids, torch.tensor([[31, 32]]))
    torch.testing.assert_close(packed.action_loss_weight, torch.tensor([[0.5, 0.5]]))


def test_plan_roundtrip_and_segment_masks_use_local_artifact():
    artifact = (
        Path(__file__).resolve().parents[5]
        / "artifacts/tokenizers/action_flow_v2_3_smolvlm_fast"
    )
    config = ActionFlowV23Config(
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        tokenizer_artifact_path=str(artifact),
    )
    tokenizer = ActionFlowV23TargetTokenizer.from_pretrained(
        config.tokenizer_artifact_path, config=config
    )
    plan = tokenizer.serialize_plan(stage="put", target_entity="pink plate")
    decoded = tokenizer.deserialize_plan(plan)
    assert decoded.stage == "put"
    assert decoded.target_entity == "pink plate"

    boxes = np.zeros((2, 8, 4), dtype=np.int64)
    visibility = np.zeros((2, 8), dtype=np.uint8)
    box_weights = np.zeros((2, 8), dtype=np.float32)
    entity_ids = np.full(8, -1, dtype=np.int64)
    roles = np.zeros(8, dtype=np.int64)
    valid = np.zeros(8, dtype=np.uint8)
    actions = np.zeros((4, 7), dtype=np.float32)
    sample = tokenizer.serialize_sample(
        normalized_relative_actions=actions,
        target_box_bins=boxes,
        target_box_visibility=visibility,
        target_box_loss_weight=box_weights,
        target_entity_ids=entity_ids,
        target_entity_roles=roles,
        target_entity_valid=valid,
        planning_stage="grasp",
        planning_target_entity="tomato",
        planning_frames_remaining=2,
    )
    assert sample.segment_action_mask.tolist() == [True, True, False, False]
    assert sample.action_token_loss_weight.sum() == 0
    assert sample.grounding_token_loss_weight.sum() == 0
