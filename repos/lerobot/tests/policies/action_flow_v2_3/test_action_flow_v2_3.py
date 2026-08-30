#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lerobot.policies.action_flow_v2_2.configuration_action_flow_v2_2 import ActionFlowV22Config
from lerobot.policies.action_flow_v2_3.configuration_action_flow_v2_3 import ActionFlowV23Config
from lerobot.policies.action_flow_v2_3.modeling_action_flow_v2_3 import (
    ActionFlowV23Model,
    build_hierarchical_attention_mask,
    compute_fast_token_accuracies,
    pack_autoregressive_targets_v2_3,
    restrict_fast_token_vocabulary,
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


def test_hierarchical_attention_conditions_action_branches_on_plan():
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
    assert not attention[0, 3:6, 6:10].any()
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


def test_fast_token_accuracies_ignore_structural_targets():
    logits = torch.full((2, 3, 10), -10.0)
    targets = torch.tensor([[4, 5, 1], [6, 7, 8]])
    weights = torch.ones((2, 3))

    # Illegal text/grounding logits must not compete with the action vocabulary.
    logits[..., 2] = 100.0

    for batch_index, sequence_index, target in (
        (0, 0, 4),
        (0, 1, 5),
        (1, 0, 6),
        (1, 2, 8),
    ):
        logits[batch_index, sequence_index, target] = 10.0
    logits[1, 1, 9] = 10.0
    logits[1, 1, 7] = 9.0

    top1, top5, exact = compute_fast_token_accuracies(
        logits,
        targets,
        weights,
        action_end_token_id=1,
        fast_token_start_id=4,
        fast_token_stop_id=10,
    )

    torch.testing.assert_close(top1, torch.tensor(0.8))
    torch.testing.assert_close(top5, torch.tensor(1.0))
    torch.testing.assert_close(exact, torch.tensor(0.5))


def test_restrict_fast_token_vocabulary_keeps_end_and_fast_symbols_only():
    logits = torch.arange(30, dtype=torch.float32).reshape(1, 3, 10)
    targets = torch.tensor([[4, 8, 1]])
    weights = torch.ones((1, 3))

    restricted_logits, restricted_targets = restrict_fast_token_vocabulary(
        logits,
        targets,
        weights,
        action_end_token_id=1,
        fast_token_start_id=4,
        fast_token_stop_id=9,
    )

    assert restricted_logits.shape == (1, 3, 6)
    torch.testing.assert_close(restricted_logits[..., 0], logits[..., 1])
    torch.testing.assert_close(restricted_logits[..., 1:], logits[..., 4:9])
    torch.testing.assert_close(restricted_targets, torch.tensor([[1, 5, 0]]))


def test_plan_roundtrip_and_stage_masks_use_local_artifact():
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
    assert sample.action_token_loss_weight.sum() == sample.action_token_mask.sum()
    assert sample.action_token_loss_weight.sum() > 0
    decoded_segment = tokenizer.deserialize_fast_actions(
        sample.action_tokens,
        token_mask=sample.action_token_mask,
        time_horizon=2,
        action_dim=7,
    )
    assert decoded_segment.shape == (2, 7)
    assert sample.grounding_token_loss_weight.sum() == 0

    within_stage = tokenizer.serialize_sample(
        normalized_relative_actions=actions,
        target_box_bins=boxes,
        target_box_visibility=visibility,
        target_box_loss_weight=box_weights,
        target_entity_ids=entity_ids,
        target_entity_roles=roles,
        target_entity_valid=valid,
        planning_stage="grasp",
        planning_target_entity="tomato",
        planning_frames_remaining=4,
    )
    assert within_stage.segment_action_mask.all()
    assert within_stage.action_token_loss_weight.sum() == within_stage.action_token_mask.sum()
    assert within_stage.action_token_loss_weight.sum() > 0


def test_architecture_contract_records_plan_conditioning_and_segment_masks():
    contract = ActionFlowV23Config(device="cpu").architecture_contract()

    assert contract["autoregressive_attention"] == (
        "plan_then_parallel_grounding_and_fast_branches"
    )
    assert contract["expert_context"] == "detached_prefix_and_plan_kv"
    assert contract["segment_aware_action_loss"] is True
    assert contract["restrict_fast_vocabulary"] is True


def test_architecture_contract_records_disabled_knowledge_insulation():
    contract = ActionFlowV23Config(
        device="cpu",
        knowledge_insulation=False,
        detach_expert_prefix_kv=False,
    ).architecture_contract()

    assert contract["knowledge_insulation"] is False
    assert contract["detach_expert_prefix_kv"] is False
    assert contract["expert_context"] == "trainable_prefix_and_plan_kv"


def test_rollout_passes_predicted_plan_to_flow_context():
    predicted_plan = torch.tensor([[101, 102, 103]])
    predicted_mask = torch.tensor([[True, True, True]])
    seen: dict[str, torch.Tensor] = {}

    def cache_plan_context(prefix, prefix_mask, prefix_groups, plan_ids, plan_mask):
        seen["plan_ids"] = plan_ids
        seen["plan_mask"] = plan_mask
        cache = {0: {"key_states": torch.ones(1), "value_states": torch.ones(1)}}
        context_mask = torch.cat([prefix_mask, plan_mask], dim=1)
        return prefix, torch.ones((1, 5, 5), dtype=torch.bool), context_mask, cache

    model = SimpleNamespace(
        config=SimpleNamespace(
            chunk_size=2,
            max_action_dim=3,
            generate_grounding_during_rollout=False,
            num_steps=1,
        ),
        rtc_processor=None,
        last_plan_token_ids=None,
        last_grounding_token_ids=None,
        embed_prefix=lambda *args, **kwargs: (
            torch.zeros((1, 2, 4)),
            torch.tensor([[True, True]]),
            torch.tensor([[False, True]]),
        ),
        _greedy_decode_plan=lambda *args: (predicted_plan, predicted_mask),
        _cache_plan_context=cache_plan_context,
        _rtc_enabled=lambda: False,
        denoise_step=lambda **kwargs: torch.zeros_like(kwargs["x_t"]),
    )
    noise = torch.ones((1, 2, 3))

    actions = ActionFlowV23Model.sample_actions(
        model,
        images=[],
        img_masks=[],
        lang_tokens=torch.zeros((1, 1), dtype=torch.long),
        lang_masks=torch.ones((1, 1), dtype=torch.bool),
        state=torch.zeros((1, 3)),
        noise=noise,
    )

    torch.testing.assert_close(seen["plan_ids"], predicted_plan)
    torch.testing.assert_close(seen["plan_mask"], predicted_mask)
    torch.testing.assert_close(actions, noise)
