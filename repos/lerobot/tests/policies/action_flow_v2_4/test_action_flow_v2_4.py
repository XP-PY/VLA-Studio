#!/usr/bin/env python

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.action_flow_v2_2.data_contracts_action_flow_v2_2 import (
    TargetRole,
    TargetVisibility,
)
from lerobot.policies.action_flow_v2_4.configuration_action_flow_v2_4 import ActionFlowV24Config
from lerobot.policies.action_flow_v2_4.modeling_action_flow_v2_4 import (
    ActionFlowV24Model,
    build_action_flow_v2_4_attention_mask,
    weighted_token_cross_entropy,
)
from lerobot.policies.action_flow_v2_4.processor_action_flow_v2_4 import validate_action_stats
from lerobot.policies.action_flow_v2_4.tokenization_action_flow_v2_4 import (
    ActionFlowV24FastTokenizer,
    ActionFlowV24TargetTokenizer,
)
from lerobot.utils.constants import ACTION


class FakePaliGemmaTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2
    bos_token_id = 3
    vocab_size = 257_152

    def __init__(self) -> None:
        self._chars: dict[str, int] = {}
        self._reverse: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        match = re.fullmatch(r"<loc(\d{4})>", text)
        if match:
            return [10_000 + int(match.group(1))]
        result = []
        for char in text:
            if char not in self._chars:
                token_id = 100 + len(self._chars)
                self._chars[char] = token_id
                self._reverse[token_id] = char
            result.append(self._chars[char])
        return result

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        result = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in {self.pad_token_id, self.eos_token_id} and skip_special_tokens:
                continue
            result.append(self._reverse.get(token_id, f"<loc{token_id - 10_000:04d}>"))
        return "".join(result)

    def convert_tokens_to_ids(self, token: str) -> int:
        match = re.fullmatch(r"<loc(\d{4})>", token)
        return 10_000 + int(match.group(1)) if match else self.unk_token_id

    def convert_ids_to_tokens(self, token_id: int) -> str:
        if 10_000 <= token_id < 11_024:
            return f"<loc{token_id - 10_000:04d}>"
        return self._reverse.get(token_id, "<unk>")


def make_config(**kwargs) -> ActionFlowV24Config:
    kwargs.setdefault("entity_names", ("tomato", "blue plate"))
    kwargs.setdefault("max_plan_tokens", 64)
    kwargs.setdefault("max_box_tokens", 128)
    return ActionFlowV24Config(
        device="cpu",
        pi05_pretrained_path=None,
        **kwargs,
    )


def test_attention_conditions_boxes_and_actions_on_plan_without_cross_dependency():
    attention, positions = build_action_flow_v2_4_attention_mask(
        shared_mask=torch.tensor([[True, True]]),
        plan_mask=torch.tensor([[True, True, True]]),
        box_mask=torch.tensor([[True, True]]),
        fast_mask=torch.tensor([[True, True]]),
        action_mask=torch.tensor([[True, True]]),
    )

    # Layout: shared [0:2], Plan [2:5], Boxes [5:7], FAST [7:9], actions [9:11].
    assert attention.shape == (1, 11, 11)
    assert attention[0, :2, :2].all()
    assert not attention[0, :2, 2:].any()
    assert attention[0, 2:5, :2].all()
    assert attention[0, 2, 2]
    assert not attention[0, 2, 3]
    assert attention[0, 5:7, :5].all()
    assert not attention[0, 5:7, 7:11].any()
    assert attention[0, 7:9, :5].all()
    assert not attention[0, 7:9, 5:7].any()
    assert not attention[0, 7, 8]
    assert attention[0, 9:11, :5].all()
    assert not attention[0, 9:11, 5:9].any()
    assert attention[0, 9:11, 9:11].all()
    torch.testing.assert_close(
        positions,
        torch.tensor([[0, 1, 2, 3, 4, 5, 6, 5, 6, 5, 6]]),
    )


def test_target_tokenizer_roundtrips_plan_and_uses_paligemma_yxyx_locations():
    config = make_config()
    tokenizer = FakePaliGemmaTokenizer()
    codec = ActionFlowV24TargetTokenizer(config, tokenizer=tokenizer)

    plan = codec.serialize_plan("grasp", "tomato")
    decoded = codec.deserialize_plan(plan.token_ids)
    assert decoded.stage == "grasp"
    assert decoded.target_entity == "tomato"
    assert plan.loss_weight[: len(codec.plan_prefix_ids)].sum() == 0
    assert plan.loss_weight.sum() > 0

    boxes = np.zeros((2, 8, 4), dtype=np.float32)
    boxes[0, 0] = [0.10, 0.20, 0.60, 0.80]  # xyxy in the dataset
    visibility = np.zeros((2, 8), dtype=np.uint8)
    visibility[0, 0] = int(TargetVisibility.VISIBLE)
    weights = np.zeros((2, 8), dtype=np.float32)
    weights[0, 0] = 1.0
    entity_ids = np.full(8, -1, dtype=np.int64)
    entity_ids[0] = 0
    roles = np.zeros(8, dtype=np.int64)
    roles[0] = int(TargetRole.SOURCE)
    valid = np.zeros(8, dtype=np.uint8)
    valid[0] = 1

    target = codec.serialize_boxes(
        target_boxes=boxes,
        target_box_visibility=visibility,
        target_box_loss_weight=weights,
        target_entity_ids=entity_ids,
        target_entity_roles=roles,
        target_entity_valid=valid,
    )
    valid_ids = target.token_ids[target.token_mask].tolist()
    expected_yxyx = [
        10_000 + round(0.20 * 1023),
        10_000 + round(0.10 * 1023),
        10_000 + round(0.80 * 1023),
        10_000 + round(0.60 * 1023),
    ]
    location_start = next(index for index, token_id in enumerate(valid_ids) if token_id >= 10_000)
    assert valid_ids[location_start : location_start + 4] == expected_yxyx


def test_target_tokenizer_canonicalizes_reviewed_container_entity_alias():
    config = make_config(
        entity_names=("tomato", "blue plate", "vegetable on yellow plate")
    )
    codec = ActionFlowV24TargetTokenizer(config, tokenizer=FakePaliGemmaTokenizer())

    plan = codec.serialize_plan("grasp", " vegetable  on  the yellow plate ")
    decoded = codec.deserialize_plan(plan.token_ids)

    assert decoded.stage == "grasp"
    assert decoded.target_entity == "vegetable on yellow plate"


def test_weighted_token_loss_ignores_zero_weight_targets():
    logits = torch.full((2, 2, 5), -10.0)
    targets = torch.tensor([[1, 2], [3, 4]])
    weights = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0

    loss, accuracy = weighted_token_cross_entropy(logits, targets, weights)

    assert loss[0] < 1e-4
    assert loss[1] == 0
    torch.testing.assert_close(accuracy, torch.tensor([1.0, 0.0]))


def test_fast_tokenizer_maps_rm65_symbols_into_paligemma_reserved_range():
    class FakeActionTokenizer:
        def __call__(self, actions):
            assert actions.shape == (1, 5, 7)
            return [[0, 3, 2_047]]

    config = make_config(max_fast_tokens=32)
    tokenizer = FakePaliGemmaTokenizer()
    codec = ActionFlowV24FastTokenizer(
        config,
        action_tokenizer=FakeActionTokenizer(),
        paligemma_tokenizer=tokenizer,
    )

    target = codec.serialize(torch.zeros(5, 7))
    valid_ids = target.token_ids[target.token_mask].tolist()

    assert config.fast_token_start_id == 254_976
    assert config.fast_token_stop_id == 257_024
    assert [257_023, 257_020, 254_976] == [
        token_id for token_id in valid_ids if config.fast_token_start_id <= token_id < 257_024
    ]


def test_restricted_fast_loss_keeps_gradients_for_symbol_and_text_positions():
    hidden_size = 4
    lm_head = torch.nn.Linear(hidden_size, 12, bias=False)
    dummy = SimpleNamespace(
        config=SimpleNamespace(
            fast_token_start_id=5,
            fast_token_stop_id=9,
            restrict_fast_vocabulary=True,
        ),
        paligemma_with_expert=SimpleNamespace(
            paligemma=SimpleNamespace(lm_head=lm_head),
        ),
    )
    hidden = torch.randn(1, 4, hidden_size, requires_grad=True)
    tokens = torch.tensor([[0, 5, 6, 2]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    weights = torch.ones_like(tokens, dtype=torch.float32)

    loss, top1, top5, exact = ActionFlowV24Model._training_fast_token_loss(
        dummy,
        hidden,
        tokens,
        mask,
        weights,
    )
    loss.sum().backward()

    assert torch.isfinite(loss).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert top1.shape == top5.shape == exact.shape == (1,)


def test_config_records_pi05_multitask_contract():
    contract = make_config().architecture_contract()

    assert contract["base_policy"] == "pi05"
    assert contract["autoregressive_targets"] == [
        "current_plan",
        "stage_target_boxes",
        "stage_local_fast_actions",
    ]
    assert contract["flow_gradient_updates_vlm"] is True


def test_action_stats_use_dataset_action_dimension_instead_of_afv22_fixed_dimension():
    config = make_config()
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    config.action_feature_names = [*(f"joint_{index}.pos" for index in range(1, 7)), "gripper.pos"]
    stats = {
        ACTION: {
            name: np.ones(7, dtype=np.float32)
            for name in ("mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99")
        }
    }

    validate_action_stats(config, stats)
