#!/usr/bin/env python

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from lerobot.policies.action_flow_v2_2 import (
    ActionFlowV22Config,
    ActionFlowV22TargetTokenizer,
    DecodedTargetBox,
    action_flow_v2_2_added_tokens,
    extend_smolvlm_tokenizer,
    resize_vlm_token_embeddings,
)
from lerobot.policies.action_flow_v2_2.data_contracts_action_flow_v2_2 import (
    TargetRole,
    TargetVisibility,
)


class FakeVLMTokenizer:
    pad_token_id = 0

    def __init__(self, base_size: int) -> None:
        self.base_size = base_size
        self.added_tokens: dict[str, int] = {}
        self.text_ids = {
            " wrist": 101,
            " top": 102,
            " tomato": 103,
            " pink plate": 104,
        }
        self.id_to_text = {token_id: text for text, token_id in self.text_ids.items()}

    def __len__(self) -> int:
        return self.base_size + len(self.added_tokens)

    def get_vocab(self) -> dict[str, int]:
        return {**self.text_ids, **self.added_tokens}

    def add_tokens(self, tokens: list[str], special_tokens: bool = False) -> int:
        assert not special_tokens
        for token in tokens:
            self.added_tokens[token] = len(self)
        return len(tokens)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.added_tokens.get(token, 0)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        if text in self.added_tokens:
            return [self.added_tokens[text]]
        return [self.text_ids[text]]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(self.id_to_text[token_id] for token_id in token_ids)


class FakeFastTokenizer:
    vocab_size = 2048

    def __call__(self, actions: np.ndarray) -> list[list[int]]:
        assert actions.ndim == 3
        return [[0, 17, 2047] for _ in actions]

    def decode(
        self,
        tokens: list[list[int]],
        *,
        time_horizon: int,
        action_dim: int,
    ) -> np.ndarray:
        assert tokens == [[0, 17, 2047]]
        return np.zeros((1, time_horizon, action_dim), dtype=np.float32)


class TinyResizableVLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 4) -> None:
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.output_embeddings = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings

    def resize_token_embeddings(self, size: int):
        old_input = self.input_embeddings
        old_output = self.output_embeddings
        self.input_embeddings = nn.Embedding(size, old_input.embedding_dim)
        self.output_embeddings = nn.Linear(old_output.in_features, size, bias=False)
        with torch.no_grad():
            self.input_embeddings.weight[: old_input.num_embeddings] = old_input.weight
            self.output_embeddings.weight[: old_output.out_features] = old_output.weight
        return self.input_embeddings


def make_target_tokenizer() -> ActionFlowV22TargetTokenizer:
    config = ActionFlowV22Config(device="cpu")
    return ActionFlowV22TargetTokenizer(
        config=config,
        vlm_tokenizer=FakeVLMTokenizer(config.base_vlm_vocab_size),
        action_tokenizer=FakeFastTokenizer(),
        entity_names=("tomato", "pink plate"),
    )


def make_box_labels() -> dict[str, np.ndarray]:
    boxes = np.zeros((2, 8, 4), dtype=np.int64)
    boxes[0, 0] = [10, 20, 30, 40]
    boxes[1, 0] = [11, 21, 31, 41]
    boxes[1, 1] = [50, 60, 70, 80]
    visibility = np.zeros((2, 8), dtype=np.uint8)
    visibility[0, 0] = 1
    visibility[1, :2] = 1
    loss_weight = np.zeros((2, 8), dtype=np.float32)
    loss_weight[:, :2] = 1
    entity_ids = np.full((8,), -1, dtype=np.int64)
    entity_ids[:2] = [0, 1]
    roles = np.zeros((8,), dtype=np.int64)
    roles[:2] = [TargetRole.SOURCE, TargetRole.DESTINATION]
    valid = np.zeros((8,), dtype=np.uint8)
    valid[:2] = 1
    return {
        "target_box_bins": boxes,
        "target_box_visibility": visibility,
        "target_box_loss_weight": loss_weight,
        "target_entity_ids": entity_ids,
        "target_entity_roles": roles,
        "target_entity_valid": valid,
    }


def test_token_extension_uses_deterministic_non_overlapping_ids():
    config = ActionFlowV22Config(device="cpu")
    tokenizer = FakeVLMTokenizer(config.base_vlm_vocab_size)

    extend_smolvlm_tokenizer(tokenizer, config)

    assert len(action_flow_v2_2_added_tokens(config)) == 2316
    assert len(tokenizer) == 51596
    assert tokenizer.convert_tokens_to_ids("<grounding>") == 49280
    assert tokenizer.convert_tokens_to_ids("<loc_000>") == 49292
    assert tokenizer.convert_tokens_to_ids("<loc_255>") == 49547
    assert tokenizer.convert_tokens_to_ids("<fast_0000>") == 49548
    assert tokenizer.convert_tokens_to_ids("<fast_2047>") == 51595
    assert extend_smolvlm_tokenizer(tokenizer, config) is tokenizer


def test_grounding_boxes_round_trip_visible_and_not_visible_targets():
    tokenizer = make_target_tokenizer()
    labels = make_box_labels()

    tokens, weights = tokenizer.serialize_grounding(**labels)
    decoded = tokenizer.deserialize_grounding(tokens)

    assert len(tokens) == len(weights)
    assert decoded == [
        DecodedTargetBox(
            camera="wrist",
            entity_name="tomato",
            entity_role=TargetRole.SOURCE,
            visibility=TargetVisibility.VISIBLE,
            coordinate_bins=(10, 20, 30, 40),
        ),
        DecodedTargetBox(
            camera="wrist",
            entity_name="pink plate",
            entity_role=TargetRole.DESTINATION,
            visibility=TargetVisibility.NOT_VISIBLE,
            coordinate_bins=None,
        ),
        DecodedTargetBox(
            camera="top",
            entity_name="tomato",
            entity_role=TargetRole.SOURCE,
            visibility=TargetVisibility.VISIBLE,
            coordinate_bins=(11, 21, 31, 41),
        ),
        DecodedTargetBox(
            camera="top",
            entity_name="pink plate",
            entity_role=TargetRole.DESTINATION,
            visibility=TargetVisibility.VISIBLE,
            coordinate_bins=(50, 60, 70, 80),
        ),
    ]
    assert decoded[0].camera == "wrist"
    assert decoded[0].entity_name == "tomato"
    assert decoded[0].coordinate_bins == (10, 20, 30, 40)
    assert decoded[1].visibility is TargetVisibility.NOT_VISIBLE
    assert decoded[2].camera == "top"
    assert decoded[3].entity_name == "pink plate"
    assert decoded[3].coordinate_bins == (50, 60, 70, 80)


def test_fast_symbol_mapping_and_fixed_size_sample_serialization_round_trip():
    tokenizer = make_target_tokenizer()
    labels = make_box_labels()
    actions = np.zeros((50, 7), dtype=np.float32)

    serialized = tokenizer.serialize_sample(
        normalized_relative_actions=actions,
        **labels,
    )
    valid_action_ids = serialized.action_tokens[serialized.action_token_mask].tolist()
    symbols = tokenizer.unmap_fast_token_ids(valid_action_ids[1:-1])
    decoded_actions = tokenizer.deserialize_fast_actions(
        serialized.action_tokens,
        token_mask=serialized.action_token_mask,
        time_horizon=50,
        action_dim=7,
    )

    assert symbols == [0, 17, 2047]
    assert tokenizer.map_fast_symbols(symbols) == valid_action_ids[1:-1]
    assert serialized.grounding_tokens.shape == (tokenizer.config.max_grounding_tokens,)
    assert serialized.action_tokens.shape == (tokenizer.config.max_fast_action_tokens,)
    assert decoded_actions.shape == (50, 7)


def test_resize_preserves_base_embedding_rows_and_extends_both_tables():
    config = ActionFlowV22Config(device="cpu")
    model = TinyResizableVLM(config.base_vlm_vocab_size)
    input_rows = model.get_input_embeddings().weight[:3].detach().clone()
    output_rows = model.get_output_embeddings().weight[:3].detach().clone()

    resize_vlm_token_embeddings(model, config)

    assert model.get_input_embeddings().weight.shape[0] == config.required_vlm_vocab_size
    assert model.get_output_embeddings().weight.shape[0] == config.required_vlm_vocab_size
    torch.testing.assert_close(model.get_input_embeddings().weight[:3], input_rows)
    torch.testing.assert_close(model.get_output_embeddings().weight[:3], output_rows)
