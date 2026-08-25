#!/usr/bin/env python

"""PaliGemma-native Plan and bounding-box target serialization for AFV2.4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..action_flow_v2_2.data_contracts_action_flow_v2_2 import TargetRole, TargetVisibility
from .configuration_action_flow_v2_4 import ActionFlowV24Config

PLAN_PREFIX = "Plan:\n"
BOX_PREFIX = "Boxes:\n"


@dataclass(frozen=True)
class SerializedTarget:
    token_ids: Tensor
    token_mask: Tensor
    loss_weight: Tensor


@dataclass(frozen=True)
class PlanPrediction:
    stage: str
    target_entity: str


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class ActionFlowV24TargetTokenizer:
    """Serialize AFV2.4 labels without changing PaliGemma's pretrained vocabulary.

    Plan words use ordinary PaliGemma text tokens. Bounding boxes use the 1,024
    location tokens already pretrained for PaliGemma grounding. PaliGemma uses
    ``y1, x1, y2, x2`` token order, while the dataset stores normalized
    ``x1, y1, x2, y2`` coordinates.
    """

    def __init__(
        self,
        config: ActionFlowV24Config,
        *,
        tokenizer: Any | None = None,
    ) -> None:
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.target_tokenizer_name)
        self.config = config
        self.vlm_tokenizer = tokenizer
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.eos_token_id = int(tokenizer.eos_token_id)
        self._plan_prefix_ids = self._encode(PLAN_PREFIX)
        self._box_prefix_ids = self._encode(BOX_PREFIX)
        self._location_ids = tuple(
            self._single_token_id(f"<loc{index:04d}>")
            for index in range(config.coordinate_token_count)
        )
        self._plan_candidate_sequences = tuple(self._build_plan_candidates())

    def _encode(self, text: str) -> list[int]:
        return [int(item) for item in self.vlm_tokenizer.encode(text, add_special_tokens=False)]

    def _single_token_id(self, token: str) -> int:
        token_id = self.vlm_tokenizer.convert_tokens_to_ids(token)
        if token_id is None or int(token_id) == int(self.vlm_tokenizer.unk_token_id):
            raise ValueError(f"PaliGemma vocabulary does not contain {token!r}.")
        if self.vlm_tokenizer.convert_ids_to_tokens(int(token_id)) != token:
            raise ValueError(f"PaliGemma token lookup did not round-trip for {token!r}.")
        return int(token_id)

    @property
    def plan_prefix_ids(self) -> tuple[int, ...]:
        return tuple(self._plan_prefix_ids)

    @property
    def box_prefix_ids(self) -> tuple[int, ...]:
        return tuple(self._box_prefix_ids)

    @property
    def plan_candidate_sequences(self) -> tuple[tuple[int, ...], ...]:
        return self._plan_candidate_sequences

    def _build_plan_candidates(self) -> list[tuple[int, ...]]:
        candidates = [self.serialize_plan_ids("complete", "")]
        for entity_name in self.config.entity_names:
            candidates.append(self.serialize_plan_ids("grasp", entity_name))
            candidates.append(self.serialize_plan_ids("put", entity_name))
        return sorted(set(candidates))

    def serialize_plan_ids(self, stage: str, target_entity: str) -> tuple[int, ...]:
        stage = stage.strip().lower()
        target_entity = target_entity.strip().lower()
        if stage not in {"grasp", "put", "complete"}:
            raise ValueError(f"Unsupported planning stage: {stage!r}.")
        if stage == "complete":
            target_entity = ""
        elif target_entity not in self.config.entity_names:
            raise ValueError(f"Unknown planning target entity: {target_entity!r}.")
        content = stage if not target_entity else f"{stage} {target_entity}"
        return tuple([*self._plan_prefix_ids, *self._encode(content + "\n"), self.eos_token_id])

    def _pad_target(
        self,
        token_ids: list[int],
        loss_weights: list[float],
        *,
        max_tokens: int,
        name: str,
    ) -> SerializedTarget:
        if len(token_ids) != len(loss_weights):
            raise ValueError(f"{name} IDs and loss weights have different lengths.")
        if len(token_ids) > max_tokens:
            raise ValueError(
                f"{name} target requires {len(token_ids)} tokens, exceeding max_tokens={max_tokens}."
            )
        valid_length = len(token_ids)
        token_ids = [*token_ids, *([self.pad_token_id] * (max_tokens - valid_length))]
        loss_weights = [*loss_weights, *([0.0] * (max_tokens - valid_length))]
        return SerializedTarget(
            token_ids=torch.tensor(token_ids, dtype=torch.long),
            token_mask=torch.arange(max_tokens) < valid_length,
            loss_weight=torch.tensor(loss_weights, dtype=torch.float32),
        )

    def serialize_plan(self, stage: str, target_entity: str) -> SerializedTarget:
        ids = list(self.serialize_plan_ids(stage, target_entity))
        prefix_length = len(self._plan_prefix_ids)
        weights = [0.0] * prefix_length + [1.0] * (len(ids) - prefix_length)
        return self._pad_target(
            ids,
            weights,
            max_tokens=self.config.max_plan_tokens,
            name="Plan",
        )

    def serialize_boxes(
        self,
        *,
        target_boxes: Any,
        target_box_visibility: Any,
        target_box_loss_weight: Any,
        target_entity_ids: Any,
        target_entity_roles: Any,
        target_entity_valid: Any,
    ) -> SerializedTarget:
        boxes = _as_numpy(target_boxes)
        visibility = _as_numpy(target_box_visibility)
        box_weights = _as_numpy(target_box_loss_weight)
        entity_ids = _as_numpy(target_entity_ids).reshape(-1)
        roles = _as_numpy(target_entity_roles).reshape(-1)
        valid = _as_numpy(target_entity_valid).reshape(-1)
        expected_box_shape = (
            len(self.config.camera_names),
            self.config.max_box_entities,
            4,
        )
        if boxes.shape != expected_box_shape:
            raise ValueError(f"Target boxes have shape {boxes.shape}, expected {expected_box_shape}.")

        token_ids = list(self._box_prefix_ids)
        loss_weights = [0.0] * len(token_ids)
        supervised_line_weights: list[float] = []
        role_names = {
            int(TargetRole.SOURCE): "source",
            int(TargetRole.DESTINATION): "destination",
            int(TargetRole.SOURCE_CONTAINER): "source container",
            int(TargetRole.DESTINATION_CONTAINER): "destination container",
        }
        for camera_index, camera_name in enumerate(self.config.camera_names):
            for entity_slot in range(self.config.max_box_entities):
                weight = float(box_weights[camera_index, entity_slot])
                if not bool(valid[entity_slot]) or weight <= 0:
                    continue
                entity_id = int(entity_ids[entity_slot])
                if not 0 <= entity_id < len(self.config.entity_names):
                    raise ValueError(f"Invalid target entity ID {entity_id}.")
                entity_name = self.config.entity_names[entity_id]
                role_name = role_names.get(int(roles[entity_slot]), "target")
                prefix_ids = self._encode(f"{camera_name} {role_name} ")
                if int(visibility[camera_index, entity_slot]) == int(TargetVisibility.VISIBLE):
                    x1, y1, x2, y2 = np.clip(boxes[camera_index, entity_slot], 0.0, 1.0)
                    coordinate_bins = [
                        int(round(value * (self.config.coordinate_token_count - 1)))
                        for value in (y1, x1, y2, x2)
                    ]
                    line_ids = [
                        *prefix_ids,
                        *(self._location_ids[index] for index in coordinate_bins),
                        *self._encode(f" {entity_name}\n"),
                    ]
                else:
                    line_ids = [*prefix_ids, *self._encode(f"not visible {entity_name}\n")]
                token_ids.extend(line_ids)
                loss_weights.extend([weight] * len(line_ids))
                supervised_line_weights.append(weight)

        if not supervised_line_weights:
            none_ids = self._encode("none\n")
            token_ids.extend(none_ids)
            loss_weights.extend([0.0] * len(none_ids))
            eos_weight = 0.0
        else:
            eos_weight = max(supervised_line_weights)
        token_ids.append(self.eos_token_id)
        loss_weights.append(eos_weight)
        return self._pad_target(
            token_ids,
            loss_weights,
            max_tokens=self.config.max_box_tokens,
            name="Box",
        )

    def deserialize_plan(self, token_ids: Tensor | list[int]) -> PlanPrediction:
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.detach().cpu().reshape(-1).tolist()
        token_ids = [int(item) for item in token_ids if int(item) != self.pad_token_id]
        text = self.vlm_tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if text.lower().startswith("plan:"):
            text = text.split(":", maxsplit=1)[1].strip()
        first_line = text.splitlines()[0].strip().lower() if text else ""
        if first_line == "complete":
            return PlanPrediction(stage="complete", target_entity="")
        for stage in ("grasp", "put"):
            prefix = stage + " "
            if first_line.startswith(prefix):
                entity = first_line[len(prefix) :].strip()
                if entity not in self.config.entity_names:
                    raise ValueError(f"Decoded Plan contains unknown entity {entity!r}.")
                return PlanPrediction(stage=stage, target_entity=entity)
        raise ValueError(f"Could not parse Plan target: {text!r}.")


class ActionFlowV24FastTokenizer:
    """Encode stage-local normalized actions into PaliGemma FAST token IDs."""

    def __init__(
        self,
        config: ActionFlowV24Config,
        *,
        action_tokenizer: Any | None = None,
        paligemma_tokenizer: Any | None = None,
    ) -> None:
        if action_tokenizer is None or paligemma_tokenizer is None:
            from transformers import AutoProcessor, AutoTokenizer

            if action_tokenizer is None:
                action_tokenizer = AutoProcessor.from_pretrained(
                    config.fast_tokenizer_name,
                    trust_remote_code=True,
                )
            if paligemma_tokenizer is None:
                paligemma_tokenizer = AutoTokenizer.from_pretrained(
                    config.target_tokenizer_name,
                    trust_remote_code=True,
                )
        self.config = config
        self.action_tokenizer = action_tokenizer
        self.paligemma_tokenizer = paligemma_tokenizer
        self.pad_token_id = int(paligemma_tokenizer.pad_token_id)
        if int(paligemma_tokenizer.vocab_size) != config.paligemma_vocab_size:
            raise ValueError(
                "PaliGemma tokenizer vocabulary does not match the PI0.5 LM head: "
                f"{paligemma_tokenizer.vocab_size} != {config.paligemma_vocab_size}."
            )
        tokenizer_vocab_size = getattr(action_tokenizer, "vocab_size", None)
        if tokenizer_vocab_size is not None and int(tokenizer_vocab_size) != config.fast_vocab_size:
            raise ValueError(
                "FAST tokenizer vocabulary does not match `fast_vocab_size`: "
                f"{tokenizer_vocab_size} != {config.fast_vocab_size}."
            )
        tokenizer_horizon = getattr(action_tokenizer, "time_horizon", None)
        if tokenizer_horizon is not None and int(tokenizer_horizon) != config.chunk_size:
            raise ValueError(
                "FAST tokenizer horizon does not match the policy chunk size: "
                f"{tokenizer_horizon} != {config.chunk_size}."
            )

    def serialize(self, normalized_actions: Tensor | np.ndarray) -> SerializedTarget:
        actions = _as_numpy(normalized_actions).astype(np.float32, copy=False)
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError(
                "FAST expects a non-empty action chunk shaped (time_horizon, action_dim)."
            )
        tokenizer_action_dim = getattr(self.action_tokenizer, "action_dim", None)
        if tokenizer_action_dim is not None and actions.shape[1] != int(tokenizer_action_dim):
            raise ValueError(
                "FAST tokenizer action dimension does not match the action chunk: "
                f"{tokenizer_action_dim} != {actions.shape[1]}."
            )
        raw_tokens = self.action_tokenizer(actions[None, ...])
        raw_tokens = np.asarray(raw_tokens[0], dtype=np.int64).reshape(-1)
        if raw_tokens.size == 0:
            raise ValueError("FAST produced an empty token sequence.")
        if raw_tokens.min() < 0 or raw_tokens.max() >= self.config.fast_vocab_size:
            raise ValueError(
                "FAST symbol IDs fall outside the configured vocabulary: "
                f"[{raw_tokens.min()}, {raw_tokens.max()}] vs {self.config.fast_vocab_size}."
            )
        mapped_tokens = self.config.fast_token_stop_id - 1 - raw_tokens
        token_ids = [
            int(self.paligemma_tokenizer.bos_token_id),
            *self.paligemma_tokenizer.encode("Action: ", add_special_tokens=False),
            *mapped_tokens.tolist(),
            *self.paligemma_tokenizer.encode("|", add_special_tokens=False),
        ]
        if len(token_ids) > self.config.max_fast_tokens:
            raise ValueError(
                f"FAST target requires {len(token_ids)} tokens, exceeding "
                f"max_fast_tokens={self.config.max_fast_tokens}."
            )
        valid_length = len(token_ids)
        padded_ids = [
            *[int(token_id) for token_id in token_ids],
            *([self.pad_token_id] * (self.config.max_fast_tokens - valid_length)),
        ]
        return SerializedTarget(
            token_ids=torch.tensor(padded_ids, dtype=torch.long),
            token_mask=torch.arange(self.config.max_fast_tokens) < valid_length,
            loss_weight=torch.cat(
                [
                    torch.ones(valid_length, dtype=torch.float32),
                    torch.zeros(
                        self.config.max_fast_tokens - valid_length,
                        dtype=torch.float32,
                    ),
                ]
            ),
        )


__all__ = [
    "ActionFlowV24TargetTokenizer",
    "ActionFlowV24FastTokenizer",
    "BOX_PREFIX",
    "PLAN_PREFIX",
    "PlanPrediction",
    "SerializedTarget",
]
