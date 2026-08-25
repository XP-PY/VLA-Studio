#!/usr/bin/env python

"""Tokenizer extension and structured targets for Action Flow V2.3."""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..action_flow_v2_2.tokenization_action_flow_v2_2 import (
    FAST_TOKENIZER_DIR,
    VLM_TOKENIZER_DIR,
    ActionFlowV22TargetTokenizer,
    action_flow_v2_2_added_tokens,
)
from .configuration_action_flow_v2_3 import ActionFlowV23Config

TOKENIZER_MANIFEST_NAME = "action_flow_v2_3_tokenizer.json"


@dataclass(frozen=True)
class DecodedPlan:
    """Structured next-step plan reconstructed from VLM token IDs."""

    stage: str
    target_entity: str


@dataclass(frozen=True)
class SerializedTargetsV23:
    """Fixed-size plan, grounding, and FAST targets for one sample."""

    plan_tokens: torch.Tensor
    plan_token_mask: torch.Tensor
    plan_token_loss_weight: torch.Tensor
    grounding_tokens: torch.Tensor
    grounding_token_mask: torch.Tensor
    grounding_token_loss_weight: torch.Tensor
    action_tokens: torch.Tensor
    action_token_mask: torch.Tensor
    action_token_loss_weight: torch.Tensor
    segment_action_mask: torch.Tensor


def action_flow_v2_3_added_tokens(config: ActionFlowV23Config) -> list[str]:
    """Return the complete extension while preserving the AFV2.2 prefix order."""

    return [*action_flow_v2_2_added_tokens(config), *config.planning_tokens]


def _token_digest(tokens: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def extend_action_flow_v2_3_tokenizer(tokenizer: Any, config: ActionFlowV23Config) -> Any:
    """Append planning tokens to a base or fully extended AFV2.2 tokenizer."""

    all_tokens = action_flow_v2_3_added_tokens(config)
    v2_2_tokens = action_flow_v2_2_added_tokens(config)
    current_size = len(tokenizer)
    supported_sizes = {
        config.base_vlm_vocab_size,
        config.action_flow_v2_2_vocab_size,
        config.required_vlm_vocab_size,
    }
    if current_size not in supported_sizes:
        raise ValueError(
            f"SmolVLM tokenizer has {current_size} tokens; expected one of "
            f"{sorted(supported_sizes)}."
        )

    existing_vocab = tokenizer.get_vocab()
    if current_size == config.base_vlm_vocab_size:
        existing = [token for token in all_tokens if token in existing_vocab]
        if existing:
            raise ValueError(f"Base tokenizer already contains AFV2 tokens: {existing[:5]}.")
        expected_to_add = all_tokens
    elif current_size == config.action_flow_v2_2_vocab_size:
        for offset, token in enumerate(v2_2_tokens):
            expected_id = config.base_vlm_vocab_size + offset
            if tokenizer.convert_tokens_to_ids(token) != expected_id:
                raise ValueError(f"AFV2.2 token {token!r} does not have ID {expected_id}.")
        unexpected = [token for token in config.planning_tokens if token in existing_vocab]
        if unexpected:
            raise ValueError(f"AFV2.2 tokenizer already contains planning tokens: {unexpected}.")
        expected_to_add = list(config.planning_tokens)
    else:
        expected_to_add = []

    if expected_to_add:
        added = tokenizer.add_tokens(expected_to_add, special_tokens=False)
        if added != len(expected_to_add):
            raise RuntimeError(f"Added {added} tokens, expected {len(expected_to_add)}.")

    if len(tokenizer) != config.required_vlm_vocab_size:
        raise RuntimeError(
            f"Extended tokenizer has {len(tokenizer)} tokens, expected "
            f"{config.required_vlm_vocab_size}."
        )
    for offset, token in enumerate(all_tokens):
        expected_id = config.base_vlm_vocab_size + offset
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != expected_id:
            raise ValueError(f"Token {token!r} has ID {token_id}; expected {expected_id}.")
        if tokenizer.encode(token, add_special_tokens=False) != [expected_id]:
            raise ValueError(f"Token {token!r} does not encode atomically.")
    return tokenizer


class ActionFlowV23TargetTokenizer(ActionFlowV22TargetTokenizer):
    """Serialize plans, boxes, and relative action chunks into stable VLM IDs."""

    config: ActionFlowV23Config

    def __init__(
        self,
        *,
        config: ActionFlowV23Config,
        vlm_tokenizer: Any,
        action_tokenizer: Any,
        entity_names: Sequence[str],
    ) -> None:
        extend_action_flow_v2_3_tokenizer(vlm_tokenizer, config)
        super().__init__(
            config=config,
            vlm_tokenizer=vlm_tokenizer,
            action_tokenizer=action_tokenizer,
            entity_names=entity_names,
        )

    def serialize_plan(self, *, stage: str, target_entity: str) -> list[int]:
        """Serialize one reviewed stage as ``<stage> [entity] </plan>``."""

        normalized_stage = str(stage).strip().lower()
        target = str(target_entity).strip().lower()
        tokens = [self.config.structural_token_id("<plan>")]
        if normalized_stage == "grasp":
            if not target:
                raise ValueError("A grasp plan requires `target_entity`.")
            tokens.extend(
                [self.config.structural_token_id("<grasp>"), *self._encode_text(target)]
            )
        elif normalized_stage == "put":
            if not target:
                raise ValueError("A put plan requires `target_entity`.")
            tokens.extend(
                [self.config.structural_token_id("<put>"), *self._encode_text(target)]
            )
        elif normalized_stage == "complete":
            tokens.append(self.config.structural_token_id("<complete>"))
        else:
            raise ValueError(f"Unsupported planning stage: {stage!r}.")
        tokens.append(self.config.structural_token_id("</plan>"))
        if len(tokens) > self.config.max_plan_tokens:
            raise ValueError(
                f"Plan target uses {len(tokens)} tokens, exceeding "
                f"`max_plan_tokens={self.config.max_plan_tokens}`."
            )
        return tokens

    def unmap_fast_token_ids(self, token_ids: Sequence[int] | np.ndarray) -> list[int]:
        """Recover FAST symbols without accepting appended planning-token IDs."""

        symbols = []
        start = self.config.fast_token_start_id
        stop = self.config.action_flow_v2_2_vocab_size
        for token_id in np.asarray(token_ids, dtype=np.int64).reshape(-1).tolist():
            if not start <= token_id < stop:
                raise ValueError(f"Token ID {token_id} is not an Action Flow V2.3 FAST token.")
            symbols.append(token_id - start)
        return symbols

    def deserialize_plan(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        token_mask: Sequence[bool] | torch.Tensor | None = None,
    ) -> DecodedPlan:
        """Parse one complete planning sequence into a stage and target entity."""

        ids = self._masked_ids(token_ids, token_mask)
        plan_start = self.config.structural_token_id("<plan>")
        plan_end = self.config.structural_token_id("</plan>")
        if len(ids) < 3 or ids[0] != plan_start or ids[-1] != plan_end:
            raise ValueError("Plan tokens must start with `<plan>` and end with `</plan>`.")
        stage_ids = {
            self.config.structural_token_id("<grasp>"): "grasp",
            self.config.structural_token_id("<put>"): "put",
            self.config.structural_token_id("<complete>"): "complete",
        }
        if ids[1] not in stage_ids:
            raise ValueError("Plan does not begin with a supported stage token.")
        stage = stage_ids[ids[1]]
        entity_ids = ids[2:-1]
        if stage == "complete":
            if entity_ids:
                raise ValueError("A complete plan cannot contain a target entity.")
            target = ""
        else:
            target = self._decode_text(entity_ids)
        return DecodedPlan(stage=stage, target_entity=target)

    def serialize_sample(
        self,
        *,
        normalized_relative_actions: torch.Tensor | np.ndarray,
        target_box_bins: torch.Tensor | np.ndarray,
        target_box_visibility: torch.Tensor | np.ndarray,
        target_box_loss_weight: torch.Tensor | np.ndarray,
        target_entity_ids: torch.Tensor | np.ndarray,
        target_entity_roles: torch.Tensor | np.ndarray,
        target_entity_valid: torch.Tensor | np.ndarray,
        planning_stage: str,
        planning_target_entity: str,
        planning_frames_remaining: int,
    ) -> SerializedTargetsV23:
        """Serialize plan-conditioned targets and construct stage-boundary masks."""

        plan_tokens = self.serialize_plan(
            stage=planning_stage,
            target_entity=planning_target_entity,
        )
        grounding_tokens, grounding_weights = self.serialize_grounding(
            target_box_bins=target_box_bins,
            target_box_visibility=target_box_visibility,
            target_box_loss_weight=target_box_loss_weight,
            target_entity_ids=target_entity_ids,
            target_entity_roles=target_entity_roles,
            target_entity_valid=target_entity_valid,
        )
        remaining = max(0, int(planning_frames_remaining))
        valid_steps = min(remaining, self.config.chunk_size)
        if self.config.segment_aware_action_loss and valid_steps == 0:
            raise ValueError("Segment-aware action supervision requires at least one remaining frame.")
        fast_actions = normalized_relative_actions
        if self.config.segment_aware_action_loss:
            fast_actions = normalized_relative_actions[:valid_steps]
        action_tokens, _ = self.serialize_fast_actions(fast_actions)

        plan_ids, plan_mask = self._pad(plan_tokens, self.config.max_plan_tokens)
        grounding_ids, grounding_mask = self._pad(
            grounding_tokens, self.config.max_grounding_tokens
        )
        action_ids, action_mask = self._pad(
            action_tokens, self.config.max_fast_action_tokens
        )
        plan_weights = plan_mask.float()
        grounding_loss_weight = torch.zeros(
            self.config.max_grounding_tokens, dtype=torch.float32
        )
        grounding_loss_weight[: len(grounding_weights)] = torch.tensor(
            grounding_weights, dtype=torch.float32
        )

        segment_action_mask = torch.arange(self.config.chunk_size) < valid_steps
        if not self.config.segment_aware_action_loss:
            segment_action_mask.fill_(True)

        action_loss_weight = action_mask.float()

        return SerializedTargetsV23(
            plan_tokens=plan_ids,
            plan_token_mask=plan_mask,
            plan_token_loss_weight=plan_weights,
            grounding_tokens=grounding_ids,
            grounding_token_mask=grounding_mask,
            grounding_token_loss_weight=grounding_loss_weight,
            action_tokens=action_ids,
            action_token_mask=action_mask,
            action_token_loss_weight=action_loss_weight,
            segment_action_mask=segment_action_mask,
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        """Save the VLM/FAST tokenizers and the V2.3 allocation manifest."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        vlm_path = output_path / VLM_TOKENIZER_DIR
        fast_path = output_path / FAST_TOKENIZER_DIR
        self.vlm_tokenizer.save_pretrained(vlm_path)
        self.action_tokenizer.save_pretrained(fast_path)

        source_path = Path(inspect.getfile(type(self.action_tokenizer)))
        if source_path.is_file():
            destination = fast_path / source_path.name
            if not destination.exists():
                shutil.copy2(source_path, destination)

        added_tokens = action_flow_v2_3_added_tokens(self.config)
        manifest = {
            "format": "action_flow_v2_3_tokenizer",
            "version": 1,
            "camera_names": list(self.config.camera_names),
            "entity_names": list(self.entity_names),
            "token_contract": self.config.token_contract(),
            "added_token_count": len(added_tokens),
            "added_tokens_sha256": _token_digest(added_tokens),
            "vlm_tokenizer_dir": VLM_TOKENIZER_DIR,
            "fast_action_tokenizer_dir": FAST_TOKENIZER_DIR,
            "fast_action_vocab_size": self.config.fast_token_count,
        }
        (output_path / TOKENIZER_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        config: ActionFlowV23Config,
    ) -> ActionFlowV23TargetTokenizer:
        """Reload and verify a saved Action Flow V2.3 tokenizer artifact."""

        from transformers import AutoProcessor, AutoTokenizer

        root = Path(path)
        manifest = json.loads((root / TOKENIZER_MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("format") != "action_flow_v2_3_tokenizer" or manifest.get("version") != 1:
            raise ValueError("Unsupported Action Flow V2.3 tokenizer manifest.")
        if manifest.get("token_contract") != config.token_contract():
            raise ValueError("Saved tokenizer token contract does not match the V2.3 config.")
        if tuple(manifest.get("camera_names", ())) != tuple(config.camera_names):
            raise ValueError("Saved tokenizer camera order does not match the policy config.")
        expected_tokens = action_flow_v2_3_added_tokens(config)
        if manifest.get("added_token_count") != len(expected_tokens):
            raise ValueError("Saved added-token count does not match the V2.3 config.")
        if manifest.get("added_tokens_sha256") != _token_digest(expected_tokens):
            raise ValueError("Saved tokenizer token digest does not match the V2.3 config.")

        vlm_tokenizer = AutoTokenizer.from_pretrained(
            root / manifest["vlm_tokenizer_dir"], local_files_only=True
        )
        action_tokenizer = AutoProcessor.from_pretrained(
            root / manifest["fast_action_tokenizer_dir"],
            trust_remote_code=True,
            local_files_only=True,
        )
        return cls(
            config=config,
            vlm_tokenizer=vlm_tokenizer,
            action_tokenizer=action_tokenizer,
            entity_names=manifest["entity_names"],
        )

    @staticmethod
    def _masked_ids(
        token_ids: Sequence[int] | torch.Tensor,
        token_mask: Sequence[bool] | torch.Tensor | None,
    ) -> list[int]:
        ids = (
            token_ids.detach().cpu().reshape(-1).tolist()
            if isinstance(token_ids, torch.Tensor)
            else np.asarray(token_ids, dtype=np.int64).reshape(-1).tolist()
        )
        if token_mask is None:
            return [int(value) for value in ids]
        mask = (
            token_mask.detach().cpu().reshape(-1).bool().tolist()
            if isinstance(token_mask, torch.Tensor)
            else np.asarray(token_mask, dtype=np.bool_).reshape(-1).tolist()
        )
        if len(ids) != len(mask):
            raise ValueError("Plan token IDs and mask have different lengths.")
        return [int(value) for value, valid in zip(ids, mask, strict=True) if valid]


__all__ = [
    "ActionFlowV23TargetTokenizer",
    "DecodedPlan",
    "FAST_TOKENIZER_DIR",
    "SerializedTargetsV23",
    "TOKENIZER_MANIFEST_NAME",
    "VLM_TOKENIZER_DIR",
    "action_flow_v2_3_added_tokens",
    "extend_action_flow_v2_3_tokenizer",
]
