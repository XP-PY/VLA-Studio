#!/usr/bin/env python

"""Tokenizer extension and target serialization for Action Flow V2.2."""

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

from .configuration_action_flow_v2_2 import ActionFlowV22Config
from .data_contracts_action_flow_v2_2 import TargetRole, TargetVisibility

TOKENIZER_MANIFEST_NAME = "action_flow_v2_2_tokenizer.json"
VLM_TOKENIZER_DIR = "vlm_tokenizer"
FAST_TOKENIZER_DIR = "fast_action_tokenizer"


@dataclass(frozen=True)
class DecodedTargetBox:
    """One target reconstructed from a serialized grounding sequence."""

    camera: str
    entity_name: str
    entity_role: TargetRole
    visibility: TargetVisibility
    coordinate_bins: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class SerializedTargets:
    """Fixed-size teacher-forcing tensors for one training sample."""

    grounding_tokens: torch.Tensor
    grounding_token_mask: torch.Tensor
    grounding_token_loss_weight: torch.Tensor
    action_tokens: torch.Tensor
    action_token_mask: torch.Tensor


def action_flow_v2_2_added_tokens(config: ActionFlowV22Config) -> list[str]:
    """Return all added tokens in their deterministic ID-allocation order."""

    return [
        *config.structural_tokens,
        *(config.location_token(index) for index in range(config.coordinate_token_count)),
        *(config.fast_token(index) for index in range(config.fast_token_count)),
    ]


def _token_digest(tokens: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def extend_smolvlm_tokenizer(tokenizer: Any, config: ActionFlowV22Config) -> Any:
    """Extend a base SmolVLM tokenizer without changing any existing token ID.

    The function is idempotent: it accepts either the unmodified tokenizer or
    an already extended tokenizer, but rejects partial or differently ordered
    extensions.
    """

    expected_tokens = action_flow_v2_2_added_tokens(config)
    current_size = len(tokenizer)
    expected_size = config.required_vlm_vocab_size
    if current_size not in (config.base_vlm_vocab_size, expected_size):
        raise ValueError(
            f"SmolVLM tokenizer has {current_size} tokens; expected the base size "
            f"{config.base_vlm_vocab_size} or extended size {expected_size}."
        )

    existing_vocab = tokenizer.get_vocab()
    existing = [token for token in expected_tokens if token in existing_vocab]
    if current_size == config.base_vlm_vocab_size:
        if existing:
            raise ValueError(
                "The base tokenizer already contains part of the Action Flow V2.2 "
                f"extension: {existing[:5]}."
            )
        added = tokenizer.add_tokens(expected_tokens, special_tokens=False)
        if added != len(expected_tokens):
            raise RuntimeError(
                f"Added {added} tokens, expected {len(expected_tokens)}."
            )

    if len(tokenizer) != expected_size:
        raise RuntimeError(
            f"Extended tokenizer has {len(tokenizer)} tokens, expected {expected_size}."
        )
    for offset, token in enumerate(expected_tokens):
        expected_id = config.base_vlm_vocab_size + offset
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != expected_id:
            raise ValueError(
                f"Token {token!r} has ID {token_id}; expected deterministic ID {expected_id}."
            )
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [expected_id]:
            raise ValueError(
                f"Token {token!r} does not encode atomically: {encoded}."
            )
    return tokenizer


def validate_vlm_token_embeddings(model: Any, config: ActionFlowV22Config) -> None:
    """Validate that both VLM embedding tables cover the extended vocabulary."""

    vlm = _find_resizable_vlm(model)
    input_embeddings = vlm.get_input_embeddings()
    output_embeddings = vlm.get_output_embeddings()
    expected = config.required_vlm_vocab_size
    if input_embeddings is None or input_embeddings.weight.shape[0] != expected:
        actual = None if input_embeddings is None else input_embeddings.weight.shape[0]
        raise ValueError(f"VLM input embedding size is {actual}, expected {expected}.")
    if output_embeddings is None or output_embeddings.weight.shape[0] != expected:
        actual = None if output_embeddings is None else output_embeddings.weight.shape[0]
        raise ValueError(f"VLM output embedding size is {actual}, expected {expected}.")


def resize_vlm_token_embeddings(model: Any, config: ActionFlowV22Config) -> Any:
    """Resize SmolVLM input embeddings and LM head for the extended tokenizer.

    Existing rows retain their IDs and weights. Newly allocated rows are left
    to the Hugging Face model's standard initialization.
    """

    vlm = _find_resizable_vlm(model)
    input_embeddings = vlm.get_input_embeddings()
    if input_embeddings is None:
        raise ValueError("The supplied VLM has no input embedding table.")
    current_size = input_embeddings.weight.shape[0]
    expected_size = config.required_vlm_vocab_size
    if current_size not in (config.base_vlm_vocab_size, expected_size):
        raise ValueError(
            f"VLM input embedding size is {current_size}; expected "
            f"{config.base_vlm_vocab_size} or {expected_size}."
        )
    if current_size != expected_size:
        vlm.resize_token_embeddings(expected_size)
    validate_vlm_token_embeddings(vlm, config)
    return model


def _find_resizable_vlm(model: Any) -> Any:
    candidates = [
        model,
        getattr(model, "vlm", None),
        getattr(getattr(model, "vlm_with_expert", None), "vlm", None),
    ]
    for candidate in candidates:
        if candidate is not None and all(
            hasattr(candidate, name)
            for name in (
                "resize_token_embeddings",
                "get_input_embeddings",
                "get_output_embeddings",
            )
        ):
            return candidate
    raise TypeError("Could not find a resizable Hugging Face VLM on the supplied model.")


class ActionFlowV22TargetTokenizer:
    """Serialize target boxes and normalized relative actions into VLM IDs."""

    def __init__(
        self,
        *,
        config: ActionFlowV22Config,
        vlm_tokenizer: Any,
        action_tokenizer: Any,
        entity_names: Sequence[str],
    ) -> None:
        self.config = config
        self.vlm_tokenizer = extend_smolvlm_tokenizer(vlm_tokenizer, config)
        self.action_tokenizer = action_tokenizer
        self.entity_names = tuple(entity_names)
        if not self.entity_names or any(not name.strip() for name in self.entity_names):
            raise ValueError("Entity names must be non-empty.")
        if len(set(self.entity_names)) != len(self.entity_names):
            raise ValueError("Entity names must be unique.")
        self._text_token_cache: dict[str, tuple[int, ...]] = {}

        fast_vocab_size = getattr(action_tokenizer, "vocab_size", None)
        if fast_vocab_size != config.fast_token_count:
            raise ValueError(
                f"FAST tokenizer vocabulary size is {fast_vocab_size}; "
                f"expected {config.fast_token_count}."
            )
        self._role_token_ids = {
            TargetRole.SOURCE: config.structural_token_id("<source>"),
            TargetRole.DESTINATION: config.structural_token_id("<destination>"),
            TargetRole.SOURCE_CONTAINER: config.structural_token_id("<source_container>"),
            TargetRole.DESTINATION_CONTAINER: config.structural_token_id(
                "<destination_container>"
            ),
        }
        self._token_id_to_role = {
            token_id: role for role, token_id in self._role_token_ids.items()
        }

    @classmethod
    def from_pretrained_sources(
        cls,
        *,
        config: ActionFlowV22Config,
        entity_names: Sequence[str],
        vlm_tokenizer_name: str | Path | None = None,
        action_tokenizer_name: str | Path | None = None,
        local_files_only: bool = False,
    ) -> ActionFlowV22TargetTokenizer:
        """Load the base tokenizers and apply the deterministic extension."""

        from transformers import AutoProcessor, AutoTokenizer

        vlm_source = _resolve_pretrained_source(
            vlm_tokenizer_name or config.vlm_tokenizer_path,
            local_files_only=local_files_only,
        )
        action_source = _resolve_pretrained_source(
            action_tokenizer_name or config.action_tokenizer_name,
            local_files_only=local_files_only,
        )
        vlm_tokenizer = AutoTokenizer.from_pretrained(
            vlm_source,
            local_files_only=local_files_only,
        )
        action_tokenizer = AutoProcessor.from_pretrained(
            action_source,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        return cls(
            config=config,
            vlm_tokenizer=vlm_tokenizer,
            action_tokenizer=action_tokenizer,
            entity_names=entity_names,
        )

    def map_fast_symbols(self, symbols: Sequence[int] | np.ndarray) -> list[int]:
        """Map FAST vocabulary symbols to dedicated SmolVLM token IDs."""

        mapped = []
        for symbol in np.asarray(symbols, dtype=np.int64).reshape(-1).tolist():
            if not 0 <= symbol < self.config.fast_token_count:
                raise ValueError(
                    f"FAST symbol {symbol} is outside [0, {self.config.fast_token_count})."
                )
            mapped.append(self.config.fast_token_id(symbol))
        return mapped

    def unmap_fast_token_ids(self, token_ids: Sequence[int] | np.ndarray) -> list[int]:
        """Recover FAST symbols from dedicated SmolVLM token IDs."""

        symbols = []
        start = self.config.fast_token_start_id
        stop = self.config.required_vlm_vocab_size
        for token_id in np.asarray(token_ids, dtype=np.int64).reshape(-1).tolist():
            if not start <= token_id < stop:
                raise ValueError(f"Token ID {token_id} is not an Action Flow V2.2 FAST token.")
            symbols.append(token_id - start)
        return symbols

    def serialize_grounding(
        self,
        *,
        target_box_bins: torch.Tensor | np.ndarray,
        target_box_visibility: torch.Tensor | np.ndarray,
        target_box_loss_weight: torch.Tensor | np.ndarray,
        target_entity_ids: torch.Tensor | np.ndarray,
        target_entity_roles: torch.Tensor | np.ndarray,
        target_entity_valid: torch.Tensor | np.ndarray,
    ) -> tuple[list[int], list[float]]:
        """Serialize one sample's ordered camera/entity box annotations."""

        boxes = _as_numpy(target_box_bins, np.int64)
        visibility = _as_numpy(target_box_visibility, np.int64)
        loss_weight = _as_numpy(target_box_loss_weight, np.float32)
        entity_ids = _as_numpy(target_entity_ids, np.int64)
        roles = _as_numpy(target_entity_roles, np.int64)
        valid = _as_numpy(target_entity_valid, np.bool_)
        camera_count = len(self.config.camera_names)
        entity_count = self.config.max_box_entities
        expected_shapes = {
            "target_box_bins": (camera_count, entity_count, 4),
            "target_box_visibility": (camera_count, entity_count),
            "target_box_loss_weight": (camera_count, entity_count),
            "target_entity_ids": (entity_count,),
            "target_entity_roles": (entity_count,),
            "target_entity_valid": (entity_count,),
        }
        actual_shapes = {
            "target_box_bins": boxes.shape,
            "target_box_visibility": visibility.shape,
            "target_box_loss_weight": loss_weight.shape,
            "target_entity_ids": entity_ids.shape,
            "target_entity_roles": roles.shape,
            "target_entity_valid": valid.shape,
        }
        for name, expected_shape in expected_shapes.items():
            if actual_shapes[name] != expected_shape:
                raise ValueError(
                    f"`{name}` has shape {actual_shapes[name]}, expected {expected_shape}."
                )

        tokens = [self.config.structural_token_id("<grounding>")]
        weights = [1.0]
        for camera_index, camera_name in enumerate(self.config.camera_names):
            self._append(tokens, weights, [self.config.structural_token_id("<camera>")])
            self._append(tokens, weights, self._encode_text(camera_name))
            for entity_index in range(entity_count):
                if not valid[entity_index] or loss_weight[camera_index, entity_index] <= 0:
                    continue
                entity_id = int(entity_ids[entity_index])
                if not 0 <= entity_id < len(self.entity_names):
                    raise ValueError(f"Entity ID {entity_id} is outside the entity vocabulary.")
                role = TargetRole(int(roles[entity_index]))
                if role not in self._role_token_ids:
                    raise ValueError(f"Entity slot {entity_index} has unsupported role {role.name}.")
                entity_weight = float(loss_weight[camera_index, entity_index])
                segment = [
                    self._role_token_ids[role],
                    *self._encode_text(self.entity_names[entity_id]),
                ]
                target_visibility = TargetVisibility(
                    int(visibility[camera_index, entity_index])
                )
                if target_visibility is TargetVisibility.VISIBLE:
                    coordinate_bins = boxes[camera_index, entity_index]
                    if np.any(coordinate_bins < 0) or np.any(
                        coordinate_bins >= self.config.coordinate_token_count
                    ):
                        raise ValueError(
                            f"Coordinate bins are outside [0, "
                            f"{self.config.coordinate_token_count}): {coordinate_bins.tolist()}."
                        )
                    segment.extend(
                        [
                            self.config.structural_token_id("<box>"),
                            *(
                                self.config.location_token_id(int(coordinate))
                                for coordinate in coordinate_bins
                            ),
                        ]
                    )
                else:
                    segment.append(self.config.structural_token_id("<not_visible>"))
                self._append(tokens, weights, segment, weight=entity_weight)
            self._append(tokens, weights, [self.config.structural_token_id("</camera>")])
        self._append(tokens, weights, [self.config.structural_token_id("</grounding>")])
        if len(tokens) > self.config.max_grounding_tokens:
            raise ValueError(
                f"Grounding target uses {len(tokens)} tokens, exceeding "
                f"`max_grounding_tokens={self.config.max_grounding_tokens}`."
            )
        return tokens, weights

    def deserialize_grounding(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        token_mask: Sequence[bool] | torch.Tensor | None = None,
    ) -> list[DecodedTargetBox]:
        """Parse a serialized grounding target back into structured boxes."""

        ids = _masked_token_ids(token_ids, token_mask)
        start_id = self.config.structural_token_id("<grounding>")
        end_id = self.config.structural_token_id("</grounding>")
        camera_start_id = self.config.structural_token_id("<camera>")
        camera_end_id = self.config.structural_token_id("</camera>")
        box_id = self.config.structural_token_id("<box>")
        not_visible_id = self.config.structural_token_id("<not_visible>")
        if len(ids) < 2 or ids[0] != start_id or ids[-1] != end_id:
            raise ValueError("Grounding tokens must start with `<grounding>` and end with `</grounding>`.")

        decoded: list[DecodedTargetBox] = []
        position = 1
        cameras = []
        while position < len(ids) - 1:
            if ids[position] != camera_start_id:
                raise ValueError(f"Expected `<camera>` at token position {position}.")
            position += 1
            camera_name_ids = []
            while position < len(ids) and ids[position] not in {
                camera_end_id,
                *self._token_id_to_role,
            }:
                camera_name_ids.append(ids[position])
                position += 1
            camera_name = self._decode_text(camera_name_ids)
            if not camera_name:
                raise ValueError("Serialized camera name cannot be empty.")
            cameras.append(camera_name)

            while position < len(ids) and ids[position] != camera_end_id:
                role_id = ids[position]
                if role_id not in self._token_id_to_role:
                    raise ValueError(f"Expected an entity role at token position {position}.")
                role = self._token_id_to_role[role_id]
                position += 1
                entity_name_ids = []
                while position < len(ids) and ids[position] not in {box_id, not_visible_id}:
                    if ids[position] in {camera_start_id, camera_end_id, end_id}:
                        raise ValueError("Entity name is missing a box or not-visible marker.")
                    entity_name_ids.append(ids[position])
                    position += 1
                entity_name = self._decode_text(entity_name_ids)
                if position >= len(ids):
                    raise ValueError("Unexpected end of grounding target.")

                if ids[position] == not_visible_id:
                    visibility = TargetVisibility.NOT_VISIBLE
                    coordinate_bins = None
                    position += 1
                else:
                    visibility = TargetVisibility.VISIBLE
                    position += 1
                    if position + 4 > len(ids):
                        raise ValueError("Visible target is missing coordinate tokens.")
                    coordinate_ids = ids[position : position + 4]
                    coordinate_bins = tuple(
                        self._location_id_to_bin(token_id) for token_id in coordinate_ids
                    )
                    position += 4
                decoded.append(
                    DecodedTargetBox(
                        camera=camera_name,
                        entity_name=entity_name,
                        entity_role=role,
                        visibility=visibility,
                        coordinate_bins=coordinate_bins,
                    )
                )
            if position >= len(ids) or ids[position] != camera_end_id:
                raise ValueError(f"Camera {camera_name!r} is missing `</camera>`.")
            position += 1

        if tuple(cameras) != tuple(self.config.camera_names):
            raise ValueError(
                f"Serialized camera order {tuple(cameras)} does not match "
                f"{tuple(self.config.camera_names)}."
            )
        return decoded

    def serialize_fast_actions(
        self,
        normalized_relative_actions: torch.Tensor | np.ndarray,
    ) -> tuple[list[int], list[int]]:
        """FAST-tokenize one normalized relative action chunk."""

        actions = _as_numpy(normalized_relative_actions, np.float32)
        if actions.ndim != 2 or actions.shape[-1] != self.config.expected_action_dim:
            raise ValueError(
                "Expected one action chunk with shape "
                f"(chunk_size, {self.config.expected_action_dim}), got {actions.shape}."
            )
        encoded = self.action_tokenizer(actions[None, ...])
        if len(encoded) != 1:
            raise RuntimeError(f"FAST tokenizer returned {len(encoded)} sequences for one chunk.")
        symbols = np.asarray(encoded[0], dtype=np.int64).reshape(-1).tolist()
        mapped = self.map_fast_symbols(symbols)
        tokens = [
            self.config.structural_token_id("<action>"),
            *mapped,
            self.config.structural_token_id("<action_end>"),
        ]
        if len(tokens) > self.config.max_fast_action_tokens:
            raise ValueError(
                f"FAST target uses {len(tokens)} tokens, exceeding "
                f"`max_fast_action_tokens={self.config.max_fast_action_tokens}`."
            )
        return tokens, symbols

    def deserialize_fast_actions(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        *,
        time_horizon: int,
        action_dim: int,
        token_mask: Sequence[bool] | torch.Tensor | None = None,
    ) -> np.ndarray:
        """Decode dedicated VLM FAST tokens into one continuous action chunk."""

        ids = _masked_token_ids(token_ids, token_mask)
        action_start = self.config.structural_token_id("<action>")
        action_end = self.config.structural_token_id("<action_end>")
        if len(ids) < 2 or ids[0] != action_start or ids[-1] != action_end:
            raise ValueError("FAST tokens must start with `<action>` and end with `<action_end>`.")
        symbols = self.unmap_fast_token_ids(ids[1:-1])
        decoded = self.action_tokenizer.decode(
            [symbols],
            time_horizon=time_horizon,
            action_dim=action_dim,
        )
        return np.asarray(decoded[0], dtype=np.float32)

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
    ) -> SerializedTargets:
        """Serialize and pad all Stage 4 targets for one sample."""

        grounding_tokens, grounding_weights = self.serialize_grounding(
            target_box_bins=target_box_bins,
            target_box_visibility=target_box_visibility,
            target_box_loss_weight=target_box_loss_weight,
            target_entity_ids=target_entity_ids,
            target_entity_roles=target_entity_roles,
            target_entity_valid=target_entity_valid,
        )
        action_tokens, _ = self.serialize_fast_actions(normalized_relative_actions)
        grounding_ids, grounding_mask = self._pad(
            grounding_tokens,
            self.config.max_grounding_tokens,
        )
        grounding_loss_weight = torch.zeros(
            self.config.max_grounding_tokens,
            dtype=torch.float32,
        )
        grounding_loss_weight[: len(grounding_weights)] = torch.tensor(
            grounding_weights,
            dtype=torch.float32,
        )
        action_ids, action_mask = self._pad(
            action_tokens,
            self.config.max_fast_action_tokens,
        )
        return SerializedTargets(
            grounding_tokens=grounding_ids,
            grounding_token_mask=grounding_mask,
            grounding_token_loss_weight=grounding_loss_weight,
            action_tokens=action_ids,
            action_token_mask=action_mask,
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        """Save both tokenizers and their deterministic mapping manifest."""

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

        added_tokens = action_flow_v2_2_added_tokens(self.config)
        manifest = {
            "format": "action_flow_v2_2_tokenizer",
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
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        config: ActionFlowV22Config,
    ) -> ActionFlowV22TargetTokenizer:
        """Reload and verify a saved Action Flow V2.2 tokenizer artifact."""

        from transformers import AutoProcessor, AutoTokenizer

        root = Path(path)
        manifest = json.loads((root / TOKENIZER_MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("format") != "action_flow_v2_2_tokenizer" or manifest.get("version") != 1:
            raise ValueError("Unsupported Action Flow V2.2 tokenizer manifest.")
        if manifest.get("token_contract") != config.token_contract():
            raise ValueError("Saved tokenizer token contract does not match the policy configuration.")
        if tuple(manifest.get("camera_names", ())) != tuple(config.camera_names):
            raise ValueError("Saved tokenizer camera order does not match the policy configuration.")
        if manifest.get("fast_action_vocab_size") != config.fast_token_count:
            raise ValueError("Saved FAST vocabulary size does not match the policy configuration.")
        expected_tokens = action_flow_v2_2_added_tokens(config)
        if manifest.get("added_token_count") != len(expected_tokens):
            raise ValueError("Saved added-token count does not match the policy configuration.")
        if manifest.get("added_tokens_sha256") != _token_digest(expected_tokens):
            raise ValueError("Saved tokenizer added-token digest does not match the configuration.")

        vlm_tokenizer = AutoTokenizer.from_pretrained(
            root / manifest["vlm_tokenizer_dir"],
            local_files_only=True,
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

    def _encode_text(self, text: str) -> list[int]:
        cleaned = text.strip()
        cached = self._text_token_cache.get(cleaned)
        if cached is not None:
            return list(cached)
        encoded = self.vlm_tokenizer.encode(f" {cleaned}", add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Text segment {text!r} produced no tokens.")
        self._text_token_cache[cleaned] = tuple(encoded)
        return encoded

    def _decode_text(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            raise ValueError("Serialized text segment cannot be empty.")
        return self.vlm_tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _location_id_to_bin(self, token_id: int) -> int:
        start = self.config.location_token_start_id
        stop = self.config.fast_token_start_id
        if not start <= token_id < stop:
            raise ValueError(f"Token ID {token_id} is not a coordinate token.")
        return token_id - start

    def _append(
        self,
        tokens: list[int],
        weights: list[float],
        segment: Sequence[int],
        *,
        weight: float = 1.0,
    ) -> None:
        tokens.extend(int(token_id) for token_id in segment)
        weights.extend([float(weight)] * len(segment))

    def _pad(self, token_ids: Sequence[int], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if len(token_ids) > max_length:
            raise ValueError(f"Cannot pad {len(token_ids)} tokens to length {max_length}.")
        pad_id = self.vlm_tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("The VLM tokenizer must define a padding token.")
        ids = torch.full((max_length,), pad_id, dtype=torch.long)
        mask = torch.zeros((max_length,), dtype=torch.bool)
        ids[: len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        mask[: len(token_ids)] = True
        return ids, mask


def _resolve_pretrained_source(source: str | Path, *, local_files_only: bool) -> str:
    path = Path(source).expanduser()
    if path.exists():
        return str(path.resolve())
    if local_files_only:
        from huggingface_hub import snapshot_download

        return snapshot_download(str(source), local_files_only=True)
    return str(source)


def _as_numpy(value: torch.Tensor | np.ndarray, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _masked_token_ids(
    token_ids: Sequence[int] | torch.Tensor,
    token_mask: Sequence[bool] | torch.Tensor | None,
) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        ids = token_ids.detach().cpu().reshape(-1).tolist()
    else:
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1).tolist()
    if token_mask is None:
        return [int(token_id) for token_id in ids]
    if isinstance(token_mask, torch.Tensor):
        mask = token_mask.detach().cpu().reshape(-1).bool().tolist()
    else:
        mask = np.asarray(token_mask, dtype=np.bool_).reshape(-1).tolist()
    if len(ids) != len(mask):
        raise ValueError(f"Token and mask lengths differ: {len(ids)} != {len(mask)}.")
    return [int(token_id) for token_id, is_valid in zip(ids, mask, strict=True) if is_valid]


__all__ = [
    "ActionFlowV22TargetTokenizer",
    "DecodedTargetBox",
    "FAST_TOKENIZER_DIR",
    "SerializedTargets",
    "TOKENIZER_MANIFEST_NAME",
    "VLM_TOKENIZER_DIR",
    "action_flow_v2_2_added_tokens",
    "extend_smolvlm_tokenizer",
    "resize_vlm_token_embeddings",
    "validate_vlm_token_embeddings",
]
