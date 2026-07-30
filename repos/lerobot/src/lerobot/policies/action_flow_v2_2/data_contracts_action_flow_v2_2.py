#!/usr/bin/env python

"""Stable dataset and model-batch contracts for Action Flow V2.2."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

from lerobot.utils.constants import ACTION_TOKEN_MASK, ACTION_TOKENS

TARGET_BOXES = "observation.target_boxes"
TARGET_BOX_BINS = "observation.target_box_bins"
TARGET_BOX_VISIBILITY = "observation.target_box_visibility"
TARGET_BOX_LOSS_WEIGHT = "observation.target_box_loss_weight"
TARGET_ENTITY_IDS = "observation.target_entity_ids"
TARGET_ENTITY_ROLES = "observation.target_entity_roles"
TARGET_ENTITY_VALID = "observation.target_entity_valid"

BOX_FEATURE_KEYS = (
    TARGET_BOXES,
    TARGET_BOX_BINS,
    TARGET_BOX_VISIBILITY,
    TARGET_BOX_LOSS_WEIGHT,
    TARGET_ENTITY_IDS,
    TARGET_ENTITY_ROLES,
    TARGET_ENTITY_VALID,
)
BOX_COORDINATE_NAMES = ("x1", "y1", "x2", "y2")

GROUNDING_TOKENS = "observation.grounding_tokens"
GROUNDING_TOKEN_MASK = "observation.grounding_token_mask"
GROUNDING_TOKEN_LOSS_WEIGHT = "observation.grounding_token_loss_weight"


class TargetRole(IntEnum):
    """Prompt-relative role assigned to one target entity."""

    PADDING = 0
    SOURCE = 1
    DESTINATION = 2
    SOURCE_CONTAINER = 3
    DESTINATION_CONTAINER = 4


class TargetVisibility(IntEnum):
    """Whether a valid entity is visible in a particular camera."""

    NOT_VISIBLE = 0
    VISIBLE = 1


@dataclass(frozen=True)
class TargetBoxRecord:
    """One entity-specific box annotation before it is packed into fixed tensors.

    Coordinates use normalized ``xyxy`` values in the padded policy image. A
    validly absent entity has ``visibility=NOT_VISIBLE`` and ``box_xyxy=None``.
    An uncertain annotation remains distinguishable through ``loss_weight=0``.
    """

    episode_index: int
    frame_index: int
    camera: str
    entity_id: int
    entity_name: str
    entity_role: TargetRole
    visibility: TargetVisibility
    box_xyxy: tuple[float, float, float, float] | None
    loss_weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_role", TargetRole(self.entity_role))
        object.__setattr__(self, "visibility", TargetVisibility(self.visibility))

        if self.episode_index < 0 or self.frame_index < 0:
            raise ValueError("Episode and frame indices must be non-negative.")
        if not self.camera.strip():
            raise ValueError("`camera` must be non-empty.")
        if self.entity_id < 0 or not self.entity_name.strip():
            raise ValueError("Entity ID must be non-negative and entity name must be non-empty.")
        if self.entity_role is TargetRole.PADDING:
            raise ValueError("A sidecar record cannot use the padding role.")
        if not math.isfinite(self.loss_weight) or self.loss_weight < 0:
            raise ValueError("`loss_weight` must be finite and non-negative.")

        if self.visibility is TargetVisibility.NOT_VISIBLE:
            if self.box_xyxy is not None:
                raise ValueError("A not-visible entity must not contain box coordinates.")
            return

        if self.box_xyxy is None:
            raise ValueError("A visible entity requires `box_xyxy`.")
        if len(self.box_xyxy) != 4 or not all(math.isfinite(value) for value in self.box_xyxy):
            raise ValueError("`box_xyxy` must contain four finite coordinates.")

        x1, y1, x2, y2 = self.box_xyxy
        if not all(0.0 <= value <= 1.0 for value in self.box_xyxy):
            raise ValueError("Box coordinates must be normalized to [0, 1].")
        if x1 > x2 or y1 > y2:
            raise ValueError("Box coordinates must follow xyxy ordering.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        data = asdict(self)
        data["entity_role"] = int(self.entity_role)
        data["visibility"] = int(self.visibility)
        data["box_xyxy"] = list(self.box_xyxy) if self.box_xyxy is not None else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetBoxRecord:
        """Construct a validated record from its JSON representation."""

        values = dict(data)
        box = values.get("box_xyxy")
        values["box_xyxy"] = tuple(box) if box is not None else None
        return cls(**values)


@dataclass(frozen=True)
class ActionFlowV22DatasetMetadata:
    """Dataset-wide identity and camera-order metadata for target-box labels."""

    camera_names: tuple[str, ...]
    entity_names: tuple[str, ...]
    max_entities: int
    schema_version: int = 1
    coordinate_bins: int = 256
    coordinate_space: str = "policy_image_normalized_xyxy"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported Action Flow V2.2 box schema version: {self.schema_version}.")
        if not self.camera_names or any(not name.strip() for name in self.camera_names):
            raise ValueError("`camera_names` must contain non-empty names.")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("`camera_names` must be unique and ordered.")
        if any(not name.strip() for name in self.entity_names):
            raise ValueError("`entity_names` must contain non-empty names.")
        if len(set(self.entity_names)) != len(self.entity_names):
            raise ValueError("`entity_names` must be unique.")
        if self.max_entities <= 0:
            raise ValueError("`max_entities` must be positive.")
        if self.coordinate_bins < 2:
            raise ValueError("`coordinate_bins` must be at least 2.")
        if self.coordinate_space != "policy_image_normalized_xyxy":
            raise ValueError(
                "Action Flow V2.2 boxes must use `policy_image_normalized_xyxy` coordinates."
            )

    def feature_metadata(self) -> dict[str, dict[str, Any]]:
        """Return LeRobot feature metadata for fixed-size box tensors.

        Tensor dimensions are ``C=len(camera_names)`` and
        ``E=max_entities``. Entity slots with ``target_entity_valid=0`` are
        padding. A camera/entity annotation with ``loss_weight=0`` is invalid
        and excluded from box-token loss; a valid absent entity instead uses
        ``visibility=NOT_VISIBLE``.
        """

        camera_count = len(self.camera_names)
        shared_metadata = {
            "camera_names": list(self.camera_names),
            "entity_names": list(self.entity_names),
            "schema_version": self.schema_version,
            "coordinate_bins": self.coordinate_bins,
        }
        return {
            TARGET_BOXES: {
                "dtype": "float32",
                "shape": [camera_count, self.max_entities, 4],
                "names": ["camera", "entity", "coordinate"],
                "coordinate_names": list(BOX_COORDINATE_NAMES),
                **shared_metadata,
            },
            TARGET_BOX_BINS: {
                "dtype": "int64",
                "shape": [camera_count, self.max_entities, 4],
                "names": ["camera", "entity", "coordinate"],
                "coordinate_names": list(BOX_COORDINATE_NAMES),
                **shared_metadata,
            },
            TARGET_BOX_VISIBILITY: {
                "dtype": "uint8",
                "shape": [camera_count, self.max_entities],
                "names": ["camera", "entity"],
                **shared_metadata,
            },
            TARGET_BOX_LOSS_WEIGHT: {
                "dtype": "float32",
                "shape": [camera_count, self.max_entities],
                "names": ["camera", "entity"],
                **shared_metadata,
            },
            TARGET_ENTITY_IDS: {
                "dtype": "int64",
                "shape": [self.max_entities],
                "names": ["entity"],
                "entity_names": list(self.entity_names),
                "schema_version": self.schema_version,
            },
            TARGET_ENTITY_ROLES: {
                "dtype": "int64",
                "shape": [self.max_entities],
                "names": ["entity"],
                "role_ids": {role.name.lower(): int(role) for role in TargetRole},
                "schema_version": self.schema_version,
            },
            TARGET_ENTITY_VALID: {
                "dtype": "uint8",
                "shape": [self.max_entities],
                "names": ["entity"],
                "schema_version": self.schema_version,
            },
        }

    def validate_feature_metadata(self, features: dict[str, dict[str, Any]]) -> None:
        """Validate dataset features against this contract."""

        expected = self.feature_metadata()
        missing = set(expected).difference(features)
        if missing:
            raise ValueError(f"Missing Action Flow V2.2 box features: {sorted(missing)}.")

        for key, expected_feature in expected.items():
            actual = features[key]
            if actual.get("dtype") != expected_feature["dtype"]:
                raise ValueError(
                    f"`{key}` has dtype {actual.get('dtype')!r}, expected {expected_feature['dtype']!r}."
                )
            if tuple(actual.get("shape", ())) != tuple(expected_feature["shape"]):
                raise ValueError(
                    f"`{key}` has shape {actual.get('shape')}, expected {expected_feature['shape']}."
                )

        saved_camera_names = tuple(features[TARGET_BOXES].get("camera_names", ()))
        if saved_camera_names != self.camera_names:
            raise ValueError(
                "Target-box camera order does not match the policy image order: "
                f"{saved_camera_names} != {self.camera_names}."
            )
        saved_entity_names = tuple(features[TARGET_BOXES].get("entity_names", ()))
        if saved_entity_names != self.entity_names:
            raise ValueError(
                "Target-box entity vocabulary does not match the sidecar metadata: "
                f"{saved_entity_names} != {self.entity_names}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible metadata dictionary."""

        return {
            "camera_names": list(self.camera_names),
            "entity_names": list(self.entity_names),
            "max_entities": self.max_entities,
            "schema_version": self.schema_version,
            "coordinate_bins": self.coordinate_bins,
            "coordinate_space": self.coordinate_space,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionFlowV22DatasetMetadata:
        """Construct metadata from a JSON-compatible dictionary."""

        values = dict(data)
        values["camera_names"] = tuple(values["camera_names"])
        values["entity_names"] = tuple(values["entity_names"])
        return cls(**values)

    def save(self, path: str | Path) -> None:
        """Save the sidecar metadata to JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> ActionFlowV22DatasetMetadata:
        """Load and validate sidecar metadata from JSON."""

        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass(frozen=True)
class BatchFieldSpec:
    """One tensor entry in the model-facing batch contract."""

    key: str
    dtype: str
    shape: tuple[str | int, ...]
    semantics: str


def make_model_batch_contract(
    *,
    camera_count: int,
    max_entities: int,
    max_grounding_tokens: int,
    max_fast_action_tokens: int,
) -> tuple[BatchFieldSpec, ...]:
    """Return the model-batch tensor contract using symbolic batch dimensions."""

    if min(camera_count, max_entities, max_grounding_tokens, max_fast_action_tokens) <= 0:
        raise ValueError("All model-batch contract dimensions must be positive.")

    return (
        BatchFieldSpec(
            TARGET_BOXES,
            "float32",
            ("batch_size", camera_count, max_entities, 4),
            "Normalized policy-image xyxy boxes; zeros for not-visible or padded entries.",
        ),
        BatchFieldSpec(
            TARGET_BOX_BINS,
            "int64",
            ("batch_size", camera_count, max_entities, 4),
            "Quantized xyxy bins aligned with target boxes; zeros when coordinates are unused.",
        ),
        BatchFieldSpec(
            TARGET_BOX_VISIBILITY,
            "uint8",
            ("batch_size", camera_count, max_entities),
            "TargetVisibility values; meaningful only where entity and annotation are valid.",
        ),
        BatchFieldSpec(
            TARGET_BOX_LOSS_WEIGHT,
            "float32",
            ("batch_size", camera_count, max_entities),
            "Zero excludes an uncertain annotation; positive values supervise box tokens.",
        ),
        BatchFieldSpec(
            TARGET_ENTITY_IDS,
            "int64",
            ("batch_size", max_entities),
            "Stable dataset entity IDs; -1 in padded slots.",
        ),
        BatchFieldSpec(
            TARGET_ENTITY_ROLES,
            "int64",
            ("batch_size", max_entities),
            "TargetRole IDs aligned with entity slots.",
        ),
        BatchFieldSpec(
            TARGET_ENTITY_VALID,
            "uint8",
            ("batch_size", max_entities),
            "One for prompt-relevant entity slots and zero for padding.",
        ),
        BatchFieldSpec(
            GROUNDING_TOKENS,
            "int64",
            ("batch_size", max_grounding_tokens),
            "Teacher-forced structural, entity-name, visibility, and location token IDs.",
        ),
        BatchFieldSpec(
            GROUNDING_TOKEN_MASK,
            "bool",
            ("batch_size", max_grounding_tokens),
            "True for non-padding grounding tokens.",
        ),
        BatchFieldSpec(
            GROUNDING_TOKEN_LOSS_WEIGHT,
            "float32",
            ("batch_size", max_grounding_tokens),
            "Per-token box CE weights propagated from annotation validity.",
        ),
        BatchFieldSpec(
            ACTION_TOKENS,
            "int64",
            ("batch_size", max_fast_action_tokens),
            "Dedicated SmolVLM IDs corresponding one-to-one with FAST symbols.",
        ),
        BatchFieldSpec(
            ACTION_TOKEN_MASK,
            "bool",
            ("batch_size", max_fast_action_tokens),
            "True for non-padding FAST action tokens.",
        ),
    )
