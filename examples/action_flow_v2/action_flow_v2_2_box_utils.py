#!/usr/bin/env python

"""Utilities for building Action Flow V2.2 entity-box labels."""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from lerobot.policies.action_flow_v2_2.data_contracts_action_flow_v2_2 import (
    TargetRole,
    TargetVisibility,
)

ROLE_BY_NAME = {
    "source": TargetRole.SOURCE,
    "destination": TargetRole.DESTINATION,
    "source container": TargetRole.SOURCE_CONTAINER,
    "destination container": TargetRole.DESTINATION_CONTAINER,
}

VISIBLE_REVIEW_STATUSES = {"manual_point", "candidate_point"}
NOT_VISIBLE_REVIEW_STATUS = "not_visible"
ReviewDecisionKey = tuple[int, str, int]


@dataclass(frozen=True)
class EntityDefinition:
    """One prompt-relevant entity shared by all camera views of an episode."""

    object_id: int
    entity_id: int
    name: str
    role: TargetRole
    task_step: int


@dataclass(frozen=True)
class ResizeGeometry:
    """SmolVLA resize-and-pad geometry for one source image shape."""

    source_width: int
    source_height: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.source_height


def compute_resize_geometry(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> ResizeGeometry:
    """Return the geometry used by SmolVLA's ``resize_with_pad``."""

    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Image dimensions must be positive.")
    ratio = max(source_width / target_width, source_height / target_height)
    resized_height = int(source_height / ratio)
    resized_width = int(source_width / ratio)
    return ResizeGeometry(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=max(0, target_width - resized_width),
        pad_top=max(0, target_height - resized_height),
    )


def mask_bounds_from_runs(
    payload: bytes,
    *,
    height: int,
    width: int,
) -> tuple[int, int, int, int, int] | None:
    """Return inclusive ``xyxy`` bounds and mask area from compressed RLE.

    SAM2 stores row-major runs as zlib-compressed uint32 ``(start, length)``
    pairs. This function derives a box without allocating a dense mask.
    """

    raw = zlib.decompress(bytes(payload))
    if not raw:
        return None
    runs = np.frombuffer(raw, dtype=np.uint32)
    if runs.size == 0:
        return None
    if runs.size % 2:
        raise ValueError("RLE payload does not contain start-length pairs.")
    runs = runs.reshape(-1, 2)

    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    area = 0
    flat_size = height * width
    for start_raw, length_raw in runs:
        start = int(start_raw)
        length = int(length_raw)
        if length <= 0:
            continue
        end = start + length
        if start < 0 or end > flat_size:
            raise ValueError(
                f"RLE run [{start}, {end}) is outside a {height}x{width} mask."
            )

        first_y, first_x = divmod(start, width)
        last_y, last_x = divmod(end - 1, width)
        min_y = min(min_y, first_y)
        max_y = max(max_y, last_y)
        if first_y == last_y:
            min_x = min(min_x, first_x)
            max_x = max(max_x, last_x)
        else:
            min_x = 0
            max_x = width - 1
        area += length

    if area == 0:
        return None
    return min_x, min_y, max_x, max_y, area


def transform_bounds(
    bounds: tuple[int, int, int, int],
    geometry: ResizeGeometry,
    *,
    coordinate_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map native inclusive bounds to normalized and quantized policy boxes."""

    if coordinate_bins < 2:
        raise ValueError("`coordinate_bins` must be at least 2.")
    x1, y1, x2, y2 = bounds
    if not (0 <= x1 <= x2 < geometry.source_width):
        raise ValueError(f"Invalid horizontal bounds: {(x1, x2)}.")
    if not (0 <= y1 <= y2 < geometry.source_height):
        raise ValueError(f"Invalid vertical bounds: {(y1, y2)}.")

    policy_x1 = int(np.floor(x1 * geometry.scale_x)) + geometry.pad_left
    policy_y1 = int(np.floor(y1 * geometry.scale_y)) + geometry.pad_top
    policy_x2 = int(np.ceil((x2 + 1) * geometry.scale_x)) - 1 + geometry.pad_left
    policy_y2 = int(np.ceil((y2 + 1) * geometry.scale_y)) - 1 + geometry.pad_top
    policy = np.array([policy_x1, policy_y1, policy_x2, policy_y2], dtype=np.float32)
    maximum = np.array(
        [
            geometry.target_width - 1,
            geometry.target_height - 1,
            geometry.target_width - 1,
            geometry.target_height - 1,
        ],
        dtype=np.float32,
    )
    policy = np.clip(policy, 0, maximum)
    normalized = policy / maximum
    quantized = np.rint(normalized * (coordinate_bins - 1)).astype(np.int64)
    return normalized, quantized


def load_review_visibility_decisions(
    grounding_root: Path,
) -> dict[ReviewDecisionKey, str]:
    """Load visibility decisions recorded by the YOLOE point-review interface."""

    path = grounding_root / "review" / "decisions.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing reviewed YOLOE decisions: {path}")
    payload = json.loads(path.read_text())
    decisions: dict[ReviewDecisionKey, str] = {}
    allowed_statuses = VISIBLE_REVIEW_STATUSES | {NOT_VISIBLE_REVIEW_STATUS}
    for row in payload.get("decisions", []):
        key = (
            int(row["episode_index"]),
            str(row["camera"]),
            int(row["object_id"]),
        )
        status = str(row["status"])
        if status not in allowed_statuses:
            raise ValueError(f"Unsupported YOLOE review status {status!r} for {key}.")
        if key in decisions:
            raise ValueError(f"Duplicate YOLOE review decision for {key}.")
        decisions[key] = status
    return decisions


def load_episode_definitions(
    grounding_root: Path,
    *,
    episode_index: int,
    cameras: tuple[str, ...],
    entity_name_to_id: dict[str, int],
    review_decisions: Mapping[ReviewDecisionKey, str],
) -> tuple[str, list[EntityDefinition], dict[str, set[int]]]:
    """Load entities while applying reviewed visibility over raw YOLOE output."""

    camera_rows: dict[str, list[dict[str, Any]]] = {}
    prompt: str | None = None
    not_visible: dict[str, set[int]] = {}
    for camera in cameras:
        path = (
            grounding_root
            / "json"
            / f"episode-{episode_index:06d}"
            / f"camera-{camera}.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing reviewed YOLOE labels: {path}")
        payload = json.loads(path.read_text())
        selected = payload.get("selected", [])
        if not selected:
            raise ValueError(f"No entity definitions in {path}.")
        camera_rows[camera] = selected
        not_visible[camera] = set()
        for row in selected:
            object_id = int(row["object_id"])
            status = review_decisions.get((episode_index, camera, object_id))
            if status == NOT_VISIBLE_REVIEW_STATUS:
                not_visible[camera].add(object_id)
            elif status not in VISIBLE_REVIEW_STATUSES and not bool(row["visible"]):
                not_visible[camera].add(object_id)
        camera_prompt = str(selected[0]["task"])
        if prompt is None:
            prompt = camera_prompt
        elif camera_prompt != prompt:
            raise ValueError(
                f"Episode {episode_index} prompt differs between camera views."
            )

    first_camera = cameras[0]
    expected = {
        int(row["object_id"]): (
            str(row["object_name"]),
            str(row["object_role"]),
            int(row["task_step"]),
        )
        for row in camera_rows[first_camera]
    }
    for camera in cameras[1:]:
        actual = {
            int(row["object_id"]): (
                str(row["object_name"]),
                str(row["object_role"]),
                int(row["task_step"]),
            )
            for row in camera_rows[camera]
        }
        if actual != expected:
            raise ValueError(
                f"Episode {episode_index} entity identities differ between "
                f"{first_camera} and {camera}."
            )

    role_order = {
        TargetRole.SOURCE: 0,
        TargetRole.SOURCE_CONTAINER: 1,
        TargetRole.DESTINATION: 2,
        TargetRole.DESTINATION_CONTAINER: 3,
    }
    definitions = []
    for object_id, (name, role_name, task_step) in expected.items():
        try:
            role = ROLE_BY_NAME[role_name]
        except KeyError as error:
            raise ValueError(
                f"Unsupported entity role {role_name!r} in episode {episode_index}."
            ) from error
        definitions.append(
            EntityDefinition(
                object_id=object_id,
                entity_id=entity_name_to_id[name],
                name=name,
                role=role,
                task_step=task_step,
            )
        )
    definitions.sort(
        key=lambda entity: (
            entity.task_step,
            role_order[entity.role],
            entity.object_id,
        )
    )
    return prompt or "", definitions, not_visible


def discover_entity_vocabulary(
    grounding_root: Path,
    *,
    episode_indices: list[int],
    cameras: tuple[str, ...],
) -> tuple[str, ...]:
    """Return a deterministic global entity vocabulary."""

    names: set[str] = set()
    for episode_index in episode_indices:
        path = (
            grounding_root
            / "json"
            / f"episode-{episode_index:06d}"
            / f"camera-{cameras[0]}.json"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        selected = json.loads(path.read_text()).get("selected", [])
        names.update(str(row["object_name"]) for row in selected)
    return tuple(sorted(names))


def load_workspace_flags(
    sam_root: Path,
    *,
    episode_index: int,
    camera: str,
) -> tuple[set[int], set[int], set[int]]:
    """Return unresolved objects, explicit absences, and excluded frames."""

    path = sam_root / "work" / f"episode-{episode_index:06d}-camera-{camera}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing SAM2 workspace: {path}")
    payload = json.loads(path.read_text())
    unresolved = {int(value) for value in payload.get("grounding_unresolved", {})}
    not_visible = {int(value) for value in payload.get("grounding_not_visible", {})}
    excluded_frames = {int(value) for value in payload.get("excluded_frames", [])}
    return unresolved, not_visible, excluded_frames


def build_camera_labels(
    sam_root: Path,
    *,
    episode_index: int,
    camera: str,
    definitions: list[EntityDefinition],
    frame_count: int,
    target_width: int,
    target_height: int,
    coordinate_bins: int,
    reviewed_not_visible: set[int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    """Build one camera's per-frame box, visibility, and weight tensors."""

    path = (
        sam_root
        / "data"
        / f"episode-{episode_index:06d}"
        / f"camera-{camera}.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing SAM2 object tracks: {path}")
    table = pd.read_parquet(path)
    required = {
        "episode_index",
        "frame_index",
        "camera",
        "object_id",
        "object_label",
        "height",
        "width",
        "mask_runs",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}.")

    unresolved, workspace_not_visible, excluded_frames = load_workspace_flags(
        sam_root,
        episode_index=episode_index,
        camera=camera,
    )
    explicit_not_visible = reviewed_not_visible | workspace_not_visible
    by_object = {
        int(object_id): rows.sort_values("frame_index").reset_index(drop=True)
        for object_id, rows in table.groupby("object_id", sort=False)
    }

    entity_count = len(definitions)
    boxes = np.zeros((frame_count, entity_count, 4), dtype=np.float32)
    bins = np.zeros((frame_count, entity_count, 4), dtype=np.int64)
    visibility = np.zeros((frame_count, entity_count), dtype=np.uint8)
    weights = np.ones((frame_count, entity_count), dtype=np.float32)
    sidecar_rows: list[dict[str, Any]] = []

    for entity_slot, entity in enumerate(definitions):
        rows = by_object.get(entity.object_id)
        if rows is None:
            if entity.object_id not in explicit_not_visible | unresolved:
                raise ValueError(
                    f"{path} has no SAM2 track for entity {entity.object_id} ({entity.name})."
                )
            rows_by_frame: dict[int, Any] = {}
        else:
            if len(rows) != frame_count:
                raise ValueError(
                    f"{path} has {len(rows)} rows for {entity.name}; expected {frame_count}."
                )
            actual_frames = rows["frame_index"].to_numpy(dtype=np.int64)
            if not np.array_equal(actual_frames, np.arange(frame_count)):
                raise ValueError(f"{path} has incomplete frame indices for {entity.name}.")
            labels = set(rows["object_label"].astype(str))
            if labels != {entity.name}:
                raise ValueError(
                    f"{path} labels {labels} do not match entity name {entity.name!r}."
                )
            rows_by_frame = {
                int(row.frame_index): row for row in rows.itertuples(index=False)
            }

        for frame_index in range(frame_count):
            row = rows_by_frame.get(frame_index)
            loss_weight = 0.0 if (
                frame_index in excluded_frames or entity.object_id in unresolved
            ) else 1.0
            weights[frame_index, entity_slot] = loss_weight
            normalized_box: np.ndarray | None = None
            quantized_box: np.ndarray | None = None
            mask_area = 0
            if row is not None and entity.object_id not in explicit_not_visible:
                bounds_and_area = mask_bounds_from_runs(
                    row.mask_runs,
                    height=int(row.height),
                    width=int(row.width),
                )
                if bounds_and_area is not None:
                    x1, y1, x2, y2, mask_area = bounds_and_area
                    geometry = compute_resize_geometry(
                        int(row.width),
                        int(row.height),
                        target_width,
                        target_height,
                    )
                    normalized_box, quantized_box = transform_bounds(
                        (x1, y1, x2, y2),
                        geometry,
                        coordinate_bins=coordinate_bins,
                    )

            if normalized_box is not None:
                boxes[frame_index, entity_slot] = normalized_box
                bins[frame_index, entity_slot] = quantized_box
                visibility[frame_index, entity_slot] = int(TargetVisibility.VISIBLE)

            sidecar_rows.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "camera": camera,
                    "object_id": entity.object_id,
                    "entity_id": entity.entity_id,
                    "entity_name": entity.name,
                    "entity_role": int(entity.role),
                    "task_step": entity.task_step,
                    "visibility": int(visibility[frame_index, entity_slot]),
                    "x1": float(boxes[frame_index, entity_slot, 0]),
                    "y1": float(boxes[frame_index, entity_slot, 1]),
                    "x2": float(boxes[frame_index, entity_slot, 2]),
                    "y2": float(boxes[frame_index, entity_slot, 3]),
                    "bin_x1": int(bins[frame_index, entity_slot, 0]),
                    "bin_y1": int(bins[frame_index, entity_slot, 1]),
                    "bin_x2": int(bins[frame_index, entity_slot, 2]),
                    "bin_y2": int(bins[frame_index, entity_slot, 3]),
                    "mask_area": mask_area,
                    "loss_weight": loss_weight,
                    "explicit_not_visible": entity.object_id in explicit_not_visible,
                }
            )
    return boxes, bins, visibility, weights, sidecar_rows
