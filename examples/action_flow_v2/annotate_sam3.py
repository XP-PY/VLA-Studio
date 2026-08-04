#!/usr/bin/env python3
"""Create task-relevance masks with SAM3.1 and reviewed YOLOE points.

The script processes one episode-camera-entity at a time to bound memory. For
each visible entity it first runs a semantic text prompt, selects the detected
instance containing the reviewed YOLOE point, and propagates that instance
through the video. Optional point refinement is available, but disabled by
default because it can suppress semantic re-detection in SAM3.1. If text
grounding does not cover the reviewed point, the script falls back to
point-only segmentation and logs that decision in the diagnostics file.

The saved per-object Parquet files use the same schema as ``annotate_sam2.py``.
Set up the isolated environment and official SAM3 repository with:

    conda create -n sam3 python=3.12 -y
    git clone https://github.com/facebookresearch/sam3.git repos/sam3
    conda run -n sam3 python -m pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128
    conda run -n sam3 python -m pip install -e repos/sam3
    conda run -n sam3 python -m pip install "numpy<2" pandas pyarrow av \
        opencv-python-headless

The three known SAM2 failure cases fit a 24 GB GPU with these two runs:

    conda run -n sam3 python examples/action_flow_v2/annotate_sam3.py \
        --pairs 7:top --frame-stride 4 --overwrite
    conda run -n sam3 python examples/action_flow_v2/annotate_sam3.py \
        --pairs 52:wrist 60:wrist --frame-stride 16 \
        --overwrite \
        2>&1 | tee logs/annotate_sam3_known_failures.log
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sam3 import build_sam3_predictor

DEFAULT_DATASET_ROOT = Path("runs/dataset/rm65_single_table_preparation_and_bussing")
DEFAULT_YOLOE_ROOT = Path("runs/yoloe/rm65_single_table_preparation_and_bussing")
DEFAULT_YOLOE_REVIEW = DEFAULT_YOLOE_ROOT / "review" / "decisions.json"
DEFAULT_SAM_ROOT = Path("runs/sam/rm65_single_table_preparation_and_bussing_yoloe_sam3_1")
DEFAULT_OUTPUT_ROOT = Path("runs/dataset/rm65_single_table_preparation_and_bussing_afv2_yoloe_sam3_1")
DEFAULT_CHECKPOINT = Path(
    ".cache/huggingface/hub/models--facebook--sam3.1/snapshots/"
    "daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt"
)
DEFAULT_PAIRS = ("7:top", "52:wrist", "60:wrist")
CAMERAS = ("top", "wrist")
COLORS = (
    np.array((255, 64, 80), dtype=np.float32),
    np.array((54, 211, 153), dtype=np.float32),
    np.array((59, 130, 246), dtype=np.float32),
    np.array((245, 158, 11), dtype=np.float32),
    np.array((168, 85, 247), dtype=np.float32),
)

logger = logging.getLogger(__name__)


def encode_mask_runs(mask: np.ndarray) -> bytes:
    """Encode a binary mask as compressed uint32 ``(start, length)`` pairs."""
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    padded = np.concatenate((np.zeros(1, dtype=np.uint8), flat, np.zeros(1, dtype=np.uint8)))
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    runs = transitions.reshape(-1, 2).astype(np.uint32)
    runs[:, 1] -= runs[:, 0]
    return zlib.compress(runs.tobytes())


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def write_parquet_atomic(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


class EpisodeVideoSource:
    """Read an episode camera segment directly from LeRobot v3 metadata."""

    def __init__(self, dataset_root: Path):
        self.root = dataset_root.resolve()
        with (self.root / "meta" / "info.json").open() as file:
            self.info = json.load(file)
        episode_files = sorted((self.root / "meta" / "episodes").glob("chunk-*/*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"No episode metadata found under {self.root}")
        episodes = pd.concat((pd.read_parquet(path) for path in episode_files), ignore_index=True)
        self.episodes = episodes.set_index("episode_index", drop=False)
        self.fps = int(self.info["fps"])

    @property
    def episode_indices(self) -> set[int]:
        return {int(index) for index in self.episodes.index}

    def prompt(self, episode_index: int) -> str:
        tasks = self.episodes.loc[episode_index, "tasks"]
        if isinstance(tasks, str):
            return tasks
        if len(tasks) != 1:
            raise ValueError(f"Episode {episode_index} has {len(tasks)} tasks; expected one")
        return str(tasks[0])

    def frames(self, episode_index: int, camera: str) -> list[np.ndarray]:
        row = self.episodes.loc[episode_index]
        prefix = f"videos/observation.images.{camera}"
        chunk_index = int(row[f"{prefix}/chunk_index"])
        file_index = int(row[f"{prefix}/file_index"])
        start_s = float(row[f"{prefix}/from_timestamp"])
        expected_frames = int(row["length"])
        video_key = f"observation.images.{camera}"
        relative_path = self.info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        video_path = self.root / relative_path
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        decoded: list[np.ndarray] = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if start_s > 0:
                seek_pts = max(0, int((start_s - 1.0 / self.fps) / float(stream.time_base)))
                container.seek(seek_pts, stream=stream, backward=True)
            tolerance_s = 0.5 / self.fps
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                timestamp_s = float(frame.pts * stream.time_base)
                if timestamp_s < start_s - tolerance_s:
                    continue
                decoded.append(frame.to_ndarray(format="rgb24"))
                if len(decoded) == expected_frames:
                    break
        if len(decoded) != expected_frames:
            raise RuntimeError(
                f"Decoded {len(decoded)} frames for episode {episode_index} camera {camera}; "
                f"metadata expects {expected_frames}"
            )
        return decoded


@dataclass(frozen=True)
class EntityPrompt:
    object_id: int
    object_name: str
    object_role: str
    text_query: str
    frame_index: int | None
    point_xy: tuple[int, int] | None
    source: str
    not_visible: bool = False


class YOLOEReview:
    """Resolve selected YOLOE points and saved browser-review overrides."""

    VALID_STATUSES: ClassVar[set[str]] = {"candidate_point", "manual_point", "not_visible"}

    def __init__(self, yoloe_root: Path, review_path: Path):
        self.root = yoloe_root.resolve()
        self.review_path = review_path.resolve()
        payload = json.loads(self.review_path.read_text())
        if int(payload.get("format_version", 0)) != 1:
            raise ValueError(f"Unsupported review format in {self.review_path}")
        self.review_sha256 = hashlib.sha256(self.review_path.read_bytes()).hexdigest()
        self.decisions: dict[tuple[int, str, int], dict[str, Any]] = {}
        for item in payload.get("decisions", []):
            status = str(item["status"])
            if status not in self.VALID_STATUSES:
                raise ValueError(f"Invalid YOLOE review status {status!r}")
            key = (int(item["episode_index"]), str(item["camera"]), int(item["object_id"]))
            self.decisions[key] = item

    def _paths(self, episode_index: int, camera: str) -> tuple[Path, Path]:
        episode_dir = f"episode-{episode_index:06d}"
        return (
            self.root / "data" / episode_dir / f"camera-{camera}.parquet",
            self.root / "candidates" / episode_dir / f"camera-{camera}.parquet",
        )

    @staticmethod
    def _valid_point(row: pd.Series) -> tuple[int, int, int] | None:
        if not bool(row["visible"]) or int(row["frame_index"]) < 0:
            return None
        values = (float(row["point_x"]), float(row["point_y"]))
        if not np.isfinite(values).all():
            return None
        x, y = (round(value) for value in values)
        width, height = int(row["image_width"]), int(row["image_height"])
        if not (0 <= x < width and 0 <= y < height):
            return None
        return int(row["frame_index"]), x, y

    def resolve(self, episode_index: int, camera: str) -> list[EntityPrompt]:
        selected_path, candidates_path = self._paths(episode_index, camera)
        if not selected_path.is_file() or not candidates_path.is_file():
            raise FileNotFoundError(f"YOLOE artifacts missing for episode {episode_index} {camera}")
        selected = pd.read_parquet(selected_path).sort_values("object_id")
        candidates = pd.read_parquet(candidates_path).sort_values(
            ["object_id", "candidate_rank"]
        )
        entities = []
        for _, selected_row in selected.iterrows():
            object_id = int(selected_row["object_id"])
            name = str(selected_row["object_name"])
            role = str(selected_row["object_role"])
            query = str(selected_row.get("query", name)).strip() or name
            decision = self.decisions.get((episode_index, camera, object_id))
            if decision is not None and decision["status"] == "not_visible":
                entities.append(
                    EntityPrompt(object_id, name, role, query, None, None, "reviewed not_visible", True)
                )
                continue

            source = "selected YOLOE point"
            point_row = selected_row
            manual_point = None
            if decision is not None and decision["status"] == "manual_point":
                manual_point = (
                    int(decision["frame_index"]),
                    int(decision["point_x"]),
                    int(decision["point_y"]),
                )
                source = "reviewed manual point"
            elif decision is not None and decision["status"] == "candidate_point":
                rank = int(decision["candidate_rank"])
                matches = candidates.loc[
                    (candidates["object_id"] == object_id)
                    & (candidates["candidate_rank"] == rank)
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Episode {episode_index} {camera} {name}: candidate {rank} unavailable"
                    )
                point_row = matches.iloc[0]
                source = f"reviewed candidate {rank} point"

            point = manual_point or self._valid_point(point_row)
            if point is None:
                raise RuntimeError(
                    f"Episode {episode_index} {camera} {name}: no usable point; "
                    "review it or mark it not_visible"
                )
            frame_index, x, y = point
            entities.append(
                EntityPrompt(object_id, name, role, query, frame_index, (x, y), source)
            )
        return entities


def normalize_masks(outputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    object_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
    raw_masks = outputs.get("out_binary_masks")
    if raw_masks is None:
        return object_ids, np.empty((0, 0, 0), dtype=bool)
    masks = np.asarray(raw_masks, dtype=bool)
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise ValueError(f"Unexpected SAM3 mask shape {masks.shape}")
    if masks.shape[0] != len(object_ids):
        raise ValueError(f"SAM3 returned {len(object_ids)} IDs and mask shape {masks.shape}")
    return object_ids, masks


def select_instance_at_point(outputs: dict[str, Any], point_xy: tuple[int, int]) -> int | None:
    """Select the text-detected instance whose mask contains the reviewed point."""
    object_ids, masks = normalize_masks(outputs)
    if not len(object_ids):
        return None
    x, y = point_xy
    if not (0 <= y < masks.shape[1] and 0 <= x < masks.shape[2]):
        return None
    matches = np.flatnonzero(masks[:, y, x])
    if not len(matches):
        return None
    scores = outputs.get("out_probs", outputs.get("out_scores"))
    if scores is None or len(matches) == 1:
        return int(object_ids[int(matches[0])])
    scores_array = np.asarray(scores).reshape(-1)
    best = matches[int(np.argmax(scores_array[matches]))]
    return int(object_ids[int(best)])


def semantic_query_candidates(entity: EntityPrompt) -> list[str]:
    """Return increasingly broad SAM3 text prompts while preserving entity semantics."""
    candidates = [entity.text_query, entity.object_name]
    name = entity.object_name.lower()
    if "plate" in name:
        color = name.removesuffix(" plate").strip()
        candidates.extend((f"{color} bowl", "bowl", "plate"))
    elif name == "collection box":
        candidates.extend(("cardboard box", "box"))
    elif name == "garlic":
        candidates.append("garlic bulb")
    return list(dict.fromkeys(candidate.strip() for candidate in candidates if candidate.strip()))


def mask_for_object(outputs: dict[str, Any], object_id: int, shape: tuple[int, int]) -> np.ndarray:
    object_ids, masks = normalize_masks(outputs)
    matches = np.flatnonzero(object_ids == object_id)
    if not len(matches):
        return np.zeros(shape, dtype=bool)
    return masks[int(matches[0])]


def select_tracking_mask(
    outputs: dict[str, Any],
    preferred_object_id: int,
    previous_mask: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, int | None, list[int]]:
    """Follow an instance if SAM3 assigns a new ID during semantic re-detection."""
    object_ids, masks = normalize_masks(outputs)
    seen_ids = [int(value) for value in object_ids]
    nonempty = np.flatnonzero(masks.reshape(len(masks), -1).any(axis=1)) if len(masks) else []
    if not len(nonempty):
        return np.zeros(shape, dtype=bool), None, seen_ids

    preferred = np.flatnonzero(object_ids == preferred_object_id)
    if len(preferred) and np.any(masks[int(preferred[0])]):
        index = int(preferred[0])
        return masks[index], int(object_ids[index]), seen_ids

    if np.any(previous_mask):
        intersections = np.logical_and(masks[nonempty], previous_mask).reshape(len(nonempty), -1).sum(axis=1)
        unions = np.logical_or(masks[nonempty], previous_mask).reshape(len(nonempty), -1).sum(axis=1)
        ious = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions > 0)
        if float(ious.max()) > 0:
            index = int(nonempty[int(np.argmax(ious))])
            return masks[index], int(object_ids[index]), seen_ids

    areas = masks[nonempty].reshape(len(nonempty), -1).sum(axis=1)
    index = int(nonempty[int(np.argmax(areas))])
    return masks[index], int(object_ids[index]), seen_ids


def nearest_fill(
    sampled_masks: dict[int, np.ndarray],
    session_to_original: list[int],
    frame_count: int,
    shape: tuple[int, int],
) -> list[np.ndarray]:
    available = sorted(
        session_to_original[session_index]
        for session_index in sampled_masks
        if 0 <= session_index < len(session_to_original)
    )
    masks_by_original = {
        session_to_original[session_index]: mask
        for session_index, mask in sampled_masks.items()
        if 0 <= session_index < len(session_to_original)
    }
    if not available:
        return [np.zeros(shape, dtype=bool) for _ in range(frame_count)]
    available_array = np.asarray(available)
    tracks = []
    for frame_index in range(frame_count):
        position = int(np.searchsorted(available_array, frame_index))
        candidates = []
        if position < len(available):
            candidates.append(available[position])
        if position > 0:
            candidates.append(available[position - 1])
        nearest = min(candidates, key=lambda value: abs(value - frame_index))
        tracks.append(masks_by_original[nearest])
    return tracks


def prepare_frame_directory(
    frames: list[np.ndarray],
    frame_stride: int,
    prompt_frames: set[int],
    directory: Path,
) -> list[int]:
    indices = set(range(0, len(frames), frame_stride))
    indices.add(len(frames) - 1)
    indices.update(prompt_frames)
    ordered = sorted(index for index in indices if 0 <= index < len(frames))
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for session_index, original_index in enumerate(ordered):
        Image.fromarray(frames[original_index]).save(
            directory / f"{session_index:06d}.jpg", quality=95, subsampling=0
        )
    return ordered


def track_entity(
    predictor: Any,
    frame_directory: Path,
    session_to_original: list[int],
    entity: EntityPrompt,
    frame_shape: tuple[int, int],
    output_prob_threshold: float,
    refine_with_point: bool,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    assert entity.frame_index is not None and entity.point_xy is not None
    session_frame = session_to_original.index(entity.frame_index)
    x, y = entity.point_xy
    height, width = frame_shape
    mode: str | None = None
    detected_ids: list[int] = []
    selected_id: int | None = None
    selected_text_query: str | None = None
    text_queries_tried: list[str] = []
    sampled_masks: dict[int, np.ndarray] = {}
    propagation_ids: set[int] = set()
    propagation_warnings: list[str] = []
    nonempty_propagation_frames = 0

    def start_prompted_session() -> tuple[str, int, np.ndarray, dict[str, Any]]:
        response = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(frame_directory),
                "offload_video_to_cpu": True,
            }
        )
        session_id = response["session_id"]
        local_mode = "text+point"
        local_detected_ids: list[int] = []
        local_selected_id: int | None = None
        local_selected_text_query: str | None = None
        local_queries_tried: list[str] = []
        text_outputs: dict[str, Any] = {}
        for text_query in semantic_query_candidates(entity):
            local_queries_tried.append(text_query)
            text_response = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": session_frame,
                    "text": text_query,
                    "output_prob_thresh": output_prob_threshold,
                }
            )
            text_outputs = text_response["outputs"]
            local_detected_ids = [int(value) for value in normalize_masks(text_outputs)[0]]
            local_selected_id = select_instance_at_point(text_outputs, entity.point_xy)
            if local_selected_id is not None:
                local_selected_text_query = text_query
                break

        if local_selected_id is None:
            local_mode = "point-only fallback"
            predictor.handle_request({"type": "reset_session", "session_id": session_id})
            local_selected_id = 0
        else:
            for other_id in local_detected_ids:
                if other_id != local_selected_id:
                    predictor.handle_request(
                        {
                            "type": "remove_object",
                            "session_id": session_id,
                            "frame_index": session_frame,
                            "obj_id": other_id,
                        }
                    )

        if local_mode == "text+point" and not refine_with_point:
            local_mode = "text-selected by point"
            prompt_mask = mask_for_object(text_outputs, local_selected_id, frame_shape)
        else:
            point_response = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": session_frame,
                    "points": [[x / width, y / height]],
                    "point_labels": [1],
                    "obj_id": local_selected_id,
                    "rel_coordinates": True,
                    "output_prob_thresh": output_prob_threshold,
                }
            )
            prompt_mask = mask_for_object(
                point_response["outputs"], local_selected_id, frame_shape
            )
        session_info = {
            "mode": local_mode,
            "detected_ids": local_detected_ids,
            "selected_text_query": local_selected_text_query,
            "text_queries_tried": local_queries_tried,
        }
        return session_id, local_selected_id, prompt_mask, session_info

    # Independent sessions prevent the first direction's feature cache from
    # doubling memory usage while propagating in the opposite direction.
    propagation_distances = {
        "forward": len(session_to_original) - 2 - session_frame,
        "backward": session_frame - 1,
    }
    directions = [(direction, distance) for direction, distance in propagation_distances.items() if distance > 0]
    if not directions:
        directions = [("forward", 0)]

    for direction, max_frames in directions:
        session_id = None
        try:
            session_id, local_selected_id, prompt_mask, session_info = start_prompted_session()
            if np.any(prompt_mask) or session_frame not in sampled_masks:
                sampled_masks[session_frame] = prompt_mask
            if mode is None:
                mode = session_info["mode"]
                detected_ids = session_info["detected_ids"]
                selected_id = local_selected_id
                selected_text_query = session_info["selected_text_query"]
                text_queries_tried = session_info["text_queries_tried"]
            if max_frames <= 0:
                continue

            previous_mask = prompt_mask
            try:
                for propagation in predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": direction,
                        "start_frame_index": session_frame,
                        "max_frame_num_to_track": max_frames,
                        "output_prob_thresh": output_prob_threshold,
                    }
                ):
                    output_frame = int(propagation["frame_index"])
                    mask, _, seen_ids = select_tracking_mask(
                        propagation["outputs"], local_selected_id, previous_mask, frame_shape
                    )
                    propagation_ids.update(seen_ids)
                    if np.any(mask) or output_frame not in sampled_masks:
                        sampled_masks[output_frame] = mask
                    if np.any(mask):
                        nonempty_propagation_frames += 1
                        previous_mask = mask
            except (IndexError, RuntimeError) as error:
                if isinstance(error, torch.OutOfMemoryError):
                    raise
                message = str(error)
                expected_empty_track = (
                    "list index out of range" in message
                    or "existing size (0)" in message
                    or "out of bounds" in message
                )
                if not expected_empty_track:
                    raise
                warning = f"{direction}: {type(error).__name__}: {message}"
                propagation_warnings.append(warning)
                logger.warning(
                    "SAM3 ended %s propagation after %d sampled outputs: %s",
                    direction,
                    len(sampled_masks),
                    message,
                )
        finally:
            if session_id is not None:
                predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": session_id,
                        "run_gc_collect": True,
                        "clear_cache_threshold": 0,
                    }
                )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    tracks = nearest_fill(
        sampled_masks,
        session_to_original,
        frame_count=max(session_to_original) + 1,
        shape=frame_shape,
    )
    diagnostics = {
        "object_id": entity.object_id,
        "object_name": entity.object_name,
        "object_role": entity.object_role,
        "text_query": entity.text_query,
        "text_queries_tried": text_queries_tried,
        "selected_text_query": selected_text_query,
        "prompt_frame": entity.frame_index,
        "point_xy": list(entity.point_xy),
        "point_source": entity.source,
        "mode": mode,
        "text_detected_object_ids": detected_ids,
        "selected_sam3_object_id": selected_id,
        "propagation_object_ids": sorted(propagation_ids),
        "sampled_outputs": len(sampled_masks),
        "nonempty_propagation_outputs": nonempty_propagation_frames,
        "propagation_warnings": propagation_warnings,
    }
    return tracks, diagnostics


def overlay_tracks(frame: np.ndarray, tracks: dict[int, list[np.ndarray]], frame_index: int) -> np.ndarray:
    result = frame.astype(np.float32).copy()
    for color_index, (object_id, masks) in enumerate(sorted(tracks.items())):
        mask = masks[frame_index]
        if not np.any(mask):
            continue
        color = COLORS[color_index % len(COLORS)]
        result[mask] = 0.55 * result[mask] + 0.45 * color
    return np.clip(result, 0, 255).astype(np.uint8)


def write_preview_video(
    path: Path,
    frames: list[np.ndarray],
    tracks: dict[int, list[np.ndarray]],
    entities: list[EntityPrompt],
    fps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    labels = {entity.object_id: entity.object_name for entity in entities}
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "veryfast"}
        for frame_index, frame in enumerate(frames):
            image = Image.fromarray(overlay_tracks(frame, tracks, frame_index))
            draw = ImageDraw.Draw(image)
            y = 6
            for color_index, object_id in enumerate(sorted(tracks)):
                color = tuple(int(value) for value in COLORS[color_index % len(COLORS)])
                draw.rectangle((6, y + 3, 16, y + 13), fill=color)
                draw.text((21, y), f"ID{object_id}: {labels[object_id]}", fill=(255, 255, 255))
                y += 16
            video_frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def save_pair_outputs(
    sam_root: Path,
    output_root: Path,
    episode_index: int,
    camera: str,
    tracks: dict[int, list[np.ndarray]],
    entities: list[EntityPrompt],
) -> None:
    labels = {entity.object_id: entity.object_name for entity in entities}
    frame_count = len(next(iter(tracks.values())))
    height, width = next(iter(tracks.values()))[0].shape
    object_rows = []
    for object_id, masks in sorted(tracks.items()):
        for frame_index, mask in enumerate(masks):
            object_rows.append(
                {
                    "episode_index": np.int64(episode_index),
                    "frame_index": np.int64(frame_index),
                    "camera": camera,
                    "object_id": np.int64(object_id),
                    "object_label": labels[object_id],
                    "height": np.int32(height),
                    "width": np.int32(width),
                    "mask_runs": encode_mask_runs(mask),
                }
            )
    object_path = sam_root / "data" / f"episode-{episode_index:06d}" / f"camera-{camera}.parquet"
    write_parquet_atomic(object_path, pd.DataFrame(object_rows))

    merged_masks = []
    for frame_index in range(frame_count):
        merged = np.zeros((height, width), dtype=bool)
        for masks in tracks.values():
            merged |= masks[frame_index]
        merged_masks.append(merged)
    merged_table = pd.DataFrame(
        {
            "episode_index": np.full(frame_count, episode_index, dtype=np.int64),
            "frame_index": np.arange(frame_count, dtype=np.int64),
            "camera": np.full(frame_count, camera, dtype=object),
            "height": np.full(frame_count, height, dtype=np.int32),
            "width": np.full(frame_count, width, dtype=np.int32),
            "mask_runs": [encode_mask_runs(mask) for mask in merged_masks],
            "loss_weight": np.ones(frame_count, dtype=np.float32),
        }
    )
    merged_path = output_root / "data" / f"episode-{episode_index:06d}" / f"camera-{camera}.parquet"
    write_parquet_atomic(merged_path, merged_table)


def parse_pairs(values: list[str], available_episodes: set[int]) -> list[tuple[int, str]]:
    pairs = []
    for value in values:
        try:
            episode_text, camera = value.split(":", 1)
            episode_index = int(episode_text)
        except ValueError as error:
            raise ValueError(f"Invalid pair {value!r}; expected EPISODE:CAMERA") from error
        if episode_index not in available_episodes:
            raise ValueError(f"Dataset does not contain episode {episode_index}")
        if camera not in CAMERAS:
            raise ValueError(f"Unsupported camera {camera!r}; expected one of {CAMERAS}")
        pairs.append((episode_index, camera))
    return pairs


def outputs_complete(
    source: EpisodeVideoSource,
    sam_root: Path,
    output_root: Path,
    episode_index: int,
    camera: str,
) -> bool:
    expected = int(source.episodes.loc[episode_index, "length"])
    object_path = sam_root / "data" / f"episode-{episode_index:06d}" / f"camera-{camera}.parquet"
    merged_path = output_root / "data" / f"episode-{episode_index:06d}" / f"camera-{camera}.parquet"
    if not object_path.is_file() or not merged_path.is_file():
        return False
    objects = pd.read_parquet(object_path, columns=["object_id", "frame_index"])
    merged = pd.read_parquet(merged_path, columns=["frame_index"])
    return len(merged) == expected and bool((objects.groupby("object_id").size() == expected).all())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--yoloe-root", type=Path, default=DEFAULT_YOLOE_ROOT)
    parser.add_argument("--yoloe-review", type=Path, default=DEFAULT_YOLOE_REVIEW)
    parser.add_argument("--sam-root", type=Path, default=DEFAULT_SAM_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    parser.add_argument("--frame-stride", type=int, default=16)
    parser.add_argument("--output-prob-threshold", type=float, default=0.5)
    parser.add_argument(
        "--point-refinement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refine a text-selected SAM3 instance with the reviewed point (default: false).",
    )
    parser.add_argument(
        "--object-ids",
        nargs="+",
        type=int,
        help="Only process these dataset object IDs; useful for diagnostics.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    source = EpisodeVideoSource(args.dataset_root)
    review = YOLOEReview(args.yoloe_root, args.yoloe_review)
    pairs = parse_pairs(args.pairs, source.episode_indices)
    sam_root = args.sam_root.resolve()
    output_root = args.output_root.resolve()
    sam_info_path = sam_root / "meta" / "info.json"
    previous_metadata = json.loads(sam_info_path.read_text()) if sam_info_path.is_file() else {}
    pair_frame_strides = dict(previous_metadata.get("pair_frame_strides", {}))
    metadata = {
        "format_version": 1,
        "model": "facebook/sam3.1",
        "checkpoint": str(checkpoint),
        "base_dataset_root": str(source.root),
        "yoloe_root": str(review.root),
        "yoloe_review": str(review.review_path),
        "yoloe_review_sha256": review.review_sha256,
        "output_prob_threshold": args.output_prob_threshold,
        "prompt_pipeline": "text detection -> reviewed point instance selection -> semantic propagation",
        "point_refinement": args.point_refinement,
        "fallback": "point-only when no text mask contains the reviewed point",
        "mask_encoding": "zlib-compressed uint32 start-length pairs in row-major order",
    }

    def write_metadata() -> None:
        unique_strides = sorted(set(pair_frame_strides.values()))
        stride = unique_strides[0] if len(unique_strides) == 1 else None
        stride_metadata = {
            "frame_stride": stride,
            "pair_frame_strides": pair_frame_strides,
        }
        write_json_atomic(
            sam_info_path,
            {**metadata, **stride_metadata, "output_root": str(output_root)},
        )
        write_json_atomic(
            output_root / "meta" / "info.json",
            {**metadata, **stride_metadata, "sam_root": str(sam_root)},
        )

    write_metadata()

    report_path = sam_root / "meta" / "sam3_batch_report.json"
    selected_keys = {(episode_index, camera) for episode_index, camera in pairs}
    report: dict[str, Any] = {"pairs": [], "failed": []}
    previous_pairs: dict[tuple[int, str], dict[str, Any]] = {}
    if report_path.is_file():
        previous_report = json.loads(report_path.read_text())
        previous_pairs = {
            (int(item["episode_index"]), str(item["camera"])): item
            for item in previous_report.get("pairs", [])
        }
        for section in ("pairs", "failed"):
            report[section] = [
                item
                for item in previous_report.get(section, [])
                if (int(item["episode_index"]), str(item["camera"])) not in selected_keys
            ]

    predictor = None
    for pair_number, (episode_index, camera) in enumerate(pairs, start=1):
        pair = {"episode_index": episode_index, "camera": camera}
        if not args.overwrite and outputs_complete(
            source, sam_root, output_root, episode_index, camera
        ):
            logger.info("[%d/%d] Skipping complete %s", pair_number, len(pairs), pair)
            existing = previous_pairs.get((episode_index, camera), {})
            report["pairs"].append({**existing, **pair, "status": "skipped_existing"})
            continue
        logger.info("[%d/%d] Processing episode %d camera %s", pair_number, len(pairs), episode_index, camera)
        frame_directory = sam_root / "frames" / f"episode-{episode_index:06d}-camera-{camera}"
        try:
            frames = source.frames(episode_index, camera)
            entities = review.resolve(episode_index, camera)
            if args.object_ids:
                requested_ids = set(args.object_ids)
                entities = [entity for entity in entities if entity.object_id in requested_ids]
                missing_ids = requested_ids - {entity.object_id for entity in entities}
                if missing_ids:
                    raise ValueError(
                        f"Episode {episode_index} {camera} does not contain object IDs {sorted(missing_ids)}"
                    )
            prompt_frames = {
                int(entity.frame_index)
                for entity in entities
                if entity.frame_index is not None and not entity.not_visible
            }
            session_to_original = prepare_frame_directory(
                frames, args.frame_stride, prompt_frames, frame_directory
            )
            frame_shape = frames[0].shape[:2]
            tracks: dict[int, list[np.ndarray]] = {}
            diagnostics = []
            for entity_number, entity in enumerate(entities, start=1):
                logger.info(
                    "Episode %d %s entity %d/%d: %s",
                    episode_index,
                    camera,
                    entity_number,
                    len(entities),
                    entity.object_name,
                )
                if entity.not_visible:
                    tracks[entity.object_id] = [
                        np.zeros(frame_shape, dtype=bool) for _ in frames
                    ]
                    diagnostics.append(
                        {
                            "object_id": entity.object_id,
                            "object_name": entity.object_name,
                            "mode": "not_visible",
                            "point_source": entity.source,
                        }
                    )
                    continue
                if predictor is None:
                    logger.info("Loading SAM3.1 from %s", checkpoint)
                    predictor = build_sam3_predictor(
                        checkpoint_path=str(checkpoint),
                        version="sam3.1",
                        compile=False,
                        max_num_objects=16,
                        multiplex_count=16,
                        use_fa3=False,
                        use_rope_real=True,
                        async_loading_frames=False,
                    )
                object_tracks, entity_diagnostics = track_entity(
                    predictor,
                    frame_directory,
                    session_to_original,
                    entity,
                    frame_shape,
                    args.output_prob_threshold,
                    args.point_refinement,
                )
                if len(object_tracks) < len(frames):
                    object_tracks.extend([object_tracks[-1]] * (len(frames) - len(object_tracks)))
                tracks[entity.object_id] = object_tracks[: len(frames)]
                diagnostics.append(entity_diagnostics)

            save_pair_outputs(
                sam_root, output_root, episode_index, camera, tracks, entities
            )
            preview_path = sam_root / "previews" / f"episode-{episode_index:06d}-camera-{camera}.mp4"
            write_preview_video(preview_path, frames, tracks, entities, source.fps)
            diagnostic_path = (
                sam_root / "diagnostics" / f"episode-{episode_index:06d}-camera-{camera}.json"
            )
            write_json_atomic(
                diagnostic_path,
                {
                    **pair,
                    "task": source.prompt(episode_index),
                    "frame_count": len(frames),
                    "frame_stride": args.frame_stride,
                    "sampled_frame_count": len(session_to_original),
                    "entities": diagnostics,
                },
            )
            pair_frame_strides[f"{episode_index}:{camera}"] = args.frame_stride
            report["pairs"].append(
                {
                    **pair,
                    "status": "processed",
                    "frame_stride": args.frame_stride,
                    "entities": diagnostics,
                }
            )
            write_metadata()
        except Exception as error:
            logger.exception("SAM3 failed for episode %d camera %s", episode_index, camera)
            report["failed"].append({**pair, "error": f"{type(error).__name__}: {error}"})
        finally:
            if not args.keep_frames:
                shutil.rmtree(frame_directory, ignore_errors=True)
            write_json_atomic(report_path, report)

    logger.info(
        "SAM3 test complete: %d successful/skipped pairs, %d failed. Report: %s",
        len(report["pairs"]),
        len(report["failed"]),
        report_path,
    )


if __name__ == "__main__":
    main()
