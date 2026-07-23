#!/usr/bin/env python3
"""Initialize Action Flow V2 object tracks with Grounding DINO.

The script derives the task-relevant entities from each episode prompt,
assigns deterministic object IDs, scans sampled video frames, and ranks views
that show each entity clearly and completely. The same object IDs and names
are used for every selected camera in an episode.

This is an inspection stage only. It does not run SAM2 and does not modify the
source LeRobot dataset. Inspect ``inspection/index.html`` before using these
boxes as SAM2 prompts.

Quick test:
    python examples/action_flow_v2/annotate_grounding.py \
        --dataset-config table_preparation \
        --episodes 1 --device cuda --overwrite

Process the bimanual assembly-parts dataset:
    python examples/action_flow_v2/annotate_grounding.py \
        --dataset-config assembly_parts \
        --episodes all --device cuda

Process the complete dataset:
    python examples/action_flow_v2/annotate_grounding.py \
        --episodes all --cameras top wrist --device cuda

Outputs default to
``runs/sam/<dataset-name>/grounding_dino``. Existing completed
episode-camera results are skipped unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

try:
    from .grounding_configs import (
        DATASET_CONFIGS,
        TABLE_PREPARATION_CONFIG,
        EntitySpec,
        GroundingDatasetConfig,
        get_dataset_config,
    )
except ImportError:
    from grounding_configs import (  # type: ignore[no-redef]
        DATASET_CONFIGS,
        TABLE_PREPARATION_CONFIG,
        EntitySpec,
        GroundingDatasetConfig,
        get_dataset_config,
    )


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = WORKSPACE / "models" / "grounding-dino-tiny"
PREVIEW_PANEL_SIZE = (384, 288)
PREVIEW_COLORS = (
    (230, 70, 70),
    (70, 190, 100),
    (70, 130, 235),
    (235, 170, 45),
    (180, 90, 220),
    (40, 190, 200),
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One grounding proposal with complete-view quality metrics."""

    visible: bool
    frame_index: int
    timestamp_s: float | None
    score: float | None
    box_xyxy: tuple[float, float, float, float] | None
    image_width: int
    image_height: int
    area_ratio: float | None
    border_margin: float | None
    sharpness_score: float | None
    quality_score: float | None
    episode_progress: float | None
    selection_score: float | None
    border_truncated: bool | None


class EpisodeVideoSource:
    """Stream sampled frames from episode segments in a LeRobot v3 dataset."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        episode_paths = sorted((self.root / "meta" / "episodes").glob("chunk-*/*.parquet"))
        if not episode_paths:
            raise FileNotFoundError(f"No episode metadata found under {self.root}")
        episodes = pd.concat((pd.read_parquet(path) for path in episode_paths), ignore_index=True)
        self.episodes = episodes.set_index("episode_index", drop=False)
        self.fps = int(self.info["fps"])

    @property
    def episode_indices(self) -> list[int]:
        return sorted(int(index) for index in self.episodes.index)

    @property
    def camera_names(self) -> tuple[str, ...]:
        prefix = "observation.images."
        return tuple(
            feature_name.removeprefix(prefix)
            for feature_name, feature in self.info["features"].items()
            if feature_name.startswith(prefix) and feature.get("dtype") == "video"
        )

    def row(self, episode_index: int) -> pd.Series:
        if episode_index not in self.episodes.index:
            raise ValueError(f"Episode {episode_index} does not exist")
        return self.episodes.loc[episode_index]

    def task(self, episode_index: int) -> str:
        tasks = self.row(episode_index)["tasks"]
        if isinstance(tasks, str):
            return tasks
        if len(tasks) != 1:
            raise ValueError(f"Episode {episode_index} has {len(tasks)} tasks; expected one")
        return str(tasks[0])

    def _video_segment(self, episode_index: int, camera: str) -> tuple[Path, float, int]:
        row = self.row(episode_index)
        video_key = f"observation.images.{camera}"
        prefix = f"videos/{video_key}"
        required = [f"{prefix}/chunk_index", f"{prefix}/file_index", f"{prefix}/from_timestamp"]
        missing = [name for name in required if name not in row.index]
        if missing:
            raise KeyError(f"Camera {camera!r} is unavailable; missing metadata fields: {missing}")

        relative_path = self.info["video_path"].format(
            video_key=video_key,
            chunk_index=int(row[f"{prefix}/chunk_index"]),
            file_index=int(row[f"{prefix}/file_index"]),
        )
        path = self.root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, float(row[f"{prefix}/from_timestamp"]), int(row["length"])

    def sampled_frames(
        self,
        episode_index: int,
        camera: str,
        frame_stride: int,
        max_sampled_frames: int,
    ) -> Iterator[tuple[int, float, Image.Image]]:
        video_path, start_s, episode_length = self._video_segment(episode_index, camera)
        yielded = 0
        decoded_index = 0
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
                if decoded_index >= episode_length:
                    break

                should_sample = decoded_index % frame_stride == 0 or decoded_index == episode_length - 1
                if should_sample:
                    yield decoded_index, timestamp_s, Image.fromarray(frame.to_ndarray(format="rgb24"))
                    yielded += 1
                    if max_sampled_frames > 0 and yielded >= max_sampled_frames:
                        return
                decoded_index += 1


def grounding_query(object_name: str) -> str:
    """Return the legacy table-dataset query used by helper scripts."""
    return TABLE_PREPARATION_CONFIG.grounding_query(object_name)


def entities_from_task(task: str) -> list[EntitySpec]:
    """Parse a legacy table-dataset prompt for helper-script compatibility."""
    return TABLE_PREPARATION_CONFIG.entities_from_task(task)


def parse_episode_selection(value: str, available: list[int]) -> list[int]:
    """Parse ``all``, comma-separated indices, and start:stop ranges."""
    if value.strip().lower() == "all":
        return available
    selected: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = item.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid episode range {item!r}; use start:stop")
            start = int(parts[0]) if parts[0] else min(available)
            stop = int(parts[1]) if parts[1] else max(available) + 1
            selected.update(range(start, stop))
        else:
            selected.add(int(item))
    missing = sorted(selected.difference(available))
    if missing:
        raise ValueError(f"Episodes do not exist: {missing}")
    return sorted(selected)


def detect_entities(
    image: Image.Image,
    entities: list[EntitySpec],
    dataset_config: GroundingDatasetConfig,
    processor,
    model,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
) -> dict[int, tuple[float, tuple[float, float, float, float]]]:
    """Ground entities with one contrastive vocabulary prompt."""
    if not entities:
        return {}
    vocabulary = list(dataset_config.detector_queries)
    for entity in entities:
        if entity.relation_container_query is None and entity.query not in vocabulary:
            vocabulary.append(entity.query)
        relation_queries = (entity.relation_container_query, *entity.relation_member_queries)
        for query in relation_queries:
            if query is not None and query not in vocabulary:
                vocabulary.append(query)
    vocabulary = list(dict.fromkeys(vocabulary))
    prompt = ". ".join(vocabulary) + "."
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]

    scores = result["scores"]
    boxes = result["boxes"]
    text_labels = result["text_labels"]
    if len(scores) == 0 and len(boxes) == 0:
        # Transformers 5.5.4 can return text_labels=[""] for an empty
        # Grounding DINO result. Treat it as the no-detection case.
        return {}
    if len(text_labels) != len(scores) or len(scores) != len(boxes):
        raise RuntimeError(
            "Grounding DINO post-processing returned misaligned outputs: "
            f"text_labels={len(text_labels)}, scores={len(scores)}, boxes={len(boxes)}"
        )

    candidates = [
        (
            " ".join(str(label).lower().split()),
            float(score.item()),
            tuple(float(value) for value in box.detach().cpu().tolist()),
        )
        for label, score, box in zip(
            text_labels,
            scores,
            boxes,
            strict=True,
        )
    ]

    def matching_candidates(query: str):
        normalized_query = " ".join(query.lower().split())
        exact = [candidate for candidate in candidates if candidate[0] == normalized_query]
        return exact or [candidate for candidate in candidates if normalized_query in candidate[0]]

    def box_iou(first_box, second_box) -> float:
        ix0 = max(first_box[0], second_box[0])
        iy0 = max(first_box[1], second_box[1])
        ix1 = min(first_box[2], second_box[2])
        iy1 = min(first_box[3], second_box[3])
        intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        first_area = max(1.0, first_box[2] - first_box[0]) * max(1.0, first_box[3] - first_box[1])
        second_area = max(1.0, second_box[2] - second_box[0]) * max(
            1.0, second_box[3] - second_box[1]
        )
        return intersection / max(first_area + second_area - intersection, 1.0)

    def deduplicate_candidates(matches):
        retained = []
        for candidate in sorted(matches, key=lambda item: item[1], reverse=True):
            if all(box_iou(candidate[2], existing[2]) < 0.8 for existing in retained):
                retained.append(candidate)
        return retained

    def order_candidates(matches, order: str):
        matches = deduplicate_candidates(matches)
        if order == "confidence":
            return sorted(matches, key=lambda item: item[1], reverse=True)

        def center(candidate):
            box = candidate[2]
            return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

        if order == "top_to_bottom":
            return sorted(matches, key=lambda item: (center(item)[1], -item[1]))
        if order == "bottom_to_top":
            return sorted(matches, key=lambda item: (-center(item)[1], -item[1]))
        if order == "left_to_right":
            return sorted(matches, key=lambda item: (center(item)[0], -item[1]))
        if order == "right_to_left":
            return sorted(matches, key=lambda item: (-center(item)[0], -item[1]))
        raise ValueError(f"Unsupported candidate order: {order}")

    def best_candidate(query: str):
        matches = order_candidates(matching_candidates(query), "confidence")
        return matches[0] if matches else None

    def containment(inner_box, outer_box) -> float:
        ix0 = max(inner_box[0], outer_box[0])
        iy0 = max(inner_box[1], outer_box[1])
        ix1 = min(inner_box[2], outer_box[2])
        iy1 = min(inner_box[3], outer_box[3])
        intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        inner_area = max(1.0, inner_box[2] - inner_box[0]) * max(1.0, inner_box[3] - inner_box[1])
        return intersection / inner_area

    detections = {}
    processed_groups = set()
    for entity in entities:
        group_key = (
            entity.object_name,
            entity.query,
            entity.cardinality,
            entity.candidate_order,
            entity.relation_container_query,
            entity.relation_member_queries,
        )
        if group_key in processed_groups:
            continue
        processed_groups.add(group_key)
        group_entities = sorted(
            (
                candidate_entity
                for candidate_entity in entities
                if (
                    candidate_entity.object_name,
                    candidate_entity.query,
                    candidate_entity.cardinality,
                    candidate_entity.candidate_order,
                    candidate_entity.relation_container_query,
                    candidate_entity.relation_member_queries,
                )
                == group_key
            ),
            key=lambda candidate_entity: candidate_entity.instance_index,
        )

        if entity.relation_container_query is not None:
            container = best_candidate(entity.relation_container_query)
            members = [
                candidate
                for query in entity.relation_member_queries
                if (candidate := best_candidate(query)) is not None
            ]
            if container is None or not members:
                continue
            ranked = [(containment(candidate[2], container[2]), candidate) for candidate in members]
            overlap, selected = max(ranked, key=lambda item: (item[0], item[1][1]))
            selected_candidates = [selected] if overlap >= 0.2 else []
        else:
            selected_candidates = order_candidates(
                matching_candidates(entity.query),
                entity.candidate_order,
            )

        if entity.cardinality == "single":
            selected_candidates = selected_candidates[:1]
        for candidate_entity in group_entities:
            if candidate_entity.instance_index >= len(selected_candidates):
                continue
            _, score, box = selected_candidates[candidate_entity.instance_index]
            if score >= box_threshold:
                detections[candidate_entity.object_id] = (score, box)
    return detections


def score_detection_view(
    image: Image.Image,
    frame_index: int,
    timestamp_s: float,
    score: float,
    box: tuple[float, float, float, float],
    entity: EntitySpec,
    camera: str,
    episode_length: int,
) -> Detection:
    """Score whether a detection is large, sharp, and fully inside the view."""
    width, height = image.size
    x0, y0, x1, y1 = box
    box_width = max(0.0, x1 - x0)
    box_height = max(0.0, y1 - y0)
    area_ratio = box_width * box_height / max(1.0, width * height)
    border_margin = min(x0 / width, y0 / height, (width - x1) / width, (height - y1) / height)
    border_truncated = border_margin < 0.01

    left = int(np.clip(math.floor(x0), 0, width - 1))
    top = int(np.clip(math.floor(y0), 0, height - 1))
    right = int(np.clip(math.ceil(x1), left + 1, width))
    bottom = int(np.clip(math.ceil(y1), top + 1, height))
    crop = np.asarray(image.convert("L"), dtype=np.float32)[top:bottom, left:right]
    if crop.shape[0] > 1 and crop.shape[1] > 1:
        gradient = (np.abs(np.diff(crop, axis=0)).mean() + np.abs(np.diff(crop, axis=1)).mean()) / 2
        sharpness_score = float(np.clip(gradient / 24.0, 0.0, 1.0))
    else:
        sharpness_score = 0.0

    area_score = float(np.clip(np.sqrt(area_ratio / 0.12), 0.0, 1.0))
    border_score = float(np.clip(border_margin / 0.05, 0.0, 1.0))
    if entity.object_name == "collection box":
        quality_score = 0.70 * score + 0.20 * area_score + 0.10 * border_score
    else:
        quality_score = 0.50 * score + 0.30 * area_score + 0.15 * border_score + 0.05 * sharpness_score
    allow_border_truncation = camera == "top" and entity.object_name == "collection box"
    if border_truncated and not allow_border_truncation:
        quality_score -= 0.50
    if area_ratio > 0.65:
        quality_score -= 0.25 * min((area_ratio - 0.65) / 0.35, 1.0)
    episode_progress = frame_index / max(episode_length - 1, 1)
    temporal_penalty = 0.50 * episode_progress if entity.object_role.startswith("source") else 0.0
    selection_score = quality_score - temporal_penalty

    return Detection(
        visible=True,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        score=score,
        box_xyxy=box,
        image_width=width,
        image_height=height,
        area_ratio=area_ratio,
        border_margin=border_margin,
        sharpness_score=sharpness_score,
        quality_score=quality_score,
        episode_progress=episode_progress,
        selection_score=selection_score,
        border_truncated=border_truncated,
    )


def missing_detection(width: int, height: int) -> Detection:
    return Detection(
        visible=False,
        frame_index=-1,
        timestamp_s=None,
        score=None,
        box_xyxy=None,
        image_width=width,
        image_height=height,
        area_ratio=None,
        border_margin=None,
        sharpness_score=None,
        quality_score=None,
        episode_progress=None,
        selection_score=None,
        border_truncated=None,
    )


def retain_ranked_candidate(
    candidates: list[tuple[Detection, Image.Image]],
    detection: Detection,
    image: Image.Image,
    top_k: int,
    minimum_frame_gap: int,
) -> None:
    """Keep high-quality candidates while avoiding nearly identical nearby frames."""
    for index, (existing, _) in enumerate(candidates):
        if abs(existing.frame_index - detection.frame_index) < minimum_frame_gap:
            if detection.selection_score > existing.selection_score:
                candidates[index] = (detection, image.copy())
            break
    else:
        candidates.append((detection, image.copy()))
    candidates.sort(key=lambda item: item[0].selection_score, reverse=True)
    del candidates[top_k:]


def ground_episode_camera(
    source: EpisodeVideoSource,
    episode_index: int,
    camera: str,
    entities: list[EntitySpec],
    dataset_config: GroundingDatasetConfig,
    processor,
    model,
    device: torch.device,
    frame_stride: int,
    max_sampled_frames: int,
    box_threshold: float,
    text_threshold: float,
    top_k: int,
    min_quality: float,
) -> tuple[
    list[Detection],
    dict[int, Image.Image],
    dict[int, list[tuple[Detection, Image.Image]]],
    Image.Image | None,
    int,
]:
    """Scan an episode and retain the best complete-view candidates per entity."""
    ranked_candidates: dict[int, list[tuple[Detection, Image.Image]]] = {
        entity.object_id: [] for entity in entities
    }
    fallback_frame: Image.Image | None = None
    sampled_count = 0
    minimum_frame_gap = max(frame_stride * 2, 1)
    episode_length = int(source.row(episode_index)["length"])
    entities_by_id = {entity.object_id: entity for entity in entities}

    for frame_index, timestamp_s, image in source.sampled_frames(
        episode_index,
        camera,
        frame_stride,
        max_sampled_frames,
    ):
        sampled_count += 1
        if fallback_frame is None:
            fallback_frame = image.copy()
        frame_detections = detect_entities(
            image,
            entities,
            dataset_config,
            processor,
            model,
            device,
            box_threshold,
            text_threshold,
        )
        for object_id, (score, box) in frame_detections.items():
            detection = score_detection_view(
                image,
                frame_index,
                timestamp_s,
                score,
                box,
                entities_by_id[object_id],
                camera,
                episode_length,
            )
            retain_ranked_candidate(
                ranked_candidates[object_id],
                detection,
                image,
                top_k,
                minimum_frame_gap,
            )

    width = fallback_frame.width if fallback_frame is not None else 0
    height = fallback_frame.height if fallback_frame is not None else 0
    accepted_candidates = {
        entity.object_id: next(
            (
                candidate
                for candidate in ranked_candidates[entity.object_id]
                if candidate[0].quality_score >= min_quality
            ),
            None,
        )
        for entity in entities
    }
    detections = [
        accepted_candidates[entity.object_id][0]
        if accepted_candidates[entity.object_id] is not None
        else missing_detection(width, height)
        for entity in entities
    ]
    detection_frames = {
        entity.object_id: accepted_candidates[entity.object_id][1]
        for entity in entities
        if accepted_candidates[entity.object_id] is not None
    }
    return detections, detection_frames, ranked_candidates, fallback_frame, sampled_count


def detection_row(
    episode_index: int,
    camera: str,
    task: str,
    entity: EntitySpec,
    detection: Detection,
    model_path: str,
    frame_stride: int,
    sampled_frames: int,
    candidate_rank: int = 1,
    selected: bool | None = None,
) -> dict:
    box = detection.box_xyxy
    width = detection.image_width
    height = detection.image_height
    row = {
        "episode_index": np.int64(episode_index),
        "camera": camera,
        "task": task,
        **asdict(entity),
        "visible": detection.visible,
        "frame_index": np.int64(detection.frame_index),
        "timestamp_s": detection.timestamp_s if detection.timestamp_s is not None else np.nan,
        "score": detection.score if detection.score is not None else np.nan,
        "quality_score": detection.quality_score if detection.quality_score is not None else np.nan,
        "selection_score": detection.selection_score if detection.selection_score is not None else np.nan,
        "episode_progress": detection.episode_progress if detection.episode_progress is not None else np.nan,
        "area_ratio": detection.area_ratio if detection.area_ratio is not None else np.nan,
        "border_margin": detection.border_margin if detection.border_margin is not None else np.nan,
        "sharpness_score": detection.sharpness_score if detection.sharpness_score is not None else np.nan,
        "border_truncated": detection.border_truncated,
        "candidate_rank": np.int32(candidate_rank),
        "selected": detection.visible and candidate_rank == 1 if selected is None else selected,
        "image_width": np.int32(width),
        "image_height": np.int32(height),
        "model_path": model_path,
        "frame_stride": np.int32(frame_stride),
        "sampled_frames": np.int32(sampled_frames),
    }
    for index, axis in enumerate(("x0", "y0", "x1", "y1")):
        value = box[index] if box is not None else np.nan
        row[f"box_{axis}"] = value
        normalizer = width if axis.startswith("x") else height
        row[f"box_{axis}_norm"] = value / normalizer if normalizer > 0 else np.nan
    return row


def draw_preview_panel(
    image: Image.Image,
    entity: EntitySpec,
    detection: Detection,
    candidate_rank: int | None = None,
) -> Image.Image:
    panel_width, image_height = PREVIEW_PANEL_SIZE
    label_height = 54
    panel = Image.new("RGB", (panel_width, image_height + label_height), color=(25, 25, 25))
    resized = image.resize((panel_width, image_height))
    panel.paste(resized, (0, label_height))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    color = PREVIEW_COLORS[(entity.object_id - 1) % len(PREVIEW_COLORS)]

    draw.text(
        (8, 7),
        f"ID {entity.object_id} | {entity.object_name} | {entity.object_role}"
        + (f" | candidate {candidate_rank}" if candidate_rank is not None else ""),
        fill=color,
        font=font,
    )
    if detection.visible and detection.box_xyxy is not None:
        score_text = "n/a" if detection.score is None else f"{detection.score:.3f}"
        draw.text(
            (8, 29),
            f"frame {detection.frame_index} | conf {score_text} | "
            f"view {detection.quality_score:.3f} | select {detection.selection_score:.3f}",
            fill=(235, 235, 235),
            font=font,
        )
        scale_x = panel_width / detection.image_width
        scale_y = image_height / detection.image_height
        x0, y0, x1, y1 = detection.box_xyxy
        box = (x0 * scale_x, y0 * scale_y + label_height, x1 * scale_x, y1 * scale_y + label_height)
        draw.rectangle(box, outline=color, width=4)
    else:
        draw.text((8, 29), f"NOT FOUND | query: {entity.query}", fill=(255, 100, 100), font=font)
    return panel


def write_contact_sheet(
    path: Path,
    episode_index: int,
    camera: str,
    task: str,
    entities: list[EntitySpec],
    detections: list[Detection],
    detection_frames: dict[int, Image.Image],
    fallback_frame: Image.Image | None,
) -> None:
    if fallback_frame is None:
        fallback_frame = Image.new("RGB", PREVIEW_PANEL_SIZE, color=(40, 40, 40))
    panels = [
        draw_preview_panel(detection_frames.get(entity.object_id, fallback_frame), entity, detection)
        for entity, detection in zip(entities, detections, strict=True)
    ]
    columns = 2
    rows = math.ceil(len(panels) / columns)
    panel_width, panel_height = panels[0].size
    header_height = 72
    sheet = Image.new("RGB", (columns * panel_width, header_height + rows * panel_height), color=(15, 15, 15))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), f"Episode {episode_index} | camera {camera}", fill=(255, 255, 255))
    for line_index, line in enumerate(textwrap.wrap(task, width=105)[:2]):
        draw.text((10, 30 + 17 * line_index), line, fill=(210, 210, 210))
    for index, panel in enumerate(panels):
        x = index % columns * panel_width
        y = header_height + index // columns * panel_height
        sheet.paste(panel, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def write_candidate_contact_sheet(
    path: Path,
    episode_index: int,
    camera: str,
    task: str,
    entities: list[EntitySpec],
    ranked_candidates: dict[int, list[tuple[Detection, Image.Image]]],
    fallback_frame: Image.Image | None,
    top_k: int,
) -> None:
    """Write one row of ranked candidate views for each task entity."""
    if fallback_frame is None:
        fallback_frame = Image.new("RGB", PREVIEW_PANEL_SIZE, color=(40, 40, 40))
    missing = missing_detection(fallback_frame.width, fallback_frame.height)
    panels = []
    for entity in entities:
        candidates = ranked_candidates[entity.object_id]
        for candidate_index in range(top_k):
            if candidate_index < len(candidates):
                detection, image = candidates[candidate_index]
            else:
                detection, image = missing, fallback_frame
            panels.append(draw_preview_panel(image, entity, detection, candidate_index + 1))

    columns = top_k
    rows = len(entities)
    panel_width, panel_height = panels[0].size
    header_height = 72
    sheet = Image.new("RGB", (columns * panel_width, header_height + rows * panel_height), color=(15, 15, 15))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (10, 8),
        f"Episode {episode_index} | camera {camera} | top {top_k} complete-view candidates",
        fill=(255, 255, 255),
    )
    for line_index, line in enumerate(textwrap.wrap(task, width=105)[:2]):
        draw.text((10, 30 + 17 * line_index), line, fill=(210, 210, 210))
    for index, panel in enumerate(panels):
        x = index % columns * panel_width
        y = header_height + index // columns * panel_height
        sheet.paste(panel, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def write_json_atomic(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
        + "\n"
    )
    temporary.replace(path)


def write_parquet_atomic(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def result_paths(
    output_root: Path,
    episode_index: int,
    camera: str,
) -> tuple[Path, Path, Path, Path, Path]:
    stem = f"camera-{camera}"
    episode_dir = f"episode-{episode_index:06d}"
    return (
        output_root / "data" / episode_dir / f"{stem}.parquet",
        output_root / "candidates" / episode_dir / f"{stem}.parquet",
        output_root / "json" / episode_dir / f"{stem}.json",
        output_root / "previews" / episode_dir / f"{stem}.jpg",
        output_root / "candidate_previews" / episode_dir / f"{stem}.jpg",
    )


def result_is_current(
    paths: tuple[Path, Path, Path, Path, Path],
    entities: list[EntitySpec] | None = None,
) -> bool:
    """Return whether artifacts use the current schema and expected entities."""
    if not all(path.is_file() for path in paths):
        return False
    try:
        candidate_columns = set(pd.read_parquet(paths[1]).columns)
        selected = (
            pd.read_parquet(paths[0], columns=["object_id"])
            if entities is not None
            else None
        )
    except Exception:
        return False
    schema_is_current = {"selection_score", "episode_progress"}.issubset(candidate_columns)
    if not schema_is_current or entities is None:
        return schema_is_current
    expected_object_ids = {entity.object_id for entity in entities}
    actual_object_ids = set(int(object_id) for object_id in selected["object_id"])
    return actual_object_ids == expected_object_ids


def rebuild_summary(output_root: Path, detector_name: str = "Grounding DINO") -> None:
    data_paths = sorted((output_root / "data").glob("episode-*/*.parquet"))
    if not data_paths:
        return
    detections = pd.concat((pd.read_parquet(path) for path in data_paths), ignore_index=True)
    detections = detections.sort_values(["episode_index", "camera", "object_id"]).reset_index(drop=True)
    write_parquet_atomic(output_root / "meta" / "detections.parquet", detections)
    candidate_paths = sorted((output_root / "candidates").glob("episode-*/*.parquet"))
    if candidate_paths:
        candidates = pd.concat((pd.read_parquet(path) for path in candidate_paths), ignore_index=True)
        candidates = candidates.sort_values(
            ["episode_index", "camera", "object_id", "candidate_rank"]
        ).reset_index(drop=True)
        write_parquet_atomic(output_root / "meta" / "candidates.parquet", candidates)

    rows = []
    for (episode_index, camera), group in detections.groupby(["episode_index", "camera"], sort=True):
        found = int(group["visible"].sum())
        total = len(group)
        task = str(group.iloc[0]["task"])
        preview = f"../previews/episode-{int(episode_index):06d}/camera-{camera}.jpg"
        candidate_preview = (
            f"../candidate_previews/episode-{int(episode_index):06d}/camera-{camera}.jpg"
        )
        rows.append(
            "<tr>"
            f"<td>{int(episode_index)}</td><td>{html.escape(str(camera))}</td>"
            f"<td>{found}/{total}</td><td>{html.escape(task)}</td>"
            f'<td><a href="{preview}"><img loading="lazy" src="{preview}" width="480"></a></td>'
            f'<td><a href="{candidate_preview}">Compare top candidates</a></td>'
            "</tr>"
        )
    escaped_detector_name = html.escape(detector_name)
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>""" + escaped_detector_name + """ inspection</title>
<style>
body { font-family: sans-serif; margin: 20px; background: #f5f5f5; color: #222; }
table { border-collapse: collapse; width: 100%; background: white; }
th, td { border: 1px solid #ccc; padding: 7px; text-align: left; vertical-align: top; }
th { position: sticky; top: 0; background: #222; color: white; }
img { display: block; }
</style></head><body>
<h1>""" + escaped_detector_name + """ initialization inspection</h1>
<p>Review every selected box and compare alternatives before using detections as SAM2 prompts.</p>
<table><thead><tr>
<th>Episode</th><th>Camera</th><th>Found</th><th>Task</th><th>Selected</th><th>Candidates</th>
</tr></thead>
<tbody>""" + "\n".join(rows) + """</tbody></table></body></html>\n"""
    index_path = output_root / "inspection" / "index.html"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(".html.tmp")
    temporary.write_text(document)
    temporary.replace(index_path)


def setup_logging(output_root: Path, log_filename: str = "grounding_dino.log") -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    log_path = output_root / log_filename
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[console, file_handler], force=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config",
        choices=sorted(DATASET_CONFIGS),
        default=TABLE_PREPARATION_CONFIG.name,
        help="Dataset semantics and default paths to use.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Override the dataset root declared by --dataset-config.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--episodes",
        default="all",
        help="all, comma-separated indices, or end-exclusive ranges such as 0:20,156",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        help="Override the default cameras declared by --dataset-config.",
    )
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3, help="Complete-view candidates retained per object.")
    parser.add_argument(
        "--max-sampled-frames",
        type=int,
        default=0,
        help="Maximum sampled frames per episode-camera; 0 scans the complete episode.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        help="Override the box threshold declared by --dataset-config.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        help="Override the text threshold declared by --dataset-config.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        help="Override the minimum complete-view quality declared by --dataset-config.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    dataset_config = get_dataset_config(args.dataset_config)
    dataset_root = args.dataset_root or dataset_config.default_dataset_root
    cameras = tuple(args.cameras) if args.cameras is not None else dataset_config.cameras
    box_threshold = (
        args.box_threshold if args.box_threshold is not None else dataset_config.box_threshold
    )
    text_threshold = (
        args.text_threshold if args.text_threshold is not None else dataset_config.text_threshold
    )
    min_quality = args.min_quality if args.min_quality is not None else dataset_config.min_quality
    source = EpisodeVideoSource(dataset_root)
    missing_cameras = sorted(set(cameras).difference(source.camera_names))
    if missing_cameras:
        available = ", ".join(source.camera_names)
        raise ValueError(
            f"Dataset does not contain cameras {missing_cameras}; available cameras: {available}"
        )
    episode_indices = parse_episode_selection(args.episodes, source.episode_indices)
    task_entities = {
        episode_index: dataset_config.entities_from_task(source.task(episode_index))
        for episode_index in episode_indices
    }
    output_root = (
        args.output_root
        if args.output_root is not None
        else WORKSPACE / "runs" / "sam" / source.root.name / "grounding_dino"
    ).resolve()
    setup_logging(output_root)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model_path = args.model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    metadata = {
        "format_version": 3,
        "role": "Grounding DINO initialization proposals for SAM2",
        "dataset_config": dataset_config.name,
        "dataset_root": str(source.root),
        "model_path": str(model_path),
        "episodes": episode_indices,
        "cameras": cameras,
        "camera_entity_names": {
            camera: sorted(names)
            for camera, names in dataset_config.camera_entity_names.items()
            if camera in cameras
        },
        "frame_stride": args.frame_stride,
        "top_k": args.top_k,
        "max_sampled_frames": args.max_sampled_frames,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "min_quality": min_quality,
        "object_id_scope": "episode; IDs are identical across cameras for the same episode",
        "selection": "highest complete-view quality score across sampled frames",
        "quality_score": (
            "View quality combines confidence, visible area, border clearance, and sharpness; "
            "border-truncated and oversized boxes are penalized"
        ),
        "selection_score": "view quality minus 0.50*episode_progress for source-role entities",
    }
    write_json_atomic(output_root / "meta" / "info.json", metadata)

    logger.info("Loading Grounding DINO from %s on %s", model_path, device)
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(model_path), local_files_only=True
    ).to(device).eval()

    total_jobs = len(episode_indices) * len(cameras)
    completed_jobs = 0
    for episode_index in episode_indices:
        task = source.task(episode_index)
        for camera in cameras:
            entities = dataset_config.entities_for_camera(task_entities[episode_index], camera)
            parquet_path, candidates_path, json_path, preview_path, candidate_preview_path = result_paths(
                output_root,
                episode_index,
                camera,
            )
            completed_outputs = (
                parquet_path,
                candidates_path,
                json_path,
                preview_path,
                candidate_preview_path,
            )
            if result_is_current(completed_outputs, entities) and not args.overwrite:
                completed_jobs += 1
                logger.info(
                    "[%d/%d] Skipping episode %d camera %s; output exists",
                    completed_jobs,
                    total_jobs,
                    episode_index,
                    camera,
                )
                continue

            logger.info(
                "[%d/%d] Grounding episode %d camera %s: %s",
                completed_jobs + 1,
                total_jobs,
                episode_index,
                camera,
                task,
            )
            (
                detections,
                detection_frames,
                ranked_candidates,
                fallback_frame,
                sampled_count,
            ) = ground_episode_camera(
                source,
                episode_index,
                camera,
                entities,
                dataset_config,
                processor,
                model,
                device,
                args.frame_stride,
                args.max_sampled_frames,
                box_threshold,
                text_threshold,
                args.top_k,
                min_quality,
            )
            records = []
            for entity, detection in zip(entities, detections, strict=True):
                selected_rank = next(
                    (
                        rank
                        for rank, (candidate, _) in enumerate(
                            ranked_candidates[entity.object_id],
                            start=1,
                        )
                        if candidate == detection
                    ),
                    1,
                )
                records.append(
                    detection_row(
                        episode_index,
                        camera,
                        task,
                        entity,
                        detection,
                        str(model_path),
                        args.frame_stride,
                        sampled_count,
                        selected_rank,
                        selected=detection.visible,
                    )
                )
            candidate_records = []
            for entity, selected_detection in zip(entities, detections, strict=True):
                candidates = ranked_candidates[entity.object_id]
                ranked_detections = [item[0] for item in candidates] or [selected_detection]
                for candidate_rank, detection in enumerate(ranked_detections, start=1):
                    candidate_records.append(
                        detection_row(
                            episode_index,
                            camera,
                            task,
                            entity,
                            detection,
                            str(model_path),
                            args.frame_stride,
                            sampled_count,
                            candidate_rank,
                            selected=selected_detection.visible and detection == selected_detection,
                        )
                    )
            write_parquet_atomic(parquet_path, pd.DataFrame(records))
            write_parquet_atomic(candidates_path, pd.DataFrame(candidate_records))
            write_json_atomic(json_path, {"selected": records, "candidates": candidate_records})
            write_contact_sheet(
                preview_path,
                episode_index,
                camera,
                task,
                entities,
                detections,
                detection_frames,
                fallback_frame,
            )
            write_candidate_contact_sheet(
                candidate_preview_path,
                episode_index,
                camera,
                task,
                entities,
                ranked_candidates,
                fallback_frame,
                args.top_k,
            )
            completed_jobs += 1
            found_count = sum(detection.visible for detection in detections)
            logger.info(
                "[%d/%d] Saved best views for %d/%d objects after scanning %d sampled frames",
                completed_jobs,
                total_jobs,
                found_count,
                len(detections),
                sampled_count,
            )

    rebuild_summary(output_root)
    logger.info("Detection table: %s", output_root / "meta" / "detections.parquet")
    logger.info("Candidate table: %s", output_root / "meta" / "candidates.parquet")
    logger.info("Inspection index: %s", output_root / "inspection" / "index.html")


if __name__ == "__main__":
    main()
