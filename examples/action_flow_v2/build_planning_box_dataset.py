#!/usr/bin/env python

"""Merge RM65 robot data, reviewed planning segments, and stage-keyframe boxes.

The source robot dataset remains unchanged. Every output frame receives its
reviewed planning stage, cycle, entities, and next-step text. Only the single
manually verified synchronized keyframe in each ``grasp`` or ``put`` segment
receives Action Flow V2.2 box supervision; all other frames have zero box-loss
weight. Original observations, actions, videos, task indices, and task prompts
are preserved.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.policies.action_flow_v2_2.data_contracts_action_flow_v2_2 import (
    TARGET_BOX_BINS,
    TARGET_BOXES,
    TARGET_BOX_LOSS_WEIGHT,
    TARGET_BOX_VISIBILITY,
    TARGET_ENTITY_IDS,
    TARGET_ENTITY_ROLES,
    TARGET_ENTITY_VALID,
    ActionFlowV22DatasetMetadata,
    TargetRole,
    TargetVisibility,
)

from action_flow_v2_2_box_utils import compute_resize_geometry, transform_bounds


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_ROOT = WORKSPACE / "runs/dataset/rm65_single_table_preparation_and_bussing"
DEFAULT_PLANNING_ROOT = WORKSPACE / "runs/planning/rm65_single_table_preparation_and_bussing"
DEFAULT_BOX_ROOT = WORKSPACE / "runs/sam/rm65_single_table_preparation_and_bussing_stage_keyframes"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE
    / "runs/dataset/rm65_single_table_preparation_and_bussing_afv2_2_planning_boxes"
)

PLANNING_STAGE = "planning.stage"
PLANNING_CYCLE_INDEX = "planning.cycle_index"
PLANNING_SEGMENT_INDEX = "planning.segment_index"
PLANNING_SOURCE_ENTITY = "planning.source_entity"
PLANNING_DESTINATION_ENTITY = "planning.destination_entity"
PLANNING_TARGET_ENTITY = "planning.target_entity"
PLANNING_NEXT_STEP = "planning.next_step"
PLANNING_SEGMENT_UID = "planning.segment_uid"
PLANNING_BOX_KEYFRAME = "planning.box_keyframe"
PLANNING_FRAMES_REMAINING = "planning.frames_remaining"

PLANNING_FEATURES = {
    PLANNING_STAGE: {"dtype": "string", "shape": [1], "names": None},
    PLANNING_CYCLE_INDEX: {"dtype": "int64", "shape": [1], "names": None},
    PLANNING_SEGMENT_INDEX: {"dtype": "int64", "shape": [1], "names": None},
    PLANNING_SOURCE_ENTITY: {"dtype": "string", "shape": [1], "names": None},
    PLANNING_DESTINATION_ENTITY: {"dtype": "string", "shape": [1], "names": None},
    PLANNING_TARGET_ENTITY: {"dtype": "string", "shape": [1], "names": None},
    PLANNING_NEXT_STEP: {"dtype": "string", "shape": [1], "names": None},
    PLANNING_SEGMENT_UID: {"dtype": "string", "shape": [1], "names": None},
    PLANNING_BOX_KEYFRAME: {"dtype": "uint8", "shape": [1], "names": None},
    PLANNING_FRAMES_REMAINING: {"dtype": "int64", "shape": [1], "names": None},
}

CAMERAS = ("wrist", "top")
STAGES = ("grasp", "put", "complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-root", type=Path, default=DEFAULT_ROBOT_ROOT)
    parser.add_argument("--planning-root", type=Path, default=DEFAULT_PLANNING_ROOT)
    parser.add_argument("--box-root", type=Path, default=DEFAULT_BOX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-width", type=int, default=512)
    parser.add_argument("--target-height", type=int, default=512)
    parser.add_argument("--coordinate-bins", type=int, default=256)
    parser.add_argument("--max-entities", type=int, default=8)
    parser.add_argument("--write-batch-size", type=int, default=128)
    parser.add_argument(
        "--video-copy-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted build from the .incomplete directory.",
    )
    return parser.parse_args()


def load_info(root: Path) -> dict[str, Any]:
    path = root / "meta/info.json"
    if not path.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {root}")
    return json.loads(path.read_text())


def load_episode_metadata(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {root}.")
    episodes = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    if episodes["episode_index"].duplicated().any():
        raise ValueError("Episode metadata contains duplicate episode indices.")
    return episodes.set_index("episode_index", drop=False).sort_index()


def canonical_entity_name(name: str) -> str:
    """Match the entity spelling used by the existing AFV2.2 tokenizer."""

    cleaned = " ".join(str(name).strip().lower().split())
    if cleaned.startswith("vegetable on the "):
        return "vegetable on " + cleaned.removeprefix("vegetable on the ")
    return cleaned


def stage_target(row: pd.Series | dict[str, Any]) -> str:
    stage = str(row["stage"])
    if stage == "grasp":
        return str(row["source_entity"]).strip()
    if stage == "put":
        return str(row["destination_entity"]).strip()
    if stage == "complete":
        return ""
    raise ValueError(f"Unsupported planning stage: {stage!r}")


def target_role(stage: str, target: str) -> TargetRole:
    if stage == "grasp":
        return TargetRole.SOURCE
    if stage == "put" and canonical_entity_name(target) == "collection box":
        return TargetRole.DESTINATION_CONTAINER
    if stage == "put":
        return TargetRole.DESTINATION
    raise ValueError(f"Stage {stage!r} does not have a target role.")


def validate_sources(
    episodes: pd.DataFrame,
    planning: pd.DataFrame,
    boxes: pd.DataFrame,
    *,
    camera_names: tuple[str, ...],
) -> None:
    episode_indices = [int(value) for value in episodes.index]
    if episode_indices != list(range(len(episode_indices))):
        raise ValueError("Robot episodes must use contiguous zero-based indices.")
    if sorted(planning["episode_index"].astype(int).unique()) != episode_indices:
        raise ValueError("Planning annotations do not cover every robot episode.")
    if not set(planning["stage"]).issubset(STAGES):
        raise ValueError(f"Planning data contains unsupported stages: {set(planning['stage'])}.")
    if planning["next_step"].astype(str).str.contains("{target}", regex=False).any():
        raise ValueError("Planning data still contains unresolved {target} placeholders.")

    for episode_index, episode_segments in planning.groupby("episode_index", sort=True):
        episode_segments = episode_segments.sort_values("segment_index")
        expected_length = int(episodes.loc[int(episode_index), "length"])
        starts = episode_segments["start_frame"].astype(int).to_numpy()
        ends = episode_segments["end_frame"].astype(int).to_numpy()
        if starts[0] != 0 or ends[-1] != expected_length:
            raise ValueError(f"Episode {episode_index} planning ranges do not cover the episode.")
        if np.any(ends <= starts) or not np.array_equal(starts[1:], ends[:-1]):
            raise ValueError(f"Episode {episode_index} planning ranges are not contiguous.")

    manipulation = planning.loc[planning["stage"].isin(["grasp", "put"])].copy()
    expected_keys = set(
        zip(
            manipulation["episode_index"].astype(int),
            manipulation["segment_index"].astype(int),
            strict=True,
        )
    )
    actual_keys = set(
        zip(boxes["episode_index"].astype(int), boxes["segment_index"].astype(int), strict=True)
    )
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:10]
        extra = sorted(actual_keys - expected_keys)[:10]
        raise ValueError(f"Stage-keyframe labels differ from planning data; missing={missing}, extra={extra}.")

    expected_rows = len(expected_keys) * len(camera_names)
    if len(boxes) != expected_rows:
        raise ValueError(f"Expected {expected_rows} camera labels, found {len(boxes)}.")
    if boxes.duplicated(["episode_index", "segment_index", "camera"]).any():
        raise ValueError("Stage-keyframe labels contain duplicate camera rows.")
    if set(boxes["camera"]) != set(camera_names):
        raise ValueError(f"Box cameras {set(boxes['camera'])} do not match {camera_names}.")
    if not boxes["reviewed"].astype(bool).all():
        raise ValueError("Every stage-keyframe camera label must be reviewed.")

    shared_frames = boxes.groupby(["episode_index", "segment_index"])["frame_index"].nunique()
    if not (shared_frames == 1).all():
        raise ValueError("Top and wrist labels must use one synchronized frame per stage.")
    for row in boxes.itertuples(index=False):
        if not int(row.start_frame) <= int(row.frame_index) < int(row.end_frame):
            raise ValueError(
                f"Episode {row.episode_index} segment {row.segment_index} keyframe is outside its range."
            )
        if bool(row.visible):
            if row.bbox_xyxy is None or len(row.bbox_xyxy) != 4:
                raise ValueError("A visible reviewed target requires one xyxy box.")
        elif row.bbox_xyxy is not None:
            raise ValueError("A not-visible target must not contain a box.")


def copy_or_link(source: str, destination: str, *, mode: str) -> str:
    if mode == "copy":
        return shutil.copy2(source, destination)
    if mode == "symlink":
        os.symlink(os.path.abspath(source), destination)
        return destination
    try:
        os.link(source, destination)
        return destination
    except OSError:
        logging.warning("Hard link failed for %s; copying it instead.", source)
        return shutil.copy2(source, destination)


def prepare_output(
    robot_root: Path,
    output_root: Path,
    *,
    metadata: ActionFlowV22DatasetMetadata,
    video_copy_mode: str,
) -> datasets.Features:
    output_root.mkdir(parents=True)
    shutil.copytree(robot_root / "meta", output_root / "meta", copy_function=shutil.copy2)
    shutil.copytree(
        robot_root / "videos",
        output_root / "videos",
        copy_function=lambda source, destination: copy_or_link(
            source,
            destination,
            mode=video_copy_mode,
        ),
    )

    info_path = output_root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"].update(PLANNING_FEATURES)
    info["features"].update(metadata.feature_metadata())
    info_path.write_text(json.dumps(info, indent=4) + "\n")
    metadata.validate_feature_metadata(info["features"])
    metadata.save(output_root / "meta/action_flow_v2_2_boxes.json")

    features = info["features"]
    for feature in features.values():
        feature["shape"] = tuple(feature["shape"])
    return get_hf_features_from_features(features)


def source_data_path(robot_root: Path, episode_row: pd.Series, info: dict[str, Any]) -> Path:
    return robot_root / info["data_path"].format(
        chunk_index=int(episode_row["data/chunk_index"]),
        file_index=int(episode_row["data/file_index"]),
    )


def build_episode_labels(
    episode_row: pd.Series,
    segments: pd.DataFrame,
    box_rows: pd.DataFrame,
    *,
    camera_names: tuple[str, ...],
    entity_name_to_id: dict[str, int],
    max_entities: int,
    target_width: int,
    target_height: int,
    coordinate_bins: int,
    source_width: int,
    source_height: int,
) -> dict[str, Any]:
    frame_count = int(episode_row["length"])
    camera_count = len(camera_names)
    labels: dict[str, Any] = {
        PLANNING_STAGE: np.full(frame_count, "", dtype=object),
        PLANNING_CYCLE_INDEX: np.full(frame_count, -1, dtype=np.int64),
        PLANNING_SEGMENT_INDEX: np.full(frame_count, -1, dtype=np.int64),
        PLANNING_SOURCE_ENTITY: np.full(frame_count, "", dtype=object),
        PLANNING_DESTINATION_ENTITY: np.full(frame_count, "", dtype=object),
        PLANNING_TARGET_ENTITY: np.full(frame_count, "", dtype=object),
        PLANNING_NEXT_STEP: np.full(frame_count, "", dtype=object),
        PLANNING_SEGMENT_UID: np.full(frame_count, "", dtype=object),
        PLANNING_BOX_KEYFRAME: np.zeros(frame_count, dtype=np.uint8),
        PLANNING_FRAMES_REMAINING: np.zeros(frame_count, dtype=np.int64),
        TARGET_BOXES: np.zeros((frame_count, camera_count, max_entities, 4), dtype=np.float32),
        TARGET_BOX_BINS: np.zeros((frame_count, camera_count, max_entities, 4), dtype=np.int64),
        TARGET_BOX_VISIBILITY: np.zeros((frame_count, camera_count, max_entities), dtype=np.uint8),
        TARGET_BOX_LOSS_WEIGHT: np.zeros((frame_count, camera_count, max_entities), dtype=np.float32),
        TARGET_ENTITY_IDS: np.full((frame_count, max_entities), -1, dtype=np.int64),
        TARGET_ENTITY_ROLES: np.zeros((frame_count, max_entities), dtype=np.int64),
        TARGET_ENTITY_VALID: np.zeros((frame_count, max_entities), dtype=np.uint8),
    }

    for segment in segments.sort_values("segment_index").to_dict("records"):
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        target = stage_target(segment)
        labels[PLANNING_STAGE][start:end] = str(segment["stage"])
        labels[PLANNING_CYCLE_INDEX][start:end] = int(segment["cycle_index"])
        labels[PLANNING_SEGMENT_INDEX][start:end] = int(segment["segment_index"])
        labels[PLANNING_SOURCE_ENTITY][start:end] = str(segment["source_entity"]).strip()
        labels[PLANNING_DESTINATION_ENTITY][start:end] = str(
            segment["destination_entity"]
        ).strip()
        labels[PLANNING_TARGET_ENTITY][start:end] = target
        labels[PLANNING_NEXT_STEP][start:end] = str(segment["next_step"]).strip()
        labels[PLANNING_SEGMENT_UID][start:end] = str(segment["segment_uid"])
        labels[PLANNING_FRAMES_REMAINING][start:end] = np.arange(
            end - start, 0, -1, dtype=np.int64
        )

    geometry = compute_resize_geometry(
        source_width,
        source_height,
        target_width,
        target_height,
    )
    camera_to_index = {camera: index for index, camera in enumerate(camera_names)}
    for (segment_index, frame_index), stage_boxes in box_rows.groupby(
        ["segment_index", "frame_index"], sort=True
    ):
        frame_index = int(frame_index)
        first = stage_boxes.iloc[0]
        stage = str(first["stage"])
        raw_target = str(first["target_entity"])
        model_target = canonical_entity_name(raw_target)
        entity_id = entity_name_to_id[model_target]
        labels[PLANNING_BOX_KEYFRAME][frame_index] = 1
        labels[TARGET_ENTITY_IDS][frame_index, 0] = entity_id
        labels[TARGET_ENTITY_ROLES][frame_index, 0] = int(target_role(stage, raw_target))
        labels[TARGET_ENTITY_VALID][frame_index, 0] = 1

        if len(stage_boxes) != len(camera_names):
            raise ValueError(
                f"Segment {segment_index} frame {frame_index} does not cover every camera."
            )
        for row in stage_boxes.to_dict("records"):
            camera_index = camera_to_index[str(row["camera"])]
            labels[TARGET_BOX_LOSS_WEIGHT][frame_index, camera_index, 0] = 1.0
            if not bool(row["visible"]):
                continue
            x1, y1, x2_exclusive, y2_exclusive = (int(value) for value in row["bbox_xyxy"])
            if not (x1 < x2_exclusive and y1 < y2_exclusive):
                raise ValueError(f"Invalid raw box {row['bbox_xyxy']}.")
            normalized, quantized = transform_bounds(
                (x1, y1, x2_exclusive - 1, y2_exclusive - 1),
                geometry,
                coordinate_bins=coordinate_bins,
            )
            labels[TARGET_BOXES][frame_index, camera_index, 0] = normalized
            labels[TARGET_BOX_BINS][frame_index, camera_index, 0] = quantized
            labels[TARGET_BOX_VISIBILITY][frame_index, camera_index, 0] = int(
                TargetVisibility.VISIBLE
            )

    if np.any(labels[PLANNING_SEGMENT_INDEX] < 0):
        raise ValueError(f"Episode {int(episode_row['episode_index'])} has unlabeled frames.")
    return labels


def write_labeled_data_file(
    source_path: Path,
    target_path: Path,
    *,
    labels_by_episode: dict[int, dict[str, Any]],
    hf_features: datasets.Features,
    write_batch_size: int,
) -> int:
    source = pd.read_parquet(source_path)
    row_count = len(source)
    merged: dict[str, Any] = {}
    for key, first in next(iter(labels_by_episode.values())).items():
        if first.dtype == object:
            merged[key] = np.full(row_count, "", dtype=object)
        else:
            merged[key] = np.zeros((row_count, *first.shape[1:]), dtype=first.dtype)
            if key == TARGET_ENTITY_IDS:
                merged[key].fill(-1)

    for episode_index, episode_labels in labels_by_episode.items():
        positions = np.flatnonzero(source["episode_index"].to_numpy(dtype=np.int64) == episode_index)
        frame_indices = source.iloc[positions]["frame_index"].to_numpy(dtype=np.int64)
        frame_count = len(episode_labels[PLANNING_STAGE])
        if len(positions) != frame_count or not np.array_equal(
            np.sort(frame_indices), np.arange(frame_count)
        ):
            raise ValueError(f"Source rows do not match episode {episode_index} frame indices.")
        for key, values in episode_labels.items():
            merged[key][positions] = values[frame_indices]

    target_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for start in range(0, row_count, write_batch_size):
            end = min(start + write_batch_size, row_count)
            columns: dict[str, Any] = source.iloc[start:end].to_dict(orient="list")
            for key, values in merged.items():
                columns[key] = values[start:end].tolist() if values.dtype == object else values[start:end]
            dataset = datasets.Dataset.from_dict(columns, features=hf_features, split="train")
            table = dataset.data.table
            if writer is None:
                writer = pq.ParquetWriter(
                    target_path,
                    table.schema,
                    compression="snappy",
                    use_dictionary=True,
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return row_count


def copy_annotation_tables(
    output_root: Path,
    planning_root: Path,
    box_root: Path,
    planning: pd.DataFrame,
    boxes: pd.DataFrame,
    entity_name_to_id: dict[str, int],
    episodes: pd.DataFrame,
) -> None:
    planning_dir = output_root / "annotations/planning"
    box_dir = output_root / "annotations/stage_keyframes"
    planning_dir.mkdir(parents=True, exist_ok=True)
    box_dir.mkdir(parents=True, exist_ok=True)
    planning.to_parquet(planning_dir / "segments.parquet", index=False)
    episodes_path = planning_root / "review/episodes.parquet"
    if episodes_path.is_file():
        shutil.copy2(episodes_path, planning_dir / "episodes.parquet")

    output_boxes = boxes.copy()
    output_boxes["model_entity_name"] = output_boxes["target_entity"].map(
        canonical_entity_name
    )
    output_boxes["entity_id"] = output_boxes["model_entity_name"].map(entity_name_to_id)
    output_boxes["entity_role"] = [
        int(target_role(stage, target))
        for stage, target in zip(
            output_boxes["stage"], output_boxes["target_entity"], strict=True
        )
    ]
    output_boxes.to_parquet(box_dir / "boxes.parquet", index=False)
    keyframes = output_boxes.drop_duplicates(
        ["episode_index", "segment_index", "frame_index"]
    ).copy()
    keyframes["dataset_index"] = [
        int(episodes.loc[int(episode_index), "dataset_from_index"]) + int(frame_index)
        for episode_index, frame_index in zip(
            keyframes["episode_index"], keyframes["frame_index"], strict=True
        )
    ]
    keyframes.to_parquet(box_dir / "keyframes.parquet", index=False)
    annotations_path = box_root / "review/annotations.json"
    if annotations_path.is_file():
        shutil.copy2(annotations_path, box_dir / "annotations.json")


def validate_output(
    output_root: Path,
    *,
    metadata: ActionFlowV22DatasetMetadata,
    expected_frames: int,
    expected_episodes: int,
    expected_keyframes: int,
    expected_visible_labels: int,
    expected_not_visible_labels: int,
) -> dict[str, Any]:
    info = load_info(output_root)
    metadata.validate_feature_metadata(info["features"])
    files = sorted((output_root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise ValueError("Output dataset contains no data files.")

    total_frames = 0
    keyframes = 0
    weighted_labels = 0
    visible_labels = 0
    not_visible_labels = 0
    stage_counts = {stage: 0 for stage in STAGES}
    for path in files:
        table = pq.read_table(
            path,
            columns=[
                PLANNING_STAGE,
                PLANNING_BOX_KEYFRAME,
                PLANNING_FRAMES_REMAINING,
                TARGET_BOX_VISIBILITY,
                TARGET_BOX_LOSS_WEIGHT,
                TARGET_ENTITY_VALID,
            ],
        )
        total_frames += table.num_rows
        stages = np.asarray(table[PLANNING_STAGE].to_pylist(), dtype=object)
        for stage in STAGES:
            stage_counts[stage] += int(np.count_nonzero(stages == stage))
        frame_keyframes = np.asarray(table[PLANNING_BOX_KEYFRAME].to_numpy(), dtype=np.uint8)
        frames_remaining = np.asarray(
            table[PLANNING_FRAMES_REMAINING].to_numpy(), dtype=np.int64
        )
        if np.any(frames_remaining <= 0):
            raise ValueError(f"Planning frames remaining must be positive in {path}.")
        keyframes += int(frame_keyframes.sum())
        weights = np.asarray(table[TARGET_BOX_LOSS_WEIGHT].to_pylist(), dtype=np.float32)
        visibility = np.asarray(table[TARGET_BOX_VISIBILITY].to_pylist(), dtype=np.uint8)
        valid = np.asarray(table[TARGET_ENTITY_VALID].to_pylist(), dtype=np.uint8)
        weighted = weights > 0
        weighted_labels += int(weighted.sum())
        visible_labels += int(np.count_nonzero(weighted & (visibility == int(TargetVisibility.VISIBLE))))
        not_visible_labels += int(
            np.count_nonzero(weighted & (visibility == int(TargetVisibility.NOT_VISIBLE)))
        )
        if np.any(weights[frame_keyframes == 0]):
            raise ValueError(f"Non-keyframe rows contain box-loss weight in {path}.")
        if np.any(valid[frame_keyframes == 0]):
            raise ValueError(f"Non-keyframe rows contain valid box entities in {path}.")

    expected = {
        "total_frames": expected_frames,
        "total_episodes": expected_episodes,
        "keyframes": expected_keyframes,
        "weighted_camera_labels": expected_keyframes * len(metadata.camera_names),
        "visible_camera_labels": expected_visible_labels,
        "not_visible_camera_labels": expected_not_visible_labels,
    }
    actual = {
        "total_frames": total_frames,
        "total_episodes": int(info["total_episodes"]),
        "keyframes": keyframes,
        "weighted_camera_labels": weighted_labels,
        "visible_camera_labels": visible_labels,
        "not_visible_camera_labels": not_visible_labels,
    }
    if actual != expected:
        raise ValueError(f"Output validation mismatch: actual={actual}, expected={expected}.")
    if sum(stage_counts.values()) != total_frames:
        raise ValueError("Planning stages do not cover every output frame.")

    result = {
        "valid": True,
        **actual,
        "planning_frame_counts": stage_counts,
        "camera_order": list(metadata.camera_names),
        "entity_vocabulary": list(metadata.entity_names),
        "max_entities": metadata.max_entities,
        "coordinate_bins": metadata.coordinate_bins,
        "box_supervision_rule": "one synchronized reviewed keyframe per grasp/put stage",
        "non_keyframe_box_loss_weight": 0.0,
    }
    (output_root / "meta/action_flow_v2_2_planning_boxes_validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    for value, name in (
        (args.target_width, "target width"),
        (args.target_height, "target height"),
        (args.coordinate_bins, "coordinate bins"),
        (args.max_entities, "max entities"),
        (args.write_batch_size, "write batch size"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.coordinate_bins < 2:
        raise ValueError("Coordinate bins must be at least 2.")
    if args.output_root.exists():
        raise FileExistsError(f"Output dataset already exists: {args.output_root}")

    incomplete_root = args.output_root.with_name(args.output_root.name + ".incomplete")
    if incomplete_root.exists() and not args.resume:
        raise FileExistsError(
            f"Incomplete output exists: {incomplete_root}. Inspect it and rerun with --resume."
        )

    robot_info = load_info(args.robot_root)
    episodes = load_episode_metadata(args.robot_root)
    planning_path = args.planning_root / "review/segments.parquet"
    boxes_path = args.box_root / "review/boxes.parquet"
    if not planning_path.is_file() or not boxes_path.is_file():
        raise FileNotFoundError("Reviewed planning segments or stage-keyframe boxes are missing.")
    planning = pd.read_parquet(planning_path).sort_values(
        ["episode_index", "segment_index"]
    )
    boxes = pd.read_parquet(boxes_path).sort_values(
        ["episode_index", "segment_index", "camera"]
    )
    validate_sources(episodes, planning, boxes, camera_names=CAMERAS)

    entity_vocabulary = tuple(
        sorted({canonical_entity_name(name) for name in boxes["target_entity"]})
    )
    entity_name_to_id = {
        name: entity_id for entity_id, name in enumerate(entity_vocabulary)
    }
    metadata = ActionFlowV22DatasetMetadata(
        camera_names=CAMERAS,
        entity_names=entity_vocabulary,
        max_entities=args.max_entities,
        coordinate_bins=args.coordinate_bins,
    )

    if incomplete_root.exists():
        logging.info("Resuming output dataset at %s", incomplete_root)
        info = load_info(incomplete_root)
        metadata.validate_feature_metadata(info["features"])
        features = info["features"]
        for feature in features.values():
            feature["shape"] = tuple(feature["shape"])
        hf_features = get_hf_features_from_features(features)
    else:
        logging.info("Preparing output dataset at %s", incomplete_root)
        hf_features = prepare_output(
            args.robot_root,
            incomplete_root,
            metadata=metadata,
            video_copy_mode=args.video_copy_mode,
        )
        copy_annotation_tables(
            incomplete_root,
            args.planning_root,
            args.box_root,
            planning,
            boxes,
            entity_name_to_id,
            episodes,
        )
        provenance = {
            "format": "rm65_action_flow_v2_2_planning_stage_boxes",
            "schema_version": 1,
            "camera_order": list(CAMERAS),
            "policy_image_size": [args.target_height, args.target_width],
            "coordinate_bins": args.coordinate_bins,
            "planning_source": str(args.planning_root),
            "box_source": str(args.box_root),
            "frame_semantics": (
                "planning labels and frames_remaining on every frame; boxes only on "
                "reviewed synchronized keyframes"
            ),
            "original_task_prompts_preserved": True,
            "video_copy_mode": args.video_copy_mode,
            "entity_name_aliases": {
                str(name): canonical_entity_name(name)
                for name in sorted(boxes["target_entity"].unique())
                if str(name) != canonical_entity_name(name)
            },
        }
        (incomplete_root / "meta/action_flow_v2_2_planning_boxes.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )

    image_feature = robot_info["features"]["observation.images.wrist"]
    source_height, source_width = map(int, image_feature["shape"][:2])
    data_groups = list(episodes.groupby(["data/chunk_index", "data/file_index"], sort=False))
    total_written = 0
    for group_number, (_, episode_rows) in enumerate(data_groups, start=1):
        source_path = source_data_path(args.robot_root, episode_rows.iloc[0], robot_info)
        target_path = incomplete_root / source_path.relative_to(args.robot_root)
        if args.resume and target_path.is_file():
            source_rows = pq.ParquetFile(source_path).metadata.num_rows
            target_file = pq.ParquetFile(target_path)
            if target_file.metadata.num_rows == source_rows and PLANNING_STAGE in target_file.schema_arrow.names:
                total_written += source_rows
                continue
            target_path.unlink(missing_ok=True)

        labels_by_episode: dict[int, dict[str, Any]] = {}
        for episode_index in episode_rows["episode_index"].astype(int):
            labels_by_episode[episode_index] = build_episode_labels(
                episodes.loc[episode_index],
                planning.loc[planning["episode_index"] == episode_index],
                boxes.loc[boxes["episode_index"] == episode_index],
                camera_names=CAMERAS,
                entity_name_to_id=entity_name_to_id,
                max_entities=args.max_entities,
                target_width=args.target_width,
                target_height=args.target_height,
                coordinate_bins=args.coordinate_bins,
                source_width=source_width,
                source_height=source_height,
            )
        written = write_labeled_data_file(
            source_path,
            target_path,
            labels_by_episode=labels_by_episode,
            hf_features=hf_features,
            write_batch_size=args.write_batch_size,
        )
        total_written += written
        if group_number == 1 or group_number % 25 == 0 or group_number == len(data_groups):
            logging.info(
                "[%d/%d] Wrote %d frames from %d episode(s).",
                group_number,
                len(data_groups),
                written,
                len(labels_by_episode),
            )

    if total_written != int(robot_info["total_frames"]):
        raise ValueError(
            f"Wrote {total_written} frames; source metadata expects {robot_info['total_frames']}."
        )
    manipulation_segments = int(planning["stage"].isin(["grasp", "put"]).sum())
    validation = validate_output(
        incomplete_root,
        metadata=metadata,
        expected_frames=int(robot_info["total_frames"]),
        expected_episodes=int(robot_info["total_episodes"]),
        expected_keyframes=manipulation_segments,
        expected_visible_labels=int(boxes["visible"].astype(bool).sum()),
        expected_not_visible_labels=int((~boxes["visible"].astype(bool)).sum()),
    )
    incomplete_root.rename(args.output_root)
    logging.info(
        "Created %s: %d episodes, %d frames, %d planning keyframes.",
        args.output_root,
        validation["total_episodes"],
        validation["total_frames"],
        validation["keyframes"],
    )


if __name__ == "__main__":
    main()
