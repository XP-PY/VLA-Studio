#!/usr/bin/env python

"""Build a LeRobot dataset with Action Flow V2.2 entity-box supervision.

The script converts reviewed, object-specific SAM2 masks to normalized boxes
without decoding dense masks. It preserves the source robot data and videos,
adds fixed-size box tensors to each training row, saves long-form annotations,
and creates a compact HTML review with sampled preview videos.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import datasets
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

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

from action_flow_v2_2_box_utils import (
    build_camera_labels,
    compute_resize_geometry,
    discover_entity_vocabulary,
    load_episode_definitions,
    load_review_visibility_decisions,
)

DEFAULT_ROBOT_ROOT = Path("runs/dataset/rm65_single_table_preparation_and_bussing")
DEFAULT_SAM_ROOT = Path("runs/sam/rm65_single_table_preparation_and_bussing_yoloe")
DEFAULT_GROUNDING_ROOT = Path("runs/yoloe/rm65_single_table_preparation_and_bussing")
DEFAULT_OUTPUT_ROOT = Path(
    "runs/dataset/rm65_single_table_preparation_and_bussing_afv2_2_yoloe_boxes"
)

ROLE_COLORS = {
    int(TargetRole.SOURCE): (239, 68, 68),
    int(TargetRole.DESTINATION): (59, 130, 246),
    int(TargetRole.SOURCE_CONTAINER): (245, 158, 11),
    int(TargetRole.DESTINATION_CONTAINER): (16, 185, 129),
}


@dataclass(frozen=True)
class EpisodeTensorLabels:
    """All fixed-size V2.2 labels for one episode."""

    episode_index: int
    boxes: np.ndarray
    bins: np.ndarray
    visibility: np.ndarray
    weights: np.ndarray
    entity_ids: np.ndarray
    entity_roles: np.ndarray
    entity_valid: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-root", type=Path, default=DEFAULT_ROBOT_ROOT)
    parser.add_argument("--sam-root", type=Path, default=DEFAULT_SAM_ROOT)
    parser.add_argument("--grounding-root", type=Path, default=DEFAULT_GROUNDING_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["wrist", "top"],
        help="Camera order; it must match the policy image-feature order.",
    )
    parser.add_argument("--target-width", type=int, default=512)
    parser.add_argument("--target-height", type=int, default=512)
    parser.add_argument("--coordinate-bins", type=int, default=256)
    parser.add_argument("--max-entities", type=int, default=8)
    parser.add_argument("--write-batch-size", type=int, default=128)
    parser.add_argument(
        "--video-copy-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How unchanged robot videos are placed in the output dataset.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=12,
        help="Number of evenly distributed episode preview videos; zero disables them.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing .incomplete output after an interrupted build.",
    )
    return parser.parse_args()


def load_info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {root}")
    return json.loads(path.read_text())


def load_episode_metadata(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {root}.")
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    if frame["episode_index"].duplicated().any():
        raise ValueError("Episode metadata contains duplicate episode indices.")
    return frame.set_index("episode_index", drop=False).sort_index()


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


def copy_static_files(
    robot_root: Path,
    output_root: Path,
    *,
    video_copy_mode: str,
) -> None:
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


def prepare_output_features(
    output_root: Path,
    *,
    metadata: ActionFlowV22DatasetMetadata,
) -> datasets.Features:
    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"].update(metadata.feature_metadata())
    info_path.write_text(json.dumps(info, indent=4) + "\n")

    features = info["features"]
    metadata.validate_feature_metadata(features)
    for feature in features.values():
        feature["shape"] = tuple(feature["shape"])
    return get_hf_features_from_features(features)


def source_data_path(robot_root: Path, episode_row: pd.Series) -> Path:
    info = load_info(robot_root)
    relative = info["data_path"].format(
        chunk_index=int(episode_row["data/chunk_index"]),
        file_index=int(episode_row["data/file_index"]),
    )
    return robot_root / relative


def validate_episode_tensors(
    *,
    episode_index: int,
    boxes: np.ndarray,
    bins: np.ndarray,
    visibility: np.ndarray,
    weights: np.ndarray,
    entity_ids: np.ndarray,
    entity_roles: np.ndarray,
    entity_valid: np.ndarray,
    coordinate_bins: int,
) -> None:
    frame_count, camera_count, max_entities, coordinate_count = boxes.shape
    expected_camera_shape = (frame_count, camera_count, max_entities)
    if coordinate_count != 4 or bins.shape != boxes.shape:
        raise ValueError(f"Episode {episode_index} has invalid box tensor shapes.")
    if visibility.shape != expected_camera_shape or weights.shape != expected_camera_shape:
        raise ValueError(f"Episode {episode_index} has invalid camera/entity tensor shapes.")
    if entity_ids.shape != (max_entities,):
        raise ValueError(f"Episode {episode_index} has invalid entity ID shape.")
    if entity_roles.shape != (max_entities,) or entity_valid.shape != (max_entities,):
        raise ValueError(f"Episode {episode_index} has invalid entity metadata shapes.")
    if not np.isfinite(boxes).all() or np.any((boxes < 0) | (boxes > 1)):
        raise ValueError(f"Episode {episode_index} has boxes outside [0, 1].")
    if np.any((bins < 0) | (bins >= coordinate_bins)):
        raise ValueError(f"Episode {episode_index} has invalid coordinate bins.")
    if not set(np.unique(visibility)).issubset(
        {int(TargetVisibility.NOT_VISIBLE), int(TargetVisibility.VISIBLE)}
    ):
        raise ValueError(f"Episode {episode_index} has invalid visibility values.")
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError(f"Episode {episode_index} has invalid loss weights.")

    padded = entity_valid == 0
    if np.any(entity_ids[padded] != -1) or np.any(entity_roles[padded] != 0):
        raise ValueError(f"Episode {episode_index} has nonempty padded entity slots.")
    if np.any(boxes[:, :, padded]) or np.any(bins[:, :, padded]):
        raise ValueError(f"Episode {episode_index} has boxes in padded entity slots.")
    if np.any(weights[:, :, padded]):
        raise ValueError(f"Episode {episode_index} supervises padded entity slots.")
    absent = visibility == int(TargetVisibility.NOT_VISIBLE)
    if np.any(boxes[absent]) or np.any(bins[absent]):
        raise ValueError(f"Episode {episode_index} has coordinates for absent entities.")


def write_episode_data(
    source_path: Path,
    target_path: Path,
    *,
    boxes: np.ndarray,
    bins: np.ndarray,
    visibility: np.ndarray,
    weights: np.ndarray,
    entity_ids: np.ndarray,
    entity_roles: np.ndarray,
    entity_valid: np.ndarray,
    hf_features: datasets.Features,
    write_batch_size: int,
) -> int:
    source = pd.read_parquet(source_path)
    if len(source) != len(boxes):
        raise ValueError(f"{source_path} frame count does not match generated labels.")
    if entity_ids.shape[0] != len(source):
        raise ValueError(f"{source_path} entity metadata does not match its row count.")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for start in range(0, len(source), write_batch_size):
            end = min(start + write_batch_size, len(source))
            columns: dict[str, Any] = source.iloc[start:end].to_dict(orient="list")
            columns[TARGET_BOXES] = boxes[start:end]
            columns[TARGET_BOX_BINS] = bins[start:end]
            columns[TARGET_BOX_VISIBILITY] = visibility[start:end]
            columns[TARGET_BOX_LOSS_WEIGHT] = weights[start:end]
            columns[TARGET_ENTITY_IDS] = entity_ids[start:end]
            columns[TARGET_ENTITY_ROLES] = entity_roles[start:end]
            columns[TARGET_ENTITY_VALID] = entity_valid[start:end]
            dataset = datasets.Dataset.from_dict(
                columns,
                features=hf_features,
                split="train",
            )
            arrow_table = dataset.data.table
            if writer is None:
                writer = pq.ParquetWriter(
                    target_path,
                    arrow_table.schema,
                    compression="snappy",
                    use_dictionary=True,
                )
            writer.write_table(arrow_table)
    finally:
        if writer is not None:
            writer.close()
    return len(source)


def track_review_rows(
    sidecar: pd.DataFrame,
    *,
    prompt: str,
) -> list[dict[str, Any]]:
    rows = []
    for (camera, entity_id), track in sidecar.groupby(["camera", "entity_id"], sort=False):
        visible_frames = int(
            (track["visibility"] == int(TargetVisibility.VISIBLE)).sum()
        )
        invalid_frames = int((track["loss_weight"] == 0).sum())
        explicit_absence = bool(track["explicit_not_visible"].any())
        if invalid_frames:
            status = "INVALID_FRAMES"
        elif visible_frames:
            status = "OK"
        elif explicit_absence:
            status = "NOT_VISIBLE"
        else:
            status = "CHECK_NEVER_VISIBLE"
        first = track.iloc[0]
        rows.append(
            {
                "episode_index": int(first["episode_index"]),
                "camera": camera,
                "entity_id": int(entity_id),
                "entity_name": first["entity_name"],
                "entity_role": int(first["entity_role"]),
                "task_step": int(first["task_step"]),
                "frame_count": len(track),
                "visible_frames": visible_frames,
                "invalid_frames": invalid_frames,
                "status": status,
                "prompt": prompt,
            }
        )
    return rows


def build_episode(
    *,
    sam_root: Path,
    grounding_root: Path,
    output_root: Path,
    episode_row: pd.Series,
    cameras: tuple[str, ...],
    entity_name_to_id: dict[str, int],
    max_entities: int,
    target_width: int,
    target_height: int,
    coordinate_bins: int,
    review_decisions: dict[tuple[int, str, int], str],
) -> tuple[EpisodeTensorLabels, list[dict[str, Any]]]:
    episode_index = int(episode_row["episode_index"])
    frame_count = int(episode_row["length"])
    prompt, definitions, reviewed_not_visible = load_episode_definitions(
        grounding_root,
        episode_index=episode_index,
        cameras=cameras,
        entity_name_to_id=entity_name_to_id,
        review_decisions=review_decisions,
    )
    if len(definitions) > max_entities:
        raise ValueError(
            f"Episode {episode_index} uses {len(definitions)} entities; "
            f"`max_entities` is {max_entities}."
        )

    boxes = np.zeros(
        (frame_count, len(cameras), max_entities, 4),
        dtype=np.float32,
    )
    bins = np.zeros_like(boxes, dtype=np.int64)
    visibility = np.zeros(
        (frame_count, len(cameras), max_entities),
        dtype=np.uint8,
    )
    weights = np.zeros_like(visibility, dtype=np.float32)
    sidecar_rows: list[dict[str, Any]] = []
    for camera_index, camera in enumerate(cameras):
        (
            camera_boxes,
            camera_bins,
            camera_visibility,
            camera_weights,
            camera_sidecar,
        ) = build_camera_labels(
            sam_root,
            episode_index=episode_index,
            camera=camera,
            definitions=definitions,
            frame_count=frame_count,
            target_width=target_width,
            target_height=target_height,
            coordinate_bins=coordinate_bins,
            reviewed_not_visible=reviewed_not_visible[camera],
        )
        entity_count = len(definitions)
        boxes[:, camera_index, :entity_count] = camera_boxes
        bins[:, camera_index, :entity_count] = camera_bins
        visibility[:, camera_index, :entity_count] = camera_visibility
        weights[:, camera_index, :entity_count] = camera_weights
        sidecar_rows.extend(camera_sidecar)

    entity_ids = np.full(max_entities, -1, dtype=np.int64)
    entity_roles = np.zeros(max_entities, dtype=np.int64)
    entity_valid = np.zeros(max_entities, dtype=np.uint8)
    for slot, entity in enumerate(definitions):
        entity_ids[slot] = entity.entity_id
        entity_roles[slot] = int(entity.role)
        entity_valid[slot] = 1

    validate_episode_tensors(
        episode_index=episode_index,
        boxes=boxes,
        bins=bins,
        visibility=visibility,
        weights=weights,
        entity_ids=entity_ids,
        entity_roles=entity_roles,
        entity_valid=entity_valid,
        coordinate_bins=coordinate_bins,
    )

    sidecar = pd.DataFrame(sidecar_rows)
    for camera in cameras:
        camera_path = (
            output_root
            / "annotations"
            / "boxes"
            / f"episode-{episode_index:06d}"
            / f"camera-{camera}.parquet"
        )
        camera_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar.loc[sidecar["camera"] == camera].to_parquet(camera_path, index=False)

    labels = EpisodeTensorLabels(
        episode_index=episode_index,
        boxes=boxes,
        bins=bins,
        visibility=visibility,
        weights=weights,
        entity_ids=entity_ids,
        entity_roles=entity_roles,
        entity_valid=entity_valid,
    )
    return labels, track_review_rows(sidecar, prompt=prompt)


def write_episode_group(
    *,
    robot_root: Path,
    output_root: Path,
    episode_rows: pd.DataFrame,
    labels_by_episode: dict[int, EpisodeTensorLabels],
    hf_features: datasets.Features,
    write_batch_size: int,
) -> int:
    """Write one source data file, which may contain multiple episodes."""

    first_row = episode_rows.iloc[0]
    source_path = source_data_path(robot_root, first_row)
    expected_path = {
        source_data_path(robot_root, row)
        for _, row in episode_rows.iterrows()
    }
    if expected_path != {source_path}:
        raise ValueError("One data-file group resolved to multiple source paths.")

    source = pd.read_parquet(source_path, columns=["episode_index", "frame_index"])
    source_episodes = {int(value) for value in source["episode_index"].unique()}
    if source_episodes != set(labels_by_episode):
        raise ValueError(
            f"{source_path} contains episodes {sorted(source_episodes)}, but labels "
            f"were built for {sorted(labels_by_episode)}."
        )

    first_labels = next(iter(labels_by_episode.values()))
    row_count = len(source)
    boxes = np.zeros((row_count, *first_labels.boxes.shape[1:]), dtype=np.float32)
    bins = np.zeros_like(boxes, dtype=np.int64)
    visibility = np.zeros(
        (row_count, *first_labels.visibility.shape[1:]),
        dtype=np.uint8,
    )
    weights = np.zeros_like(visibility, dtype=np.float32)
    entity_ids = np.full(
        (row_count, first_labels.entity_ids.shape[0]),
        -1,
        dtype=np.int64,
    )
    entity_roles = np.zeros_like(entity_ids)
    entity_valid = np.zeros_like(entity_ids, dtype=np.uint8)

    for episode_index, labels in labels_by_episode.items():
        positions = np.flatnonzero(
            source["episode_index"].to_numpy(dtype=np.int64) == episode_index
        )
        frame_indices = source.iloc[positions]["frame_index"].to_numpy(dtype=np.int64)
        if len(positions) != len(labels.boxes):
            raise ValueError(
                f"{source_path} has {len(positions)} rows for episode {episode_index}; "
                f"labels contain {len(labels.boxes)} frames."
            )
        if not np.array_equal(np.sort(frame_indices), np.arange(len(labels.boxes))):
            raise ValueError(
                f"{source_path} has incomplete frame indices for episode {episode_index}."
            )
        boxes[positions] = labels.boxes[frame_indices]
        bins[positions] = labels.bins[frame_indices]
        visibility[positions] = labels.visibility[frame_indices]
        weights[positions] = labels.weights[frame_indices]
        entity_ids[positions] = labels.entity_ids
        entity_roles[positions] = labels.entity_roles
        entity_valid[positions] = labels.entity_valid

    relative_path = source_path.relative_to(robot_root)
    return write_episode_data(
        source_path,
        output_root / relative_path,
        boxes=boxes,
        bins=bins,
        visibility=visibility,
        weights=weights,
        entity_ids=entity_ids,
        entity_roles=entity_roles,
        entity_valid=entity_valid,
        hf_features=hf_features,
        write_batch_size=write_batch_size,
    )


def iter_episode_video(
    robot_root: Path,
    info: dict[str, Any],
    episode_row: pd.Series,
    *,
    camera: str,
) -> Iterator[np.ndarray]:
    prefix = f"videos/observation.images.{camera}"
    video_path = robot_root / info["video_path"].format(
        video_key=f"observation.images.{camera}",
        chunk_index=int(episode_row[f"{prefix}/chunk_index"]),
        file_index=int(episode_row[f"{prefix}/file_index"]),
    )
    start_s = float(episode_row[f"{prefix}/from_timestamp"])
    frame_count = int(episode_row["length"])
    fps = int(info["fps"])
    yielded = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if start_s > 0:
            seek_pts = max(0, int((start_s - 1.0 / fps) / float(stream.time_base)))
            container.seek(seek_pts, stream=stream, backward=True)
        tolerance_s = 0.5 / fps
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp_s = float(frame.pts * stream.time_base)
            if timestamp_s < start_s - tolerance_s:
                continue
            yield frame.to_ndarray(format="rgb24")
            yielded += 1
            if yielded == frame_count:
                break
    if yielded != frame_count:
        raise ValueError(
            f"Decoded {yielded} frames for episode {int(episode_row['episode_index'])} "
            f"camera {camera}; expected {frame_count}."
        )


def render_camera_frame(
    frame: np.ndarray,
    *,
    camera: str,
    frame_index: int,
    boxes: np.ndarray,
    visibility: np.ndarray,
    weights: np.ndarray,
    entity_names: list[str],
    entity_roles: np.ndarray,
    target_width: int,
    target_height: int,
) -> Image.Image:
    geometry = compute_resize_geometry(
        frame.shape[1],
        frame.shape[0],
        target_width,
        target_height,
    )
    source = Image.fromarray(frame)
    resized = source.resize(
        (geometry.resized_width, geometry.resized_height),
        Image.Resampling.BILINEAR,
    )
    image = Image.new("RGB", (target_width, target_height))
    image.paste(resized, (geometry.pad_left, geometry.pad_top))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, target_width - 1, 21), fill=(10, 10, 10))
    draw.text((6, 5), f"{camera} | frame {frame_index}", fill=(255, 255, 255))

    hidden = []
    for slot, name in enumerate(entity_names):
        if weights[slot] == 0:
            hidden.append(f"{name}:invalid")
            continue
        if visibility[slot] != int(TargetVisibility.VISIBLE):
            hidden.append(f"{name}:not visible")
            continue
        x1, y1, x2, y2 = boxes[slot]
        xyxy = (
            int(round(x1 * (target_width - 1))),
            int(round(y1 * (target_height - 1))),
            int(round(x2 * (target_width - 1))),
            int(round(y2 * (target_height - 1))),
        )
        color = ROLE_COLORS.get(int(entity_roles[slot]), (255, 255, 255))
        draw.rectangle(xyxy, outline=color, width=3)
        label_y = max(23, xyxy[1] - 14)
        draw.text((xyxy[0] + 2, label_y), name, fill=color)
    if hidden:
        draw.rectangle(
            (0, target_height - 20, target_width - 1, target_height - 1),
            fill=(10, 10, 10),
        )
        draw.text(
            (5, target_height - 16),
            " | ".join(hidden)[:95],
            fill=(220, 220, 220),
        )
    return image


def create_preview_video(
    *,
    robot_root: Path,
    output_root: Path,
    info: dict[str, Any],
    episode_row: pd.Series,
    cameras: tuple[str, ...],
    entity_vocabulary: tuple[str, ...],
    target_width: int,
    target_height: int,
) -> Path:
    episode_index = int(episode_row["episode_index"])
    data_path = output_root / source_data_path(robot_root, episode_row).relative_to(robot_root)
    label_table = pq.read_table(
        data_path,
        columns=[
            "episode_index",
            "frame_index",
            TARGET_BOXES,
            TARGET_BOX_VISIBILITY,
            TARGET_BOX_LOSS_WEIGHT,
            TARGET_ENTITY_IDS,
            TARGET_ENTITY_ROLES,
            TARGET_ENTITY_VALID,
        ],
    )
    label_table = label_table.filter(
        pc.equal(label_table["episode_index"], episode_index)
    ).sort_by("frame_index")
    if label_table.num_rows != int(episode_row["length"]):
        raise ValueError(
            f"Preview labels for episode {episode_index} have {label_table.num_rows} rows; "
            f"expected {int(episode_row['length'])}."
        )
    entity_valid = np.asarray(
        label_table[TARGET_ENTITY_VALID][0].as_py(),
        dtype=np.uint8,
    )
    entity_ids = np.asarray(
        label_table[TARGET_ENTITY_IDS][0].as_py(),
        dtype=np.int64,
    )
    entity_roles = np.asarray(
        label_table[TARGET_ENTITY_ROLES][0].as_py(),
        dtype=np.int64,
    )
    valid_slots = np.flatnonzero(entity_valid)
    entity_names = [entity_vocabulary[entity_ids[slot]] for slot in valid_slots]

    output_path = output_root / "review" / "videos" / f"episode-{episode_index:06d}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    streams = [
        iter_episode_video(robot_root, info, episode_row, camera=camera)
        for camera in cameras
    ]
    output_width = target_width * len(cameras)
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(info["fps"]))
        stream.width = output_width
        stream.height = target_height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "24", "preset": "veryfast"}
        for frame_index, camera_frames in enumerate(zip(*streams, strict=True)):
            boxes = np.asarray(
                label_table[TARGET_BOXES][frame_index].as_py(),
                dtype=np.float32,
            )
            visibility = np.asarray(
                label_table[TARGET_BOX_VISIBILITY][frame_index].as_py(),
                dtype=np.uint8,
            )
            weights = np.asarray(
                label_table[TARGET_BOX_LOSS_WEIGHT][frame_index].as_py(),
                dtype=np.float32,
            )
            rendered = [
                render_camera_frame(
                    frame,
                    camera=camera,
                    frame_index=frame_index,
                    boxes=boxes[camera_index, valid_slots],
                    visibility=visibility[camera_index, valid_slots],
                    weights=weights[camera_index, valid_slots],
                    entity_names=entity_names,
                    entity_roles=entity_roles[valid_slots],
                    target_width=target_width,
                    target_height=target_height,
                )
                for camera_index, (camera, frame) in enumerate(
                    zip(cameras, camera_frames, strict=True)
                )
            ]
            combined = Image.new("RGB", (output_width, target_height))
            for camera_index, image in enumerate(rendered):
                combined.paste(image, (camera_index * target_width, 0))
            video_frame = av.VideoFrame.from_ndarray(
                np.asarray(combined),
                format="rgb24",
            )
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output_path


def write_review_report(
    output_root: Path,
    *,
    review: pd.DataFrame,
    preview_episodes: list[int],
) -> None:
    review_root = output_root / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    review.to_parquet(review_root / "tracks.parquet", index=False)
    status_counts = {
        str(key): int(value)
        for key, value in review["status"].value_counts().sort_index().items()
    }
    summary = {
        "track_count": len(review),
        "status_counts": status_counts,
        "preview_episodes": preview_episodes,
        "review_rule": (
            "CHECK_NEVER_VISIBLE and INVALID_FRAMES require inspection; "
            "NOT_VISIBLE is an intentional supervised absence."
        ),
    }
    (review_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = []
    for row in review.itertuples(index=False):
        episode = int(row.episode_index)
        video = (
            f'<a href="videos/episode-{episode:06d}.mp4">video</a>'
            if episode in preview_episodes
            else ""
        )
        rows.append(
            "<tr>"
            f"<td>{episode}</td><td>{html.escape(row.camera)}</td>"
            f"<td>{html.escape(row.entity_name)}</td><td>{row.entity_role}</td>"
            f"<td>{row.visible_frames}/{row.frame_count}</td>"
            f"<td>{row.invalid_frames}</td><td>{html.escape(row.status)}</td>"
            f"<td>{video}</td><td>{html.escape(row.prompt)}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Action Flow V2.2 box review</title>
<style>
body {{ font-family: sans-serif; margin: 24px; color: #202124; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #dadce0; padding: 6px; text-align: left; }}
th {{ position: sticky; top: 0; background: #f1f3f4; }}
tr:nth-child(even) {{ background: #fafafa; }}
code {{ background: #f1f3f4; padding: 2px 4px; }}
</style></head><body>
<h1>Action Flow V2.2 Box Review</h1>
<p>Status counts: <code>{html.escape(json.dumps(status_counts))}</code>.
Inspect <code>CHECK_NEVER_VISIBLE</code> and <code>INVALID_FRAMES</code>.</p>
<table><thead><tr><th>Episode</th><th>Camera</th><th>Entity</th><th>Role ID</th>
<th>Visible frames</th><th>Invalid frames</th><th>Status</th><th>Preview</th>
<th>Prompt</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>
"""
    (review_root / "index.html").write_text(document)


def choose_preview_episodes(
    episode_indices: list[int],
    review: pd.DataFrame,
    *,
    count: int,
) -> list[int]:
    if count <= 0:
        return []
    suspicious = sorted(
        int(value)
        for value in review.loc[
            review["status"].isin(["CHECK_NEVER_VISIBLE", "INVALID_FRAMES"]),
            "episode_index",
        ].unique()
    )
    chosen = suspicious[:count]
    remaining = count - len(chosen)
    if remaining > 0:
        positions = np.linspace(0, len(episode_indices) - 1, remaining, dtype=int)
        for position in positions:
            episode = episode_indices[int(position)]
            if episode not in chosen:
                chosen.append(episode)
    return sorted(chosen[:count])


def validate_output(
    output_root: Path,
    *,
    metadata: ActionFlowV22DatasetMetadata,
    expected_frames: int,
    review: pd.DataFrame,
) -> dict[str, Any]:
    info = load_info(output_root)
    metadata.validate_feature_metadata(info["features"])
    data_files = sorted((output_root / "data").glob("chunk-*/*.parquet"))
    actual_frames = sum(pq.ParquetFile(path).metadata.num_rows for path in data_files)
    if actual_frames != expected_frames:
        raise ValueError(f"Output has {actual_frames} frames; expected {expected_frames}.")
    if not data_files:
        raise ValueError("Output dataset has no data files.")
    sample = pd.read_parquet(
        data_files[0],
        columns=[
            TARGET_BOXES,
            TARGET_BOX_BINS,
            TARGET_BOX_VISIBILITY,
            TARGET_BOX_LOSS_WEIGHT,
            TARGET_ENTITY_IDS,
            TARGET_ENTITY_ROLES,
            TARGET_ENTITY_VALID,
        ],
    )
    if sample.empty:
        raise ValueError("First output data file is empty.")

    result = {
        "valid": True,
        "total_episodes": int(info["total_episodes"]),
        "total_frames": actual_frames,
        "camera_order": list(metadata.camera_names),
        "entity_vocabulary": list(metadata.entity_names),
        "max_entities_per_frame": metadata.max_entities,
        "coordinate_bins": metadata.coordinate_bins,
        "track_status_counts": {
            str(key): int(value)
            for key, value in review["status"].value_counts().sort_index().items()
        },
    }
    path = output_root / "meta" / "action_flow_v2_2_validation.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def output_data_is_complete(
    robot_root: Path,
    output_root: Path,
    episodes: pd.DataFrame,
) -> bool:
    """Return whether every source data file has a complete labeled counterpart."""

    required_columns = {
        TARGET_BOXES,
        TARGET_BOX_BINS,
        TARGET_BOX_VISIBILITY,
        TARGET_BOX_LOSS_WEIGHT,
        TARGET_ENTITY_IDS,
        TARGET_ENTITY_ROLES,
        TARGET_ENTITY_VALID,
    }
    for _, rows in episodes.groupby(["data/chunk_index", "data/file_index"], sort=False):
        source_path = source_data_path(robot_root, rows.iloc[0])
        target_path = output_root / source_path.relative_to(robot_root)
        if not target_path.is_file():
            return False
        source_rows = pq.ParquetFile(source_path).metadata.num_rows
        target_file = pq.ParquetFile(target_path)
        if target_file.metadata.num_rows != source_rows:
            return False
        if not required_columns.issubset(target_file.schema_arrow.names):
            return False
    return True


def load_existing_review(
    output_root: Path,
    grounding_root: Path,
    *,
    episode_indices: list[int],
    cameras: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Rebuild track-level review rows from saved long-form sidecars."""

    review_rows: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        sidecars = []
        for camera in cameras:
            path = (
                output_root
                / "annotations"
                / "boxes"
                / f"episode-{episode_index:06d}"
                / f"camera-{camera}.parquet"
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing box sidecar during resume: {path}")
            sidecars.append(pd.read_parquet(path))
        grounding_path = (
            grounding_root
            / "json"
            / f"episode-{episode_index:06d}"
            / f"camera-{cameras[0]}.json"
        )
        selected = json.loads(grounding_path.read_text()).get("selected", [])
        if not selected:
            raise ValueError(f"No reviewed entities in {grounding_path}.")
        review_rows.extend(
            track_review_rows(
                pd.concat(sidecars, ignore_index=True),
                prompt=str(selected[0]["task"]),
            )
        )
    return review_rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    cameras = tuple(args.cameras)
    if len(set(cameras)) != len(cameras):
        raise ValueError("Camera names must be unique.")
    if min(args.target_width, args.target_height, args.write_batch_size) <= 0:
        raise ValueError("Target dimensions and write batch size must be positive.")
    if args.coordinate_bins < 2 or args.max_entities <= 0:
        raise ValueError("Coordinate bins and max entities must be valid.")
    if args.output_root.exists():
        raise FileExistsError(f"Output dataset already exists: {args.output_root}")

    incomplete_root = args.output_root.with_name(args.output_root.name + ".incomplete")
    if incomplete_root.exists() and not args.resume:
        raise FileExistsError(
            f"Incomplete output exists: {incomplete_root}. Use `--resume` after inspecting it."
        )

    robot_info = load_info(args.robot_root)
    episodes = load_episode_metadata(args.robot_root)
    episode_indices = [int(value) for value in episodes.index]
    if episode_indices != list(range(len(episode_indices))):
        raise ValueError("Stage 2 expects contiguous zero-based episode indices.")
    review_decisions = load_review_visibility_decisions(args.grounding_root)
    entity_vocabulary = discover_entity_vocabulary(
        args.grounding_root,
        episode_indices=episode_indices,
        cameras=cameras,
    )
    entity_name_to_id = {
        name: entity_id for entity_id, name in enumerate(entity_vocabulary)
    }
    metadata = ActionFlowV22DatasetMetadata(
        camera_names=cameras,
        entity_names=entity_vocabulary,
        max_entities=args.max_entities,
        coordinate_bins=args.coordinate_bins,
    )

    if incomplete_root.exists():
        logging.info("Resuming output dataset at %s", incomplete_root)
        if not (incomplete_root / "meta" / "info.json").is_file():
            raise ValueError(f"Incomplete output has no metadata: {incomplete_root}")
    else:
        logging.info("Preparing output dataset at %s", incomplete_root)
        incomplete_root.mkdir(parents=True)
        copy_static_files(
            args.robot_root,
            incomplete_root,
            video_copy_mode=args.video_copy_mode,
        )
    hf_features = prepare_output_features(incomplete_root, metadata=metadata)
    metadata.save(incomplete_root / "meta" / "action_flow_v2_2_boxes.json")
    provenance = {
        "format": "action_flow_v2_2_entity_boxes",
        "schema_version": metadata.schema_version,
        "robot_dataset_root": str(args.robot_root.resolve()),
        "sam_track_root": str(args.sam_root.resolve()),
        "reviewed_grounding_root": str(args.grounding_root.resolve()),
        "review_decisions": str(
            (args.grounding_root / "review" / "decisions.json").resolve()
        ),
        "camera_order": list(cameras),
        "policy_image_size": [args.target_height, args.target_width],
        "coordinate_bins": args.coordinate_bins,
        "box_source": "bounding box of each reviewed entity-specific SAM2 mask",
        "video_copy_mode": args.video_copy_mode,
    }
    (incomplete_root / "meta" / "action_flow_v2_2.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    total_frames = 0
    review_rows: list[dict[str, Any]] = []
    if args.resume and output_data_is_complete(
        args.robot_root,
        incomplete_root,
        episodes,
    ):
        logging.info("All labeled data files are complete; resuming at review generation.")
        total_frames = int(robot_info["total_frames"])
        review_rows = load_existing_review(
            incomplete_root,
            args.grounding_root,
            episode_indices=episode_indices,
            cameras=cameras,
        )
    else:
        data_file_groups = list(
            episodes.groupby(["data/chunk_index", "data/file_index"], sort=False)
        )
        episode_number = 0
        for file_number, (_, episode_rows) in enumerate(data_file_groups, start=1):
            labels_by_episode: dict[int, EpisodeTensorLabels] = {}
            for episode_index in episode_rows["episode_index"].astype(int):
                labels, episode_review = build_episode(
                    sam_root=args.sam_root,
                    grounding_root=args.grounding_root,
                    output_root=incomplete_root,
                    episode_row=episodes.loc[episode_index],
                    cameras=cameras,
                    entity_name_to_id=entity_name_to_id,
                    max_entities=args.max_entities,
                    target_width=args.target_width,
                    target_height=args.target_height,
                    coordinate_bins=args.coordinate_bins,
                    review_decisions=review_decisions,
                )
                labels_by_episode[episode_index] = labels
                total_frames += len(labels.boxes)
                review_rows.extend(episode_review)
                episode_number += 1
                logging.info(
                    "[%d/%d] Converted episode %d (%d frames).",
                    episode_number,
                    len(episode_indices),
                    episode_index,
                    len(labels.boxes),
                )
            written_frames = write_episode_group(
                robot_root=args.robot_root,
                output_root=incomplete_root,
                episode_rows=episode_rows,
                labels_by_episode=labels_by_episode,
                hf_features=hf_features,
                write_batch_size=args.write_batch_size,
            )
            logging.info(
                "[%d/%d] Wrote source data file containing %d episodes and %d frames.",
                file_number,
                len(data_file_groups),
                len(labels_by_episode),
                written_frames,
            )

    if total_frames != int(robot_info["total_frames"]):
        raise ValueError(
            f"Built {total_frames} frames; source metadata expects "
            f"{robot_info['total_frames']}."
        )
    review = pd.DataFrame(review_rows)
    preview_episodes = choose_preview_episodes(
        episode_indices,
        review,
        count=args.preview_count,
    )
    for number, episode_index in enumerate(preview_episodes, start=1):
        path = create_preview_video(
            robot_root=args.robot_root,
            output_root=incomplete_root,
            info=robot_info,
            episode_row=episodes.loc[episode_index],
            cameras=cameras,
            entity_vocabulary=entity_vocabulary,
            target_width=args.target_width,
            target_height=args.target_height,
        )
        logging.info(
            "[%d/%d] Wrote preview %s.",
            number,
            len(preview_episodes),
            path,
        )
    write_review_report(
        incomplete_root,
        review=review,
        preview_episodes=preview_episodes,
    )
    validation = validate_output(
        incomplete_root,
        metadata=metadata,
        expected_frames=int(robot_info["total_frames"]),
        review=review,
    )
    incomplete_root.rename(args.output_root)
    logging.info(
        "Created %s: %d episodes, %d frames, %d entities, status=%s.",
        args.output_root,
        validation["total_episodes"],
        validation["total_frames"],
        len(entity_vocabulary),
        validation["track_status_counts"],
    )


if __name__ == "__main__":
    main()
