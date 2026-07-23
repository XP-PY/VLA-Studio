#!/usr/bin/env python

"""Merge frame-aligned segmentation labels into a standard LeRobot dataset.

The source robot dataset remains unchanged. The output dataset copies its
metadata and videos, then adds two model-ready columns to every data row:

* ``observation.relevance_mask``: ``uint8[num_cameras, height, width]``
* ``observation.relevance_loss_weight``: ``float32[num_cameras]``

Raw full-resolution SAM2 masks remain in the segmentation source dataset. The
merged masks are resized with the same aspect-ratio-preserving, left/top
padding convention used by SmolVLA image preprocessing.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import zlib
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.datasets.feature_utils import get_hf_features_from_features

RELEVANCE_MASK = "observation.relevance_mask"
RELEVANCE_LOSS_WEIGHT = "observation.relevance_loss_weight"

DEFAULT_ROBOT_ROOT = Path("runs/dataset/rm65_single_table_preparation_and_bussing")
DEFAULT_SEGMENTATION_ROOT = Path(
    "runs/dataset/rm65_single_table_preparation_and_bussing_afv2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "runs/dataset/rm65_single_table_preparation_and_bussing_afv2_segmentation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-root", type=Path, default=DEFAULT_ROBOT_ROOT)
    parser.add_argument("--segmentation-root", type=Path, default=DEFAULT_SEGMENTATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["wrist", "top"],
        help="Segmentation cameras in the same order as the policy image features.",
    )
    parser.add_argument("--target-width", type=int, default=8)
    parser.add_argument("--target-height", type=int, default=8)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    return parser.parse_args()


def decode_mask_runs(payload: bytes, height: int, width: int) -> np.ndarray:
    """Decode one zlib-compressed uint32 ``(start, length)`` mask."""

    mask = np.zeros(height * width, dtype=np.uint8)
    raw_runs = zlib.decompress(bytes(payload))
    if raw_runs:
        runs = np.frombuffer(raw_runs, dtype=np.uint32).reshape(-1, 2)
        for start, length in runs:
            mask[int(start) : int(start + length)] = 1
    return mask.reshape(height, width)


def resize_masks_with_pad(
    masks: np.ndarray,
    *,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Resize a mask batch using SmolVLA-compatible max pooling and padding."""

    if masks.ndim != 3:
        raise ValueError(f"Expected masks shaped (B, H, W), got {masks.shape}.")

    current_height, current_width = masks.shape[-2:]
    ratio = max(current_width / target_width, current_height / target_height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    tensor = torch.from_numpy(masks)[:, None].float()
    if resized_height <= current_height and resized_width <= current_width:
        resized = F.adaptive_max_pool2d(
            tensor,
            output_size=(resized_height, resized_width),
        )
    else:
        resized = F.interpolate(
            tensor,
            size=(resized_height, resized_width),
            mode="nearest",
        )

    pad_height = max(0, target_height - resized_height)
    pad_width = max(0, target_width - resized_width)
    resized = F.pad(resized, (pad_width, 0, pad_height, 0), value=0)
    return resized[:, 0].to(dtype=torch.uint8).numpy()


def decode_and_resize_table(
    table: pd.DataFrame,
    *,
    target_width: int,
    target_height: int,
    batch_size: int,
) -> np.ndarray:
    """Decode and resize all masks from one episode-camera table."""

    output = np.zeros((len(table), target_height, target_width), dtype=np.uint8)
    for start in range(0, len(table), batch_size):
        rows = table.iloc[start : start + batch_size]
        shapes = {(int(row.height), int(row.width)) for row in rows.itertuples()}
        if len(shapes) != 1:
            raise ValueError(f"One annotation batch contains mixed mask shapes: {sorted(shapes)}")
        source_height, source_width = shapes.pop()
        masks = np.stack(
            [
                decode_mask_runs(row.mask_runs, source_height, source_width)
                for row in rows.itertuples()
            ]
        )
        output[start : start + len(rows)] = resize_masks_with_pad(
            masks,
            target_width=target_width,
            target_height=target_height,
        )
    return output


def load_episode_camera_labels(
    segmentation_root: Path,
    *,
    episode_index: int,
    camera: str,
    expected_frame_indices: np.ndarray,
    target_width: int,
    target_height: int,
    decode_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = (
        segmentation_root
        / "data"
        / f"episode-{episode_index:06d}"
        / f"camera-{camera}.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing segmentation labels: {path}")

    table = pd.read_parquet(path)
    required = {
        "episode_index",
        "frame_index",
        "camera",
        "height",
        "width",
        "mask_runs",
        "loss_weight",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if not np.all(table["episode_index"].to_numpy() == episode_index):
        raise ValueError(f"{path} contains a different episode index.")
    if not np.all(table["camera"].to_numpy() == camera):
        raise ValueError(f"{path} contains a different camera name.")
    if not np.array_equal(table["frame_index"].to_numpy(), expected_frame_indices):
        raise ValueError(f"{path} frame indices do not match the robot dataset.")

    masks = decode_and_resize_table(
        table,
        target_width=target_width,
        target_height=target_height,
        batch_size=decode_batch_size,
    )
    weights = table["loss_weight"].to_numpy(dtype=np.float32)
    return masks, weights


def build_augmented_file(
    source_path: Path,
    target_path: Path,
    *,
    segmentation_root: Path,
    cameras: list[str],
    target_width: int,
    target_height: int,
    decode_batch_size: int,
    hf_features: datasets.Features,
) -> int:
    frame = pd.read_parquet(source_path)
    masks = np.zeros(
        (len(frame), len(cameras), target_height, target_width),
        dtype=np.uint8,
    )
    weights = np.zeros((len(frame), len(cameras)), dtype=np.float32)

    for episode_index, episode_rows in frame.groupby("episode_index", sort=False):
        row_positions = frame.index.get_indexer(episode_rows.index)
        expected_frames = episode_rows["frame_index"].to_numpy()
        for camera_index, camera in enumerate(cameras):
            camera_masks, camera_weights = load_episode_camera_labels(
                segmentation_root,
                episode_index=int(episode_index),
                camera=camera,
                expected_frame_indices=expected_frames,
                target_width=target_width,
                target_height=target_height,
                decode_batch_size=decode_batch_size,
            )
            masks[row_positions, camera_index] = camera_masks
            weights[row_positions, camera_index] = camera_weights

    columns: dict[str, Any] = frame.to_dict(orient="list")
    columns[RELEVANCE_MASK] = masks
    columns[RELEVANCE_LOSS_WEIGHT] = weights
    dataset = datasets.Dataset.from_dict(columns, features=hf_features, split="train")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        dataset.data.table,
        target_path,
        compression="snappy",
        use_dictionary=True,
    )
    return len(frame)


def copy_static_dataset_files(robot_root: Path, output_root: Path) -> None:
    """Copy videos and metadata that are unchanged by label merging."""

    shutil.copytree(robot_root / "videos", output_root / "videos", copy_function=shutil.copy2)
    shutil.copytree(robot_root / "meta", output_root / "meta", copy_function=shutil.copy2)


def write_output_metadata(
    output_root: Path,
    *,
    robot_root: Path,
    segmentation_root: Path,
    cameras: list[str],
    target_width: int,
    target_height: int,
) -> dict[str, dict]:
    info_path = output_root / "meta" / "info.json"
    with info_path.open() as file:
        info = json.load(file)

    features = info["features"]
    features[RELEVANCE_MASK] = {
        "dtype": "uint8",
        "shape": [len(cameras), target_height, target_width],
        "names": ["camera", "height", "width"],
    }
    features[RELEVANCE_LOSS_WEIGHT] = {
        "dtype": "float32",
        "shape": [len(cameras)],
        "names": cameras,
    }
    with info_path.open("w") as file:
        json.dump(info, file, indent=4)
        file.write("\n")

    provenance = {
        "format": "action_flow_v2_relevance_segmentation",
        "robot_dataset_root": str(robot_root.resolve()),
        "segmentation_dataset_root": str(segmentation_root.resolve()),
        "camera_order": cameras,
        "target_size": [target_height, target_width],
        "mask_feature": RELEVANCE_MASK,
        "loss_weight_feature": RELEVANCE_LOSS_WEIGHT,
        "raw_masks_preserved_in_segmentation_dataset": True,
    }
    with (output_root / "meta" / "segmentation.json").open("w") as file:
        json.dump(provenance, file, indent=4)
        file.write("\n")

    # get_hf_features_from_features expects the in-memory DatasetInfo form,
    # where JSON shape lists have already been converted to tuples.
    for feature in features.values():
        feature["shape"] = tuple(feature["shape"])
    return features


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.target_width <= 0 or args.target_height <= 0:
        raise ValueError("Target width and height must be positive.")
    if args.decode_batch_size <= 0:
        raise ValueError("Decode batch size must be positive.")
    if len(set(args.cameras)) != len(args.cameras):
        raise ValueError("Camera names must be unique.")
    if not (args.robot_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {args.robot_root}")
    if not (args.segmentation_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a segmentation dataset: {args.segmentation_root}")
    if args.output_root.exists():
        raise FileExistsError(f"Output dataset already exists: {args.output_root}")

    temporary_root = args.output_root.with_name(f"{args.output_root.name}.incomplete")
    if temporary_root.exists():
        raise FileExistsError(
            f"Incomplete output already exists: {temporary_root}. Inspect or remove it before retrying."
        )

    logging.info("Copying robot metadata and videos to %s", temporary_root)
    temporary_root.mkdir(parents=True)
    copy_static_dataset_files(args.robot_root, temporary_root)
    features = write_output_metadata(
        temporary_root,
        robot_root=args.robot_root,
        segmentation_root=args.segmentation_root,
        cameras=args.cameras,
        target_width=args.target_width,
        target_height=args.target_height,
    )
    hf_features = get_hf_features_from_features(features)

    source_files = sorted((args.robot_root / "data").glob("chunk-*/*.parquet"))
    if not source_files:
        raise FileNotFoundError(f"No robot data Parquet files found under {args.robot_root}")

    total_frames = 0
    for file_number, source_path in enumerate(source_files, start=1):
        relative_path = source_path.relative_to(args.robot_root)
        target_path = temporary_root / relative_path
        total_frames += build_augmented_file(
            source_path,
            target_path,
            segmentation_root=args.segmentation_root,
            cameras=args.cameras,
            target_width=args.target_width,
            target_height=args.target_height,
            decode_batch_size=args.decode_batch_size,
            hf_features=hf_features,
        )
        logging.info(
            "[%d/%d] Merged %s",
            file_number,
            len(source_files),
            relative_path,
        )

    with (temporary_root / "meta" / "info.json").open() as file:
        expected_frames = int(json.load(file)["total_frames"])
    if total_frames != expected_frames:
        raise RuntimeError(f"Merged {total_frames} frames, expected {expected_frames}.")

    temporary_root.rename(args.output_root)
    logging.info(
        "Created %s with %d frames and camera order %s.",
        args.output_root,
        total_frames,
        args.cameras,
    )


if __name__ == "__main__":
    main()
