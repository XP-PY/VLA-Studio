#!/usr/bin/env python3
"""Generate stage-specific RM65 keyframe boxes with INSID3.

The reviewed planning sidecar supplies one active entity per stage: the source
for ``grasp`` and the destination for ``put``. One keyframe is sampled from
each manipulation segment and processed independently for the top and wrist
cameras. ``complete`` segments do not produce box items.

Outputs are resumable. Each prediction is saved immediately as JSON, a binary
mask, and an overlay. A consolidated ``data/boxes.parquet`` is rebuilt after
every run. The source LeRobot and planning datasets are never modified.

Clear-plate grasp plans currently contain relational labels such as
``vegetable on the pink plate`` instead of a visual class. These rows are saved
with ``status=unresolved_reference`` for later manual annotation rather than
being assigned an unreliable generic vegetable box.
"""

from __future__ import annotations

import argparse
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

from insid3_compat import DEFAULT_INSID3_ROOT, DEFAULT_TIMM_MODEL, build_insid3_with_timm


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = WORKSPACE / "runs/dataset/rm65_single_table_preparation_and_bussing"
DEFAULT_PLANNING_ROOT = WORKSPACE / "runs/planning/rm65_single_table_preparation_and_bussing"
DEFAULT_REFERENCE_MANIFEST = (
    WORKSPACE / "artifacts/insid3/references/rm65_tabletop/references.json"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE / "runs/insid3/rm65_single_table_preparation_and_bussing"
logger = logging.getLogger(__name__)


def parse_episode_selection(value: str, available: list[int]) -> list[int]:
    if value.strip().lower() == "all":
        return available
    selected: set[int] = set()
    available_set = set(available)
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            start_text, stop_text = token.split(":", maxsplit=1)
            start = int(start_text) if start_text else min(available)
            stop = int(stop_text) if stop_text else max(available) + 1
            selected.update(range(start, stop))
        else:
            selected.add(int(token))
    missing = sorted(selected.difference(available_set))
    if missing:
        raise ValueError(f"Requested episodes are absent: {missing}")
    return sorted(selected)


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, float]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool), 0, 0.0
    sizes = np.bincount(labels.reshape(-1))
    foreground_total = int(sizes[1:].sum())
    sizes[0] = 0
    component = labels == int(sizes.argmax())
    fraction = float(component.sum() / foreground_total) if foreground_total else 0.0
    return component, int(count), fraction


def mask_to_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def save_overlay(image: Image.Image, mask: np.ndarray, box: list[int] | None, path: Path) -> None:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    overlay = pixels.copy()
    color = np.array([20, 210, 120], dtype=np.uint8)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    rendered = Image.fromarray(overlay)
    if box is not None:
        draw = ImageDraw.Draw(rendered)
        draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline="#ff304f", width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(path)


class LeRobotVideoSource:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        files = sorted((self.root / "meta/episodes").glob("chunk-*/*.parquet"))
        if not files:
            raise FileNotFoundError(f"No episode metadata under {self.root}")
        episodes = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
        self.episodes = episodes.set_index("episode_index", drop=False).sort_index()
        self.fps = float(self.info["fps"])
        self.camera_keys = {
            key.rsplit(".", maxsplit=1)[-1]: key
            for key, feature in self.info["features"].items()
            if feature.get("dtype") == "video"
        }

    def row(self, episode_index: int) -> pd.Series:
        return self.episodes.loc[episode_index]

    def _video_path_and_timestamp(
        self, episode_index: int, camera: str, frame_index: int
    ) -> tuple[Path, float]:
        camera_key = self.camera_keys[camera]
        row = self.row(episode_index)
        prefix = f"videos/{camera_key}"
        relative = self.info["video_path"].format(
            video_key=camera_key,
            chunk_index=int(row[f"{prefix}/chunk_index"]),
            file_index=int(row[f"{prefix}/file_index"]),
        )
        timestamp = float(row[f"{prefix}/from_timestamp"]) + frame_index / self.fps
        return self.root / relative, timestamp

    @lru_cache(maxsize=64)
    def frame(self, episode_index: int, camera: str, frame_index: int) -> Image.Image:
        length = int(self.row(episode_index)["length"])
        frame_index = int(np.clip(frame_index, 0, length - 1))
        path, target_timestamp = self._video_path_and_timestamp(
            episode_index, camera, frame_index
        )
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            container.seek(
                int(max(0.0, target_timestamp - 1.0) / time_base),
                stream=stream,
                backward=True,
                any_frame=False,
            )
            nearest: tuple[float, Image.Image] | None = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                timestamp = float(frame.pts * stream.time_base)
                distance = abs(timestamp - target_timestamp)
                image = frame.to_image().convert("RGB")
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, image)
                if timestamp >= target_timestamp:
                    break
        if nearest is None:
            raise RuntimeError(f"Could not decode frame {frame_index} from {path}")
        return nearest[1]


def stage_target(segment: pd.Series) -> str | None:
    stage = str(segment["stage"])
    if stage == "grasp":
        return str(segment["source_entity"]).strip()
    if stage == "put":
        return str(segment["destination_entity"]).strip()
    return None


def keyframe_index(segment: pd.Series, grasp_fraction: float, put_fraction: float) -> int:
    start = int(segment["start_frame"])
    end = int(segment["end_frame"])
    fraction = grasp_fraction if segment["stage"] == "grasp" else put_fraction
    return start + int(round(fraction * max(0, end - start - 1)))


def item_stem(episode_index: int, segment_index: int, camera: str) -> str:
    return f"episode-{episode_index:06d}-segment-{segment_index:03d}-camera-{camera}"


def load_records(record_root: Path) -> pd.DataFrame:
    rows = [json.loads(path.read_text()) for path in sorted(record_root.glob("*.json"))]
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--planning-root", type=Path, default=DEFAULT_PLANNING_ROOT)
    parser.add_argument("--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episodes", default="all")
    parser.add_argument("--cameras", nargs="+", default=["top", "wrist"])
    parser.add_argument("--grasp-frame-fraction", type=float, default=0.50)
    parser.add_argument("--put-frame-fraction", type=float, default=0.65)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--insid3-root", type=Path, default=DEFAULT_INSID3_ROOT)
    parser.add_argument("--model-name", default=DEFAULT_TIMM_MODEL)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--svd-components", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--merge-threshold", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for fraction_name in ("grasp_frame_fraction", "put_frame_fraction"):
        value = float(getattr(args, fraction_name))
        if not 0 <= value <= 1:
            raise ValueError(f"--{fraction_name.replace('_', '-')} must be in [0, 1]")

    planning_path = args.planning_root / "review/segments.parquet"
    if not planning_path.is_file():
        raise FileNotFoundError(f"Reviewed planning segments are missing: {planning_path}")
    source = LeRobotVideoSource(args.dataset_root)
    missing_cameras = sorted(set(args.cameras).difference(source.camera_keys))
    if missing_cameras:
        raise ValueError(f"Dataset does not contain cameras: {missing_cameras}")

    manifest = json.loads(args.reference_manifest.read_text())
    references = manifest["references"]
    planning = pd.read_parquet(planning_path).sort_values(
        ["episode_index", "segment_index"]
    )
    available = sorted(planning["episode_index"].astype(int).unique().tolist())
    episodes = set(parse_episode_selection(args.episodes, available))
    planning = planning.loc[
        planning["episode_index"].isin(episodes) & planning["stage"].isin(["grasp", "put"])
    ]

    items: list[dict[str, Any]] = []
    for segment in planning.to_dict("records"):
        segment_series = pd.Series(segment)
        target = stage_target(segment_series)
        frame_index = keyframe_index(
            segment_series,
            args.grasp_frame_fraction,
            args.put_frame_fraction,
        )
        for camera in args.cameras:
            items.append({**segment, "camera": camera, "frame_index": frame_index, "target": target})
    record_root = args.output_root / "records"
    mask_root = args.output_root / "masks"
    preview_root = args.output_root / "previews"
    record_root.mkdir(parents=True, exist_ok=True)
    model = None
    processed = 0
    attempted = 0

    for ordinal, item in enumerate(items, start=1):
        episode_index = int(item["episode_index"])
        segment_index = int(item["segment_index"])
        camera = str(item["camera"])
        target = str(item["target"])
        stem = item_stem(episode_index, segment_index, camera)
        record_path = record_root / f"{stem}.json"
        if record_path.exists() and args.resume:
            continue
        if record_path.exists() and not args.resume:
            raise FileExistsError(f"{record_path} exists; pass --resume to continue")
        if args.max_items > 0 and attempted >= args.max_items:
            break
        attempted += 1

        base_record: dict[str, Any] = {
            "episode_index": episode_index,
            "segment_index": segment_index,
            "cycle_index": int(item["cycle_index"]),
            "stage": str(item["stage"]),
            "next_step": str(item["next_step"]),
            "camera": camera,
            "frame_index": int(item["frame_index"]),
            "target_entity": target,
            "bbox_format": "xyxy_inclusive_exclusive",
        }
        if target.startswith("vegetable on the "):
            record = {
                **base_record,
                "status": "unresolved_reference",
                "visible": None,
                "bbox_xyxy": None,
                "review_required": True,
                "reason": "Relational clear-plate source has no exact visual reference class.",
            }
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            continue
        if target not in references:
            record = {
                **base_record,
                "status": "unresolved_reference",
                "visible": None,
                "bbox_xyxy": None,
                "review_required": True,
                "reason": f"No INSID3 reference for {target!r}.",
            }
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            continue

        if model is None:
            model = build_insid3_with_timm(
                insid3_root=args.insid3_root,
                model_name=args.model_name,
                image_size=args.image_size,
                svd_components=args.svd_components,
                tau=args.tau,
                merge_threshold=args.merge_threshold,
                device=args.device,
            )
        target_image = source.frame(episode_index, camera, int(item["frame_index"]))
        reference = references[target]
        reference_images = reference["reference_images"]
        reference_masks = reference["reference_masks"]
        reference_cameras = reference.get("reference_cameras")
        if reference_cameras is not None:
            if len(reference_cameras) != len(reference_images):
                raise ValueError(f"Invalid reference camera entries for {target}")
            matching = [
                index
                for index, reference_camera in enumerate(reference_cameras)
                if reference_camera == camera
            ]
            if matching:
                reference_images = [reference_images[index] for index in matching]
                reference_masks = [reference_masks[index] for index in matching]
        if len(reference_images) != len(reference_masks) or not reference_images:
            raise ValueError(f"Invalid reference entries for {target}")
        for image_path, mask_path in zip(reference_images, reference_masks, strict=True):
            model.set_reference(image_path, mask_path)
        model.set_target(target_image)
        raw_mask = model.segment().detach().cpu().numpy().astype(bool)
        mask, component_count, largest_fraction = largest_component(raw_mask)
        box = mask_to_xyxy(mask)
        foreground_pixels = int(mask.sum())
        image_area = target_image.width * target_image.height
        box_area = (
            (box[2] - box[0]) * (box[3] - box[1]) if box is not None else 0
        )
        border_touch = bool(
            box is not None
            and (box[0] == 0 or box[1] == 0 or box[2] == target_image.width or box[3] == target_image.height)
        )
        status = "predicted" if box is not None else "empty_prediction"
        review_required = bool(
            box is None
            or border_touch
            or component_count > 3
            or not 0.001 <= foreground_pixels / image_area <= 0.45
        )

        mask_path = mask_root / f"{stem}.png"
        preview_path = preview_root / f"{stem}.jpg"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        save_overlay(target_image, mask, box, preview_path)
        record = {
            **base_record,
            "status": status,
            "visible": box is not None,
            "bbox_xyxy": box,
            "image_width": target_image.width,
            "image_height": target_image.height,
            "foreground_pixels": foreground_pixels,
            "foreground_fraction": foreground_pixels / image_area,
            "box_area_fraction": box_area / image_area,
            "component_count": component_count,
            "largest_component_fraction": largest_fraction,
            "border_touch": border_touch,
            "review_required": review_required,
            "mask_path": str(mask_path.relative_to(args.output_root)),
            "preview_path": str(preview_path.relative_to(args.output_root)),
            "reference_mask_method": reference.get("mask_method", "unknown"),
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        processed += 1
        if ordinal % 10 == 0 or ordinal == len(items):
            logger.info("Processed %d/%d items", ordinal, len(items))

    table = load_records(record_root)
    data_path = args.output_root / "data/boxes.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = data_path.with_suffix(".parquet.tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(data_path)
    summary = {
        "format": "insid3_rm65_stage_boxes",
        "format_version": 1,
        "dataset_root": str(args.dataset_root.resolve()),
        "planning_root": str(args.planning_root.resolve()),
        "reference_manifest": str(args.reference_manifest.resolve()),
        "item_count": len(table),
        "new_predictions": processed,
        "status_counts": table["status"].value_counts().to_dict() if len(table) else {},
        "review_required_count": int(table["review_required"].fillna(True).sum()) if len(table) else 0,
        "parameters": {
            "cameras": args.cameras,
            "grasp_frame_fraction": args.grasp_frame_fraction,
            "put_frame_fraction": args.put_frame_fraction,
            "model_name": args.model_name,
            "image_size": args.image_size,
            "svd_components": args.svd_components,
            "tau": args.tau,
            "merge_threshold": args.merge_threshold,
        },
    }
    info_path = args.output_root / "meta/info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Wrote %d records to %s", len(table), data_path)


if __name__ == "__main__":
    main()
