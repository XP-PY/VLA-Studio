#!/usr/bin/env python3
"""Detect YOLOE targets throughout a video and derive positive SAM2 points.

The reference image and boxes initialize YOLOE visual-prompt classes. Every
video frame is processed by default. For each configured target, the script
retains the globally highest-confidence detection and places a positive point
at the maximum of its segmentation mask's distance transform.

Example:

python examples/action_flow_v2/ref/test_yoloe_target_points.py \
    --video /path/to/top-or-wrist-video.mp4 \
    --config examples/action_flow_v2/ref/yoloe_reference_realsense_viewer.yaml \
    --device 0

Outputs include an annotated video, per-frame point CSV, best-point CSV/JSON,
and a contact sheet showing the best frame and point for every target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np

from test_yoloe_visual_prompt_video import (
    COLORS,
    add_panel_header,
    draw_detection,
    get_video_fps,
    import_yoloe,
    load_reference_config,
    overlay_mask,
    select_highest_confidence,
)


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("yoloe_reference_realsense_viewer_with_collection_box.yaml")
DEFAULT_MODEL = WORKSPACE / ".cache" / "yoloe" / "weights" / "yoloe-26x-seg.pt"


@dataclass
class BestTargetPoint:
    class_id: int
    name: str
    confidence: float
    frame_index: int
    point_xy: tuple[int, int]
    point_source: str
    box_xyxy: tuple[float, float, float, float]
    annotated_image: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Local YOLOE segmentation checkpoint or an Ultralytics model name.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        help="Optional names such as tomato or pink_plate. Default: all reference classes.",
    )
    parser.add_argument("--output-dir", type=Path, default="runs/yoloe/test_yoloe_target_points")
    parser.add_argument("--device", default="0", help="Ultralytics device, such as 0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth video frame. The default 1 processes every frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum processed frames for a quick test. Zero processes the complete video.",
    )
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--mask-alpha", type=float, default=0.3)
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def select_targets(objects, requested_names: list[str] | None):
    if requested_names is None:
        return objects
    objects_by_name = {}
    for item in objects:
        objects_by_name[normalize_name(item.name)] = item
        objects_by_name[normalize_name(item.display_name)] = item

    selected = []
    unknown = []
    for requested_name in requested_names:
        item = objects_by_name.get(normalize_name(requested_name))
        if item is None:
            unknown.append(requested_name)
        elif item not in selected:
            selected.append(item)
    if unknown:
        available = ", ".join(item.name for item in objects)
        raise ValueError(f"Unknown targets {unknown}. Available targets: {available}")
    return selected


def decode_video_batches(
    video_path: Path,
    batch_size: int,
    frame_stride: int,
    max_frames: int,
) -> Iterator[list[tuple[int, np.ndarray]]]:
    """Yield batches of ``(original_frame_index, BGR frame)`` pairs."""
    batch = []
    processed_frames = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index % frame_stride != 0:
                continue
            batch.append((frame_index, frame.to_ndarray(format="bgr24")))
            processed_frames += 1
            if len(batch) == batch_size:
                yield batch
                batch = []
            if max_frames > 0 and processed_frames >= max_frames:
                break
    if batch:
        yield batch


def resize_binary_mask(
    mask: np.ndarray,
    width: int,
    height: int,
    threshold: float,
) -> np.ndarray:
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return (mask >= threshold).astype(np.uint8)


def mask_interior_point(mask: np.ndarray) -> tuple[int, int] | None:
    """Return the mask pixel farthest from its boundary as an ``(x, y)`` point."""
    if mask.ndim != 2 or not np.any(mask):
        return None
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    if distance[y, x] <= 0:
        return None
    return int(x), int(y)


def box_center(box_xyxy: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x0, y0, x1, y1 = box_xyxy
    x = int(np.clip(round(0.5 * (x0 + x1)), 0, width - 1))
    y = int(np.clip(round(0.5 * (y0 + y1)), 0, height - 1))
    return x, y


def draw_positive_point(
    image: np.ndarray,
    point_xy: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = point_xy
    cv2.circle(image, (x, y), 8, (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.circle(image, (x, y), 5, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.drawMarker(
        image,
        (x, y),
        (0, 0, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=13,
        thickness=2,
        line_type=cv2.LINE_AA,
    )


def save_best_contact_sheet(
    output_path: Path,
    selected_objects,
    best_points: dict[int, BestTargetPoint],
    frame_shape: tuple[int, int],
    columns: int = 2,
) -> None:
    frame_height, frame_width = frame_shape
    panels = []
    for item in selected_objects:
        best = best_points.get(item.class_id)
        color = COLORS[item.class_id % len(COLORS)]
        if best is None:
            image = np.full((frame_height, frame_width, 3), 24, dtype=np.uint8)
            header = f"{item.display_name} | NOT DETECTED"
        else:
            image = best.annotated_image
            header = (
                f"{item.display_name} | confidence={best.confidence:.3f} | "
                f"frame={best.frame_index} | point={best.point_xy}"
            )
        panels.append(add_panel_header(image, header, color))

    panel_height, panel_width = panels[0].shape[:2]
    blank = np.full((panel_height, panel_width, 3), 24, dtype=np.uint8)
    rows = []
    for row_index in range(math.ceil(len(panels) / columns)):
        row = panels[row_index * columns : (row_index + 1) * columns]
        while len(row) < columns:
            row.append(blank.copy())
        rows.append(np.hstack(row))
    if not cv2.imwrite(str(output_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not save contact sheet: {output_path}")


def serializable_best_point(best: BestTargetPoint) -> dict:
    return {
        "class_id": best.class_id,
        "name": best.name,
        "detected": True,
        "confidence": best.confidence,
        "frame_index": best.frame_index,
        "point_x": best.point_xy[0],
        "point_y": best.point_xy[1],
        "sam_label": 1,
        "point_source": best.point_source,
        "box_xyxy": list(best.box_xyxy),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.frame_stride < 1 or args.imgsz < 1:
        raise ValueError("--batch-size, --frame-stride, and --imgsz must be positive")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("--conf and --iou must be between 0 and 1")
    if not 0 <= args.mask_threshold <= 1 or not 0 <= args.mask_alpha <= 1:
        raise ValueError("--mask-threshold and --mask-alpha must be between 0 and 1")

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    reference_image, objects = load_reference_config(args.config.expanduser().resolve())
    selected_objects = select_targets(objects, args.targets)
    object_by_id = {item.class_id: item for item in selected_objects}
    known_class_ids = set(object_by_id)
    visual_prompts = {
        "bboxes": np.asarray([item.bbox_xyxy for item in objects], dtype=np.float32),
        "cls": np.asarray([item.class_id for item in objects], dtype=np.int64),
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{video_path.stem}_target_points"
    video_output = output_dir / f"{output_stem}.mp4"
    frame_csv_path = output_dir / f"{output_stem}_per_frame.csv"
    best_csv_path = output_dir / f"{output_stem}_best.csv"
    best_json_path = output_dir / f"{output_stem}_best.json"
    contact_sheet_path = output_dir / f"{output_stem}_best.jpg"

    YOLOE, YOLOEVPSegPredictor = import_yoloe()
    model = YOLOE(args.model)
    reference_initialized = False
    writer: cv2.VideoWriter | None = None
    best_points: dict[int, BestTargetPoint] = {}
    frame_shape: tuple[int, int] | None = None
    processed_frames = 0
    input_fps = get_video_fps(video_path)

    frame_fieldnames = [
        "frame_index",
        "class_id",
        "name",
        "confidence",
        "point_x",
        "point_y",
        "sam_label",
        "point_source",
        "box_xyxy",
    ]
    with frame_csv_path.open("w", newline="") as frame_csv_file:
        frame_csv = csv.DictWriter(frame_csv_file, fieldnames=frame_fieldnames)
        frame_csv.writeheader()
        try:
            for frame_batch in decode_video_batches(
                video_path,
                args.batch_size,
                args.frame_stride,
                args.max_frames,
            ):
                frame_indices = [item[0] for item in frame_batch]
                frames = [item[1] for item in frame_batch]
                prediction_args = {
                    "source": frames,
                    "imgsz": args.imgsz,
                    "conf": args.conf,
                    "iou": args.iou,
                    "device": args.device,
                    "retina_masks": True,
                    "verbose": False,
                }
                if not reference_initialized:
                    prediction_args.update(
                        {
                            "refer_image": str(reference_image.resolve()),
                            "visual_prompts": visual_prompts,
                            "predictor": YOLOEVPSegPredictor,
                        }
                    )
                    reference_initialized = True

                results = model.predict(**prediction_args)
                if len(results) != len(frames):
                    raise RuntimeError(
                        f"YOLOE returned {len(results)} results for {len(frames)} frames"
                    )

                for frame_index, result in zip(frame_indices, results, strict=True):
                    annotated = result.orig_img.copy()
                    frame_shape = annotated.shape[:2]
                    height, width = frame_shape
                    selected = select_highest_confidence(result, known_class_ids)
                    boxes = (
                        result.boxes.xyxy.detach().cpu().numpy()
                        if result.boxes is not None
                        else np.empty((0, 4), dtype=np.float32)
                    )
                    confidences = (
                        result.boxes.conf.detach().cpu().numpy()
                        if result.boxes is not None
                        else np.empty((0,), dtype=np.float32)
                    )
                    masks = (
                        result.masks.data.detach().cpu().numpy()
                        if result.masks is not None
                        else None
                    )

                    frame_records = []
                    for class_id, detection_index in sorted(selected.items()):
                        item = object_by_id[class_id]
                        box = boxes[detection_index]
                        point = None
                        point_source = "box_center"
                        binary_mask = None
                        if masks is not None and detection_index < len(masks):
                            binary_mask = resize_binary_mask(
                                masks[detection_index],
                                width,
                                height,
                                args.mask_threshold,
                            )
                            point = mask_interior_point(binary_mask)
                            if point is not None:
                                point_source = "mask_interior"
                        if point is None:
                            point = box_center(box, width, height)

                        confidence = float(confidences[detection_index])
                        color = COLORS[class_id % len(COLORS)]
                        if binary_mask is not None:
                            overlay_mask(annotated, binary_mask, color, args.mask_alpha)
                        draw_detection(
                            annotated,
                            box,
                            f"{item.display_name} {confidence:.2f} point={point}",
                            color,
                        )
                        draw_positive_point(annotated, point, color)

                        target_annotated = result.orig_img.copy()
                        if binary_mask is not None:
                            overlay_mask(
                                target_annotated,
                                binary_mask,
                                color,
                                args.mask_alpha,
                            )
                        draw_detection(
                            target_annotated,
                            box,
                            f"{item.display_name} {confidence:.2f} point={point}",
                            color,
                        )
                        draw_positive_point(target_annotated, point, color)
                        record = {
                            "frame_index": frame_index,
                            "class_id": class_id,
                            "name": item.name,
                            "confidence": confidence,
                            "point_x": point[0],
                            "point_y": point[1],
                            "sam_label": 1,
                            "point_source": point_source,
                            "box_xyxy": [float(value) for value in box],
                        }
                        frame_records.append(record)
                        previous = best_points.get(class_id)
                        if previous is None or confidence > previous.confidence:
                            best_points[class_id] = BestTargetPoint(
                                class_id=class_id,
                                name=item.name,
                                confidence=confidence,
                                frame_index=frame_index,
                                point_xy=point,
                                point_source=point_source,
                                box_xyxy=tuple(float(value) for value in box),
                                annotated_image=target_annotated,
                            )

                    for record in frame_records:
                        frame_csv.writerow(record)

                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(video_output),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            input_fps / args.frame_stride,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not create output video: {video_output}")
                    writer.write(annotated)
                    processed_frames += 1
                print(f"Processed {processed_frames} frames", end="\r", flush=True)
        finally:
            if writer is not None:
                writer.release()

    if processed_frames == 0 or frame_shape is None:
        raise RuntimeError("No video frames were processed")

    best_records = []
    for item in selected_objects:
        best = best_points.get(item.class_id)
        if best is None:
            best_records.append(
                {
                    "class_id": item.class_id,
                    "name": item.name,
                    "detected": False,
                    "confidence": None,
                    "frame_index": None,
                    "point_x": None,
                    "point_y": None,
                    "sam_label": None,
                    "point_source": None,
                    "box_xyxy": None,
                }
            )
        else:
            best_records.append(serializable_best_point(best))

    best_json_path.write_text(
        json.dumps(
            {
                "video": str(video_path),
                "reference_config": str(args.config.expanduser().resolve()),
                "processed_frames": processed_frames,
                "frame_stride": args.frame_stride,
                "points": best_records,
            },
            indent=2,
        )
        + "\n"
    )
    best_fieldnames = [
        "class_id",
        "name",
        "detected",
        "confidence",
        "frame_index",
        "point_x",
        "point_y",
        "sam_label",
        "point_source",
        "box_xyxy",
    ]
    with best_csv_path.open("w", newline="") as best_csv_file:
        writer_csv = csv.DictWriter(best_csv_file, fieldnames=best_fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(best_records)
    save_best_contact_sheet(
        contact_sheet_path,
        selected_objects,
        best_points,
        frame_shape,
    )

    print()
    for record in best_records:
        if record["detected"]:
            print(
                f"{record['name']}: frame={record['frame_index']}, "
                f"point=({record['point_x']}, {record['point_y']}), "
                f"confidence={record['confidence']:.3f}"
            )
        else:
            print(f"{record['name']}: NOT DETECTED")
    print(f"Annotated video: {video_output}")
    print(f"Per-frame points: {frame_csv_path}")
    print(f"Best-point visualization: {contact_sheet_path}")
    print(f"Best-point JSON: {best_json_path}")
    print(f"Best-point CSV: {best_csv_path}")


if __name__ == "__main__":
    main()
