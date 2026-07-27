#!/usr/bin/env python3
"""Test YOLOE visual-prompt detection on a video.

The script loads a reference image and its object boxes from
``yoloe_reference.yaml``. For every video frame, it keeps only the
highest-confidence detection for each configured class and writes:

- An annotated MP4 containing the selected boxes and instance masks.
- A CSV file containing the selected per-frame detections.
- A contact sheet containing the highest-confidence frame for each class.

Example:

    python runs/sam/ref/test_yoloe_visual_prompt_video.py \
        --video path/to/input.mp4 \
        --model models/yoloe-11l-seg.pt

The current reference config contains seven classes. Add a clean collection-box
reference before expecting the eighth target to be detected.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class ReferenceObject:
    class_id: int
    name: str
    display_name: str
    bbox_xyxy: tuple[float, float, float, float]


@dataclass
class BestDetection:
    confidence: float
    frame_index: int
    image: np.ndarray


COLORS = (
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
    (213, 94, 0),
    (204, 121, 167),
    (128, 128, 128),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOE visual-prompt detection and retain one result per class and frame."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input top- or wrist-camera MP4.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("yoloe_reference.yaml"),
        help="Reference YAML containing the reference image and object boxes.",
    )
    parser.add_argument(
        "--model",
        default="yoloe-11l-seg.pt",
        help="YOLOE segmentation checkpoint path or an Ultralytics model name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Annotated MP4 path. Defaults to yoloe_outputs/<video-stem>_detections.mp4.",
    )
    parser.add_argument("--device", default="0", help="Ultralytics inference device, such as 0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of CPU-decoded video frames passed to YOLOE together.",
    )
    parser.add_argument("--conf", type=float, default=0.15, help="Minimum detection confidence.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--mask-alpha", type=float, default=0.35, help="Mask overlay opacity.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for a quick test.",
    )
    return parser.parse_args()


def load_reference_config(config_path: Path) -> tuple[Path, list[ReferenceObject]]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Reference config does not exist: {config_path}")

    config = yaml.safe_load(config_path.read_text())
    reference = config["reference"]
    reference_image = Path(reference["image"])
    if not reference_image.is_absolute():
        reference_image = config_path.parent / reference_image
    if not reference_image.is_file():
        raise FileNotFoundError(f"Reference image does not exist: {reference_image}")

    objects = [
        ReferenceObject(
            class_id=int(item["class_id"]),
            name=str(item["name"]),
            display_name=str(item.get("display_name", item["name"])),
            bbox_xyxy=tuple(float(value) for value in item["bbox_xyxy"]),
        )
        for item in config["objects"]
    ]
    if not objects:
        raise ValueError("Reference config contains no objects")

    class_ids = [item.class_id for item in objects]
    if class_ids != list(range(len(objects))):
        raise ValueError(
            "YOLOE visual-prompt class IDs must be sequential and start at zero; "
            f"received {class_ids}"
        )

    image = cv2.imread(str(reference_image))
    if image is None:
        raise RuntimeError(f"OpenCV could not read reference image: {reference_image}")
    image_height, image_width = image.shape[:2]
    configured_size = (int(reference["width"]), int(reference["height"]))
    if configured_size != (image_width, image_height):
        raise ValueError(
            f"Reference size mismatch: YAML={configured_size}, image={(image_width, image_height)}"
        )

    for item in objects:
        x1, y1, x2, y2 = item.bbox_xyxy
        if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
            raise ValueError(f"Invalid box for {item.name}: {item.bbox_xyxy}")

    return reference_image, objects


def import_yoloe() -> tuple[Any, Any]:
    try:
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    except ImportError as exc:
        raise ImportError(
            "YOLOE requires Ultralytics. Install it with: python -m pip install -U ultralytics"
        ) from exc
    return YOLOE, YOLOEVPSegPredictor


def select_highest_confidence(
    result: Any,
    known_class_ids: set[int],
) -> dict[int, int]:
    if result.boxes is None or len(result.boxes) == 0:
        return {}

    class_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    confidences = result.boxes.conf.detach().cpu().numpy()
    selected: dict[int, int] = {}
    for detection_index, (class_id, confidence) in enumerate(zip(class_ids, confidences, strict=True)):
        class_id = int(class_id)
        if class_id not in known_class_ids:
            continue
        previous_index = selected.get(class_id)
        if previous_index is None or confidence > confidences[previous_index]:
            selected[class_id] = detection_index
    return selected


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    height, width = image.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(bool)
    if not mask.any():
        return
    colored = np.empty_like(image)
    colored[:] = color
    image[mask] = cv2.addWeighted(image[mask], 1.0 - alpha, colored[mask], alpha, 0)


def draw_detection(
    image: np.ndarray,
    box_xyxy: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = np.rint(box_xyxy).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_width, text_height = text_size
    text_y = max(y1, text_height + baseline + 4)
    cv2.rectangle(
        image,
        (x1, text_y - text_height - baseline - 4),
        (x1 + text_width + 6, text_y),
        color,
        thickness=-1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 3, text_y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def add_panel_header(
    image: np.ndarray,
    text: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    header_height = 42
    header = np.full((header_height, image.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(
        header,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    return np.vstack((header, image))


def save_best_detection_contact_sheet(
    output_path: Path,
    objects: list[ReferenceObject],
    best_detections: dict[int, BestDetection],
    frame_shape: tuple[int, int],
    columns: int = 2,
) -> Path:
    frame_height, frame_width = frame_shape
    panels: list[np.ndarray] = []
    for item in objects:
        color = COLORS[item.class_id % len(COLORS)]
        best = best_detections.get(item.class_id)
        if best is None:
            image = np.full((frame_height, frame_width, 3), 24, dtype=np.uint8)
            header = f"{item.display_name} | NOT DETECTED"
        else:
            image = best.image
            header = (
                f"{item.display_name} | confidence={best.confidence:.3f} "
                f"| frame={best.frame_index}"
            )
        panels.append(add_panel_header(image, header, color))

    rows = []
    panel_shape = panels[0].shape
    for start in range(0, len(panels), columns):
        row = panels[start : start + columns]
        while len(row) < columns:
            row.append(np.full(panel_shape, 24, dtype=np.uint8))
        rows.append(np.hstack(row))

    contact_sheet = np.vstack(rows)
    summary_path = output_path.with_name(f"{output_path.stem}_best_detections.jpg")
    if not cv2.imwrite(str(summary_path), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not save best-detection contact sheet: {summary_path}")
    return summary_path


def get_video_fps(video_path: Path) -> float:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        return float(rate) if rate is not None and float(rate) > 0 else 30.0


def decode_video_batches(
    video_path: Path,
    batch_size: int,
    max_frames: int | None,
):
    """Decode video frames in software with PyAV and yield BGR frame batches."""
    batch: list[np.ndarray] = []
    decoded_frames = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if max_frames is not None and decoded_frames >= max_frames:
                break
            batch.append(frame.to_ndarray(format="bgr24"))
            decoded_frames += 1
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def run(args: argparse.Namespace) -> None:
    if not args.video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {args.video}")
    if not 0 <= args.mask_alpha <= 1:
        raise ValueError("--mask-alpha must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    reference_image, objects = load_reference_config(args.config.resolve())
    object_by_id = {item.class_id: item for item in objects}
    known_class_ids = set(object_by_id)

    output_path = args.output
    if output_path is None:
        output_path = args.config.parent / "yoloe_outputs" / f"{args.video.stem}_detections.mp4"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")

    visual_prompts = {
        "bboxes": np.asarray([item.bbox_xyxy for item in objects], dtype=np.float32),
        "cls": np.asarray([item.class_id for item in objects], dtype=np.int64),
    }

    YOLOE, YOLOEVPSegPredictor = import_yoloe()
    model = YOLOE(args.model)

    writer: cv2.VideoWriter | None = None
    fps = get_video_fps(args.video)
    frame_count = 0
    detection_count = 0
    reference_initialized = False
    best_detections: dict[int, BestDetection] = {}
    frame_shape: tuple[int, int] | None = None

    with csv_path.open("w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["frame_index", "class_id", "name", "confidence", "x1", "y1", "x2", "y2"]
        )

        try:
            for frame_batch in decode_video_batches(args.video, args.batch_size, args.max_frames):
                prediction_args = {
                    "source": frame_batch,
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
                for result in results:
                    frame_index = frame_count
                    annotated = result.orig_img.copy()
                    frame_shape = annotated.shape[:2]
                    selected = select_highest_confidence(result, known_class_ids)
                    boxes_xyxy = (
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

                    for class_id, detection_index in sorted(selected.items()):
                        reference_object = object_by_id[class_id]
                        color = COLORS[class_id % len(COLORS)]
                        confidence = float(confidences[detection_index])
                        if masks is not None and detection_index < len(masks):
                            overlay_mask(annotated, masks[detection_index], color, args.mask_alpha)
                        box = boxes_xyxy[detection_index]
                        draw_detection(
                            annotated,
                            box,
                            f"{reference_object.display_name} {confidence:.2f}",
                            color,
                        )
                        previous_best = best_detections.get(class_id)
                        if previous_best is None or confidence > previous_best.confidence:
                            best_image = result.orig_img.copy()
                            if masks is not None and detection_index < len(masks):
                                overlay_mask(
                                    best_image,
                                    masks[detection_index],
                                    color,
                                    args.mask_alpha,
                                )
                            draw_detection(
                                best_image,
                                box,
                                f"{reference_object.display_name} {confidence:.2f}",
                                color,
                            )
                            best_detections[class_id] = BestDetection(
                                confidence=confidence,
                                frame_index=frame_index,
                                image=best_image,
                            )
                        csv_writer.writerow(
                            [
                                frame_index,
                                class_id,
                                reference_object.name,
                                f"{confidence:.6f}",
                                *(f"{coordinate:.2f}" for coordinate in box),
                            ]
                        )
                        detection_count += 1

                    if writer is None:
                        height, width = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not create output video: {output_path}")

                    writer.write(annotated)
                    frame_count += 1
                print(f"Processed {frame_count} frames", end="\r", flush=True)
        finally:
            if writer is not None:
                writer.release()

    if frame_count == 0:
        raise RuntimeError("YOLOE produced no video frames")
    if frame_shape is None:
        raise RuntimeError("Could not determine the input video frame shape")

    summary_path = save_best_detection_contact_sheet(
        output_path,
        objects,
        best_detections,
        frame_shape,
    )

    print()
    print(f"Reference image: {reference_image}")
    print(f"Configured classes ({len(objects)}): {[item.name for item in objects]}")
    print(f"Processed frames: {frame_count}")
    print(f"Selected detections: {detection_count}")
    print(f"Annotated video: {output_path}")
    print(f"Detection CSV: {csv_path}")
    print(f"Best-detection contact sheet: {summary_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
