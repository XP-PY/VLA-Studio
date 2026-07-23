#!/usr/bin/env python3

"""Run Grounding DINO over synchronized videos and visualize every sampled frame.

The default inputs are episode 0 from the table-preparation dataset. Grounding
DINO processes the top and wrist frames as one batch. The script writes:

- ``detections.mp4``: synchronized camera views with labeled boxes.
- ``detections.jsonl``: pixel and normalized boxes for each sampled frame.
- ``run.json``: model, query, threshold, and video metadata.

Example:
    python examples/action_flow_v2/inspect_grounding_video.py \
        --queries "tomato" "pink square bowl" "white garlic" "cream square bowl" \
        --frame-stride 5 --device cuda

Use ``--frame-stride 1`` to infer every source frame. Larger strides reduce
runtime; the output FPS is reduced by the same factor so playback duration
still matches the source video.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


WORKSPACE = Path(__file__).resolve().parents[2]

DEFAULT_QUERIES = (
    "red tomato", 
    "yellow corn",
    "green cucumber",
    "white garlic", 
    "yellow square bowl",
    "pink square bowl", 
    "blue square bowl", 
    "brown cardboard box",
)
DEFAULT_BXO_THRESHOLD=0.7
DEFAULT_TEXT_THRESHOLD=0.7
DEFAULT_VIDEO_ID='030'

DEFAULT_MODEL_PATH = WORKSPACE / "models" / "grounding-dino-tiny"
DEFAULT_VIDEO_ROOT = (
    WORKSPACE
    / "runs"
    / "dataset"
    / "rm65_single_table_preparation_and_bussing"
    / "videos"
)
DEFAULT_VIDEOS = (
    (
        "top",
        DEFAULT_VIDEO_ROOT / "observation.images.top" / "chunk-000" / f"file-{DEFAULT_VIDEO_ID}.mp4",
    ),
    (
        "wrist",
        DEFAULT_VIDEO_ROOT / "observation.images.wrist" / "chunk-000" / f"file-{DEFAULT_VIDEO_ID}.mp4",
    ),
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE
    / "runs"
    / "sam"
    / "rm65_single_table_preparation_and_bussing"
    / "grounding_dino_video"
    / "episode-000"
)
COLORS = (
    (64, 64, 240),
    (80, 200, 80),
    (240, 130, 50),
    (60, 200, 220),
    (210, 80, 210),
    (220, 180, 60),
    (150, 50, 60),
    (110, 150, 10),
    (200, 20, 60),
)

logger = logging.getLogger(__name__)


@dataclass
class VideoInput:
    name: str
    path: Path
    container: av.container.InputContainer
    decoder: Iterator[av.VideoFrame]
    width: int
    height: int
    fps: float
    frame_count: int


def setup_logging(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(output_root / "inspect.log", mode="w")
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[console, file_handler], force=True)


def parse_video_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--video must use NAME=PATH")
    name, path = value.split("=", maxsplit=1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--video must use non-empty NAME=PATH")
    return name.strip(), Path(path).expanduser()


def open_video(name: str, path: Path) -> VideoInput:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    width = int(stream.width)
    height = int(stream.height)
    fps = float(stream.average_rate)
    frame_count = int(stream.frames)
    if width <= 0 or height <= 0 or fps <= 0:
        container.close()
        raise RuntimeError(
            f"Invalid video metadata for {path}: width={width}, height={height}, fps={fps}"
        )
    return VideoInput(
        name,
        path,
        container,
        iter(container.decode(stream)),
        width,
        height,
        fps,
        frame_count,
    )


def post_process_detections(
    processor,
    outputs,
    input_ids: torch.Tensor,
    frame_shapes: list[tuple[int, int]],
    box_threshold: float,
    text_threshold: float,
) -> list[list[dict]]:
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=frame_shapes,
    )
    detections_by_image = []
    paired_results = zip(results, frame_shapes, strict=True)
    for image_index, (result, (height, width)) in enumerate(paired_results):
        labels = result["text_labels"]
        scores = result["scores"]
        boxes = result["boxes"]
        count = min(len(labels), len(scores), len(boxes))
        if len(labels) != len(scores) or len(scores) != len(boxes):
            logger.warning(
                "Result %d has misaligned fields: labels=%d scores=%d boxes=%d; using %d.",
                image_index,
                len(labels),
                len(scores),
                len(boxes),
                count,
            )

        detections = []
        for label, score, box in zip(labels[:count], scores[:count], boxes[:count], strict=True):
            x0, y0, x1, y1 = (float(value) for value in box.detach().cpu().tolist())
            detections.append(
                {
                    "label": " ".join(str(label).split()),
                    "score": float(score.item()),
                    "box_xyxy": [x0, y0, x1, y1],
                    "box_xyxy_normalized": [
                        x0 / width,
                        y0 / height,
                        x1 / width,
                        y1 / height,
                    ],
                }
            )
        detections_by_image.append(detections)
    return detections_by_image


def draw_detections(
    frame: np.ndarray,
    camera: str,
    frame_index: int,
    timestamp_s: float,
    detections: list[dict],
    query_colors: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 38), (20, 20, 20), thickness=-1)
    cv2.putText(
        output,
        f"{camera} | frame {frame_index} | {timestamp_s:.2f}s | detections {len(detections)}",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    for detection_index, detection in enumerate(detections):
        label = detection["label"]
        color = query_colors.get(label, COLORS[detection_index % len(COLORS)])
        x0, y0, x1, y1 = (round(value) for value in detection["box_xyxy"])
        x0 = int(np.clip(x0, 0, output.shape[1] - 1))
        y0 = int(np.clip(y0, 0, output.shape[0] - 1))
        x1 = int(np.clip(x1, 0, output.shape[1] - 1))
        y1 = int(np.clip(y1, 0, output.shape[0] - 1))
        cv2.rectangle(output, (x0, y0), (x1, y1), color, thickness=2)
        text = f"{label} {detection['score']:.3f}"
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )
        label_y0 = max(38, y0 - text_height - 8)
        cv2.rectangle(
            output,
            (x0, label_y0),
            (min(output.shape[1] - 1, x0 + text_width + 8), label_y0 + text_height + 8),
            color,
            thickness=-1,
        )
        cv2.putText(
            output,
            text,
            (x0 + 4, label_y0 + text_height + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        action="append",
        type=parse_video_arg,
        help="Synchronized input as NAME=PATH. Repeat for each camera.",
    )
    parser.add_argument("--queries", nargs="+", default=list(DEFAULT_QUERIES))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument(
        "--max-sampled-frames",
        type=int,
        default=0,
        help="Stop after this many output frames; 0 processes the complete videos.",
    )
    parser.add_argument("--box-threshold", type=float, default=DEFAULT_BXO_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    parser.add_argument("--device", default="cuda", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--codec", default="libx264")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if args.max_sampled_frames < 0:
        raise ValueError("--max-sampled-frames cannot be negative")

    output_root = args.output_root.resolve()
    setup_logging(output_root)
    video_specs = args.video or list(DEFAULT_VIDEOS)
    videos = [open_video(name, path) for name, path in video_specs]
    output_container = None
    output_stream = None
    try:
        reference = videos[0]
        for video in videos[1:]:
            if abs(video.fps - reference.fps) > 1e-3:
                raise ValueError(
                    f"Videos are not synchronized: {reference.name}={reference.fps} FPS, "
                    f"{video.name}={video.fps} FPS"
                )
            if video.height != reference.height:
                raise ValueError(
                    f"Videos must have equal heights: {reference.name}={reference.height}, "
                    f"{video.name}={video.height}"
                )

        queries = list(dict.fromkeys(query.strip() for query in args.queries if query.strip()))
        if not queries:
            raise ValueError("--queries must include at least one non-empty phrase")
        prompt = ". ".join(queries) + "."
        query_colors = {query: COLORS[index % len(COLORS)] for index, query in enumerate(queries)}

        device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device_name == "auto":
            device_name = "cpu"
        device = torch.device(device_name)
        model_path = args.model_path.resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)

        logger.info("Loading Grounding DINO from %s on %s.", model_path, device)
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(model_path), local_files_only=True
        ).to(device).eval()

        output_video = output_root / "detections.mp4"
        output_jsonl = output_root / "detections.jsonl"
        output_width = sum(video.width for video in videos)
        output_fps = reference.fps / args.frame_stride
        output_container = av.open(str(output_video), mode="w")
        output_stream = output_container.add_stream(
            args.codec,
            rate=Fraction(output_fps).limit_denominator(1000),
        )
        output_stream.width = output_width
        output_stream.height = reference.height
        output_stream.pix_fmt = "yuv420p"
        if args.codec == "libx264":
            output_stream.options = {"crf": "20", "preset": "fast"}

        run_metadata = {
            "model_path": str(model_path),
            "device": str(device),
            "videos": [
                {
                    "camera": video.name,
                    "path": str(video.path),
                    "width": video.width,
                    "height": video.height,
                    "fps": video.fps,
                    "frame_count": video.frame_count,
                }
                for video in videos
            ],
            "queries": queries,
            "grounding_prompt": prompt,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "frame_stride": args.frame_stride,
            "output_fps": output_fps,
            "output_codec": args.codec,
        }
        (output_root / "run.json").write_text(
            json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
        )

        frame_index = 0
        sampled_index = 0
        with output_jsonl.open("w", encoding="utf-8") as detection_file:
            while True:
                frames = []
                for video in videos:
                    try:
                        decoded_frame = next(video.decoder)
                    except StopIteration:
                        frames = []
                        break
                    frames.append(decoded_frame.to_ndarray(format="bgr24"))
                if not frames:
                    break
                if frame_index % args.frame_stride != 0:
                    frame_index += 1
                    continue

                rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
                images = [Image.fromarray(frame) for frame in rgb_frames]
                inputs = processor(
                    images=images,
                    text=[prompt] * len(images),
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                with torch.inference_mode():
                    outputs = model(**inputs)
                detections_by_camera = post_process_detections(
                    processor,
                    outputs,
                    inputs.input_ids,
                    [(video.height, video.width) for video in videos],
                    args.box_threshold,
                    args.text_threshold,
                )

                timestamp_s = frame_index / reference.fps
                panels = []
                for video, frame, detections in zip(
                    videos, frames, detections_by_camera, strict=True
                ):
                    panels.append(
                        draw_detections(
                            frame,
                            video.name,
                            frame_index,
                            timestamp_s,
                            detections,
                            query_colors,
                        )
                    )
                    detection_file.write(
                        json.dumps(
                            {
                                "source_frame_index": frame_index,
                                "sampled_frame_index": sampled_index,
                                "timestamp_s": timestamp_s,
                                "camera": video.name,
                                "detections": detections,
                            }
                        )
                        + "\n"
                    )
                combined_frame = np.concatenate(panels, axis=1)
                av_frame = av.VideoFrame.from_ndarray(combined_frame, format="bgr24")
                for packet in output_stream.encode(av_frame):
                    output_container.mux(packet)
                sampled_index += 1
                if sampled_index % 10 == 0:
                    logger.info(
                        "Processed %d sampled frames through source frame %d.",
                        sampled_index,
                        frame_index,
                    )
                if (
                    args.max_sampled_frames > 0
                    and sampled_index >= args.max_sampled_frames
                ):
                    break
                frame_index += 1

        for packet in output_stream.encode():
            output_container.mux(packet)
        output_container.close()
        output_container = None
        logger.info(
            "Finished %d sampled frames. Video: %s | Detections: %s",
            sampled_index,
            output_video,
            output_jsonl,
        )
    finally:
        if output_container is not None:
            output_container.close()
        for video in videos:
            video.container.close()


if __name__ == "__main__":
    main()
