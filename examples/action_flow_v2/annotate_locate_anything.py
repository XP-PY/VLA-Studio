#!/usr/bin/env python3
"""Initialize Action Flow V2 object tracks with NVIDIA LocateAnything-3B.

The script reads task entities from a LeRobot dataset, queries one entity at a
time so that every returned box keeps its deterministic episode-local object
ID, and scans sampled top/wrist frames for a clear initialization view. Raw
model responses, selected boxes, ranked alternatives, and review contact
sheets are saved without modifying the source dataset.

Quick test (episode indices are zero-based):
    python examples/action_flow_v2/annotate_locate_anything.py \
        --episodes 0 --cameras top --max-sampled-frames 2 \
        --device cuda --overwrite

python examples/action_flow_v2/annotate_locate_anything.py \
        --episodes 0,1,2 --device cuda --overwrite

Process the complete dataset:
    python examples/action_flow_v2/annotate_locate_anything.py \
        --episodes all --cameras top wrist --device cuda

Outputs default to ``runs/sam/<dataset-name>/locate_anything``. Open
``inspection/index.html`` to review the selected boxes before passing them to
SAM2. Existing results produced with the same configuration are skipped unless
``--overwrite`` is supplied.

The model uses Hugging Face remote code. Review and trust the model repository
before running this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

try:
    from .annotate_grounding import (
        Detection,
        EntitySpec,
        EpisodeVideoSource,
        detection_row,
        entities_from_task,
        missing_detection,
        parse_episode_selection,
        rebuild_summary,
        result_is_current,
        result_paths,
        retain_ranked_candidate,
        setup_logging,
        write_candidate_contact_sheet,
        write_contact_sheet,
        write_json_atomic,
        write_parquet_atomic,
    )
except ImportError:
    from annotate_grounding import (  # type: ignore[no-redef]
        Detection,
        EntitySpec,
        EpisodeVideoSource,
        detection_row,
        entities_from_task,
        missing_detection,
        parse_episode_selection,
        rebuild_summary,
        result_is_current,
        result_paths,
        retain_ranked_candidate,
        setup_logging,
        write_candidate_contact_sheet,
        write_contact_sheet,
        write_json_atomic,
        write_parquet_atomic,
    )


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = WORKSPACE / "runs" / "dataset" / "rm65_single_table_preparation_and_bussing"
DEFAULT_MODEL = "nvidia/LocateAnything-3B"
SUPPORTED_CAMERAS = ("top", "wrist")
MIN_QUALITY = 0.6
LOCATE_QUERIES = {
    "tomato": "tomato",
    "garlic": "white garlic",
    "corn": "corn",
    "cucumber": "cucumber",
    "pink plate": "pink square bowl",
    "yellow plate": "cream square bowl",
    "blue plate": "blue square bowl",
    # "pink plate": "the pink square dish",
    # "yellow plate": "the cream yellow square dish",
    # "blue plate": "the blue square dish",
    "collection box": "a open brown cardboard collection box",
}
BOX_PATTERN = re.compile(r"<box>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*</box>")
PLAIN_BOX_PATTERN = re.compile(
    r"<box>\s*(\d+)\s*[, ]+\s*(\d+)\s*[, ]+\s*(\d+)\s*[, ]+\s*(\d+)\s*</box>"
)

logger = logging.getLogger(__name__)


def locate_query(entity: EntitySpec) -> str:
    """Return a LocateAnything referring expression for an entity."""
    relation = re.fullmatch(r"vegetable on (pink|yellow|blue) plate", entity.object_name)
    if relation:
        plate_query = LOCATE_QUERIES[f"{relation.group(1)} plate"]
        return f"the vegetable model resting in {plate_query}"
    return LOCATE_QUERIES.get(entity.object_name, entity.object_name)


def normalize_entities(entities: list[EntitySpec]) -> list[EntitySpec]:
    """Replace detector-specific queries while preserving semantic metadata."""
    return [replace(entity, query=locate_query(entity)) for entity in entities]


def normalize_answer(answer) -> str:
    """Normalize the custom generation API's answer into one string."""
    if isinstance(answer, (list, tuple)):
        answer = answer[0] if answer else ""
    return answer if isinstance(answer, str) else str(answer)


def parse_boxes(answer: str, width: int, height: int) -> list[tuple[float, float, float, float]]:
    """Parse normalized ``[0, 1000]`` box coordinates into image pixels."""
    matches = list(BOX_PATTERN.finditer(answer))
    if not matches:
        matches = list(PLAIN_BOX_PATTERN.finditer(answer))

    boxes = []
    for match in matches:
        values = [float(np.clip(int(value), 0, 1000)) for value in match.groups()]
        x0, x1 = sorted((values[0] / 1000.0 * width, values[2] / 1000.0 * width))
        y0, y1 = sorted((values[1] / 1000.0 * height, values[3] / 1000.0 * height))
        if x1 - x0 >= 1.0 and y1 - y0 >= 1.0:
            boxes.append((x0, y0, x1, y1))
    return boxes


class LocateAnythingWorker:
    """Load LocateAnything once and serve single-entity grounding queries."""

    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device,
        dtype: torch.dtype,
        local_files_only: bool,
    ):
        self.device = device
        self.dtype = dtype
        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, **load_kwargs)
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            **load_kwargs,
        ).to(device).eval()

    @torch.inference_mode()
    def ground_single(
        self,
        image: Image.Image,
        phrase: str,
        generation_mode: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        top_p: float,
        repetition_penalty: float,
        verbose: bool,
    ) -> str:
        """Return the model's raw answer for one referring expression."""
        question = f"Locate a single instance that matches the following description: {phrase}."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to(self.device)

        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            verbose=verbose,
        )
        answer = response[0] if isinstance(response, tuple) else response
        return normalize_answer(answer)


def score_box_view(
    image: Image.Image,
    frame_index: int,
    timestamp_s: float,
    box: tuple[float, float, float, float],
    entity: EntitySpec,
    camera: str,
    episode_length: int,
) -> Detection:
    """Score box completeness without detector confidence, which LocateAnything does not expose."""
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
        quality_score = 0.65 * area_score + 0.35 * border_score
    else:
        quality_score = 0.55 * area_score + 0.30 * border_score + 0.15 * sharpness_score
    allow_border_truncation = camera == "top" and entity.object_name == "collection box"
    if border_truncated and not allow_border_truncation:
        quality_score -= 0.50
    if area_ratio > 0.65:
        quality_score -= 0.25 * min((area_ratio - 0.65) / 0.35, 1.0)

    episode_progress = frame_index / max(episode_length - 1, 1)
    temporal_penalty = 0.50 * episode_progress if entity.object_role.startswith("source") else 0.0
    return Detection(
        visible=True,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        score=None,
        box_xyxy=box,
        image_width=width,
        image_height=height,
        area_ratio=area_ratio,
        border_margin=border_margin,
        sharpness_score=sharpness_score,
        quality_score=quality_score,
        episode_progress=episode_progress,
        selection_score=quality_score - temporal_penalty,
        border_truncated=border_truncated,
    )


def ground_episode_camera(
    source: EpisodeVideoSource,
    episode_index: int,
    camera: str,
    entities: list[EntitySpec],
    worker: LocateAnythingWorker,
    args: argparse.Namespace,
) -> tuple[
    list[Detection],
    dict[int, Image.Image],
    dict[int, list[tuple[Detection, Image.Image]]],
    Image.Image | None,
    int,
    list[dict],
]:
    """Scan sampled frames and retain the best LocateAnything box per entity."""
    ranked_candidates: dict[int, list[tuple[Detection, Image.Image]]] = {
        entity.object_id: [] for entity in entities
    }
    responses = []
    fallback_frame: Image.Image | None = None
    sampled_count = 0
    minimum_frame_gap = max(args.frame_stride * 2, 1)
    episode_length = int(source.row(episode_index)["length"])

    for frame_index, timestamp_s, image in source.sampled_frames(
        episode_index,
        camera,
        args.frame_stride,
        args.max_sampled_frames,
    ):
        sampled_count += 1
        if fallback_frame is None:
            fallback_frame = image.copy()
        for entity in entities:
            answer = worker.ground_single(
                image,
                entity.query,
                args.generation_mode,
                args.max_new_tokens,
                args.temperature,
                args.do_sample,
                args.top_p,
                args.repetition_penalty,
                args.verbose_model,
            )
            boxes = parse_boxes(answer, image.width, image.height)
            responses.append(
                {
                    "episode_index": episode_index,
                    "camera": camera,
                    "frame_index": frame_index,
                    "timestamp_s": timestamp_s,
                    "object_id": entity.object_id,
                    "object_name": entity.object_name,
                    "object_role": entity.object_role,
                    "query": entity.query,
                    "answer": answer,
                    "boxes_xyxy": boxes,
                }
            )
            if not boxes:
                continue
            frame_candidates = [
                score_box_view(
                    image,
                    frame_index,
                    timestamp_s,
                    box,
                    entity,
                    camera,
                    episode_length,
                )
                for box in boxes
            ]
            detection = max(frame_candidates, key=lambda candidate: candidate.selection_score)
            retain_ranked_candidate(
                ranked_candidates[entity.object_id],
                detection,
                image,
                args.top_k,
                minimum_frame_gap,
            )

    width = fallback_frame.width if fallback_frame is not None else 0
    height = fallback_frame.height if fallback_frame is not None else 0
    accepted = {
        entity.object_id: next(
            (
                candidate
                for candidate in ranked_candidates[entity.object_id]
                if candidate[0].quality_score >= args.min_quality
            ),
            None,
        )
        for entity in entities
    }
    detections = [
        accepted[entity.object_id][0]
        if accepted[entity.object_id] is not None
        else missing_detection(width, height)
        for entity in entities
    ]
    detection_frames = {
        entity.object_id: accepted[entity.object_id][1]
        for entity in entities
        if accepted[entity.object_id] is not None
    }
    return detections, detection_frames, ranked_candidates, fallback_frame, sampled_count, responses


def response_answer(responses: list[dict], object_id: int, frame_index: int) -> str:
    """Return the raw response associated with a retained candidate."""
    return next(
        (
            response["answer"]
            for response in responses
            if response["object_id"] == object_id and response["frame_index"] == frame_index
        ),
        "",
    )


def locate_result_paths(
    output_root: Path,
    episode_index: int,
    camera: str,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Return selected, candidate, preview, and raw-response paths."""
    common_paths = result_paths(output_root, episode_index, camera)
    response_path = (
        output_root
        / "responses"
        / f"episode-{episode_index:06d}"
        / f"camera-{camera}.json"
    )
    return (*common_paths, response_path)


def configuration_signature(args: argparse.Namespace) -> str:
    """Build a stable signature used to decide whether an output can be resumed."""
    payload = {
        "model": args.model,
        "frame_stride": args.frame_stride,
        "max_sampled_frames": args.max_sampled_frames,
        "top_k": args.top_k,
        "min_quality": args.min_quality,
        "generation_mode": args.generation_mode,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "dtype": args.dtype,
        "seed": args.seed,
        "queries": LOCATE_QUERIES,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def output_is_current(paths: tuple[Path, Path, Path, Path, Path, Path], signature: str) -> bool:
    """Return whether every artifact exists and matches this run configuration."""
    if not paths[5].is_file() or not result_is_current(paths[:5]):
        return False
    try:
        table = pd.read_parquet(paths[0], columns=["configuration_signature"])
    except Exception:
        return False
    return not table.empty and set(table["configuration_signature"].astype(str)) == {signature}


def make_record(
    episode_index: int,
    camera: str,
    task: str,
    entity: EntitySpec,
    detection: Detection,
    model: str,
    frame_stride: int,
    sampled_count: int,
    candidate_rank: int,
    selected: bool,
    signature: str,
    responses: list[dict],
) -> dict:
    """Create one compatible detection row with LocateAnything provenance."""
    row = detection_row(
        episode_index,
        camera,
        task,
        entity,
        detection,
        model,
        frame_stride,
        sampled_count,
        candidate_rank,
        selected,
    )
    row.update(
        {
            "detector": "nvidia/LocateAnything-3B",
            "configuration_signature": signature,
            "model_answer": response_answer(responses, entity.object_id, detection.frame_index),
        }
    )
    return row


def resolve_device(value: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available."""
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    """Resolve the requested inference precision."""
    if value == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local directory.")
    parser.add_argument(
        "--episodes",
        default="0",
        help="all, comma-separated zero-based indices, or end-exclusive ranges such as 0:20,156",
    )
    parser.add_argument("--cameras", nargs="+", choices=SUPPORTED_CAMERAS, default=list(SUPPORTED_CAMERAS))
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument(
        "--max-sampled-frames",
        type=int,
        default=0,
        help="Maximum sampled frames per episode-camera; 0 scans the complete episode.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-quality", type=float, default=MIN_QUALITY)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--generation-mode", choices=("fast", "slow", "hybrid"), default="hybrid")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose-model", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if args.max_sampled_frames < 0:
        raise ValueError("--max-sampled-frames cannot be negative")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if not 0.0 <= args.min_quality <= 1.0:
        raise ValueError("--min-quality must be between 0 and 1")

    source = EpisodeVideoSource(args.dataset_root)
    episode_indices = parse_episode_selection(args.episodes, source.episode_indices)
    task_entities = {
        episode_index: normalize_entities(entities_from_task(source.task(episode_index)))
        for episode_index in episode_indices
    }
    output_root = (
        args.output_root
        if args.output_root is not None
        else WORKSPACE / "runs" / "sam" / args.dataset_root.resolve().name / "locate_anything"
    ).resolve()
    setup_logging(output_root, "locate_anything.log")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    signature = configuration_signature(args)
    metadata = {
        "format_version": 1,
        "role": "LocateAnything initialization proposals for SAM2",
        "dataset_root": str(source.root),
        "model": args.model,
        "episodes": episode_indices,
        "cameras": args.cameras,
        "frame_stride": args.frame_stride,
        "max_sampled_frames": args.max_sampled_frames,
        "top_k": args.top_k,
        "min_quality": args.min_quality,
        "generation_mode": args.generation_mode,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "dtype": str(dtype),
        "seed": args.seed,
        "configuration_signature": signature,
        "object_id_scope": "episode; IDs are identical across cameras for the same episode",
        "quality_score": "Geometric view quality; LocateAnything does not expose box confidence.",
    }
    write_json_atomic(output_root / "meta" / "info.json", metadata)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    logger.info("Loading %s on %s with %s", args.model, device, dtype)
    worker = LocateAnythingWorker(args.model, device, dtype, args.local_files_only)

    total_jobs = len(episode_indices) * len(args.cameras)
    completed_jobs = 0
    for episode_index in episode_indices:
        task = source.task(episode_index)
        entities = task_entities[episode_index]
        for camera in args.cameras:
            paths = locate_result_paths(output_root, episode_index, camera)
            if output_is_current(paths, signature) and not args.overwrite:
                completed_jobs += 1
                logger.info(
                    "[%d/%d] Skipping episode %d camera %s; current output exists",
                    completed_jobs,
                    total_jobs,
                    episode_index,
                    camera,
                )
                continue

            logger.info(
                "[%d/%d] Locating episode %d camera %s: %s",
                completed_jobs + 1,
                total_jobs,
                episode_index,
                camera,
                task,
            )
            detections, detection_frames, ranked, fallback_frame, sampled_count, responses = (
                ground_episode_camera(source, episode_index, camera, entities, worker, args)
            )

            records = []
            candidate_records = []
            for entity, selected_detection in zip(entities, detections, strict=True):
                candidates = ranked[entity.object_id]
                selected_rank = next(
                    (
                        rank
                        for rank, (candidate, _) in enumerate(candidates, start=1)
                        if candidate == selected_detection
                    ),
                    1,
                )
                records.append(
                    make_record(
                        episode_index,
                        camera,
                        task,
                        entity,
                        selected_detection,
                        args.model,
                        args.frame_stride,
                        sampled_count,
                        selected_rank,
                        selected_detection.visible,
                        signature,
                        responses,
                    )
                )
                candidate_detections = [item[0] for item in candidates] or [selected_detection]
                for candidate_rank, detection in enumerate(candidate_detections, start=1):
                    candidate_records.append(
                        make_record(
                            episode_index,
                            camera,
                            task,
                            entity,
                            detection,
                            args.model,
                            args.frame_stride,
                            sampled_count,
                            candidate_rank,
                            selected_detection.visible and detection == selected_detection,
                            signature,
                            responses,
                        )
                    )

            parquet_path, candidates_path, json_path, preview_path, candidate_preview_path, response_path = paths
            write_parquet_atomic(parquet_path, pd.DataFrame(records))
            write_parquet_atomic(candidates_path, pd.DataFrame(candidate_records))
            write_json_atomic(json_path, {"selected": records, "candidates": candidate_records})
            write_json_atomic(response_path, responses)
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
                ranked,
                fallback_frame,
                args.top_k,
            )
            completed_jobs += 1
            found_count = sum(detection.visible for detection in detections)
            logger.info(
                "[%d/%d] Saved %d/%d objects after %d sampled frames and %d model queries",
                completed_jobs,
                total_jobs,
                found_count,
                len(detections),
                sampled_count,
                len(responses),
            )

    rebuild_summary(output_root, "NVIDIA LocateAnything-3B")
    logger.info("Detection table: %s", output_root / "meta" / "detections.parquet")
    logger.info("Raw responses: %s", output_root / "responses")
    logger.info("Inspection index: %s", output_root / "inspection" / "index.html")


if __name__ == "__main__":
    main()
