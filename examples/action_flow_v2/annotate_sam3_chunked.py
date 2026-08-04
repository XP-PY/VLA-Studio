#!/usr/bin/env python3
"""Create full-frame SAM3.1 relevance masks with bounded GPU memory.

Unlike ``annotate_sam3.py``, this converter never loads a complete episode into
one SAM3 session. It divides each camera video into overlapping chunks, runs
SAM3 on every frame in each chunk, and carries target identity between adjacent
chunks with a point sampled from the overlap mask. Reviewed YOLOE points seed
the first chunk for each entity; text-only grounding is used only when overlap
continuity is unavailable.

Frames for which SAM3 does not return a valid result are not filled with an old
mask. Their merged ``loss_weight`` is set to zero so they can be excluded from
the relevance loss.

Example:

    conda run -n sam3 python examples/action_flow_v2/annotate_sam3_chunked.py \
        --episodes all \
        --cameras top wrist \
        --chunk-size 32 \
        --chunk-overlap 16 \
        --resume

or

    mkdir -p logs

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    conda run --no-capture-output -n sam3 python \
        examples/action_flow_v2/annotate_sam3_chunked.py \
        --episodes all \
        --cameras top wrist \
        --chunk-size 32 \
        --chunk-overlap 16 \
        --resume \
        2>&1 | tee -a logs/annotate_sam3_chunked.log
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import annotate_sam3 as base
import numpy as np
import pandas as pd
import torch
from PIL import Image

DEFAULT_SAM_ROOT = Path(
    "runs/sam/rm65_single_table_preparation_and_bussing_yoloe_sam3_1_full_frames"
)
DEFAULT_OUTPUT_ROOT = Path(
    "runs/dataset/rm65_single_table_preparation_and_bussing_afv2_yoloe_sam3_1_full_frames"
)

logger = logging.getLogger(__name__)


class PromptUnavailable(RuntimeError):
    """Raised when a chunk has no usable semantic or point prompt."""


@dataclass(frozen=True)
class ChunkSpec:
    index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def center(self) -> float:
        return (self.start + self.end - 1) / 2

    def contains(self, frame_index: int) -> bool:
        return self.start <= frame_index < self.end


@dataclass
class ChunkResult:
    masks: list[np.ndarray]
    valid: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class EntityTrack:
    masks: list[np.ndarray]
    valid: np.ndarray
    diagnostics: dict[str, Any]


def make_chunk_specs(frame_count: int, chunk_size: int, overlap: int) -> list[ChunkSpec]:
    """Cover a video with overlapping chunks, including the final frame."""
    if frame_count < 1:
        raise ValueError("A video must contain at least one frame")
    if chunk_size < 2:
        raise ValueError("--chunk-size must be at least 2")
    if not 0 <= overlap < chunk_size:
        raise ValueError("--chunk-overlap must satisfy 0 <= overlap < chunk_size")
    if frame_count <= chunk_size:
        return [ChunkSpec(0, 0, frame_count)]

    step = chunk_size - overlap
    starts = list(range(0, frame_count - chunk_size + 1, step))
    final_start = frame_count - chunk_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return [
        ChunkSpec(index=index, start=start, end=min(start + chunk_size, frame_count))
        for index, start in enumerate(starts)
    ]


def parse_episode_indices(specification: str, available: set[int]) -> list[int]:
    """Parse ``all`` or comma-separated indices and inclusive ranges."""
    if specification.strip().lower() == "all":
        return sorted(available)

    selected: set[int] = set()
    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid episode range {token!r}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    missing = selected - available
    if missing:
        raise ValueError(f"Dataset does not contain episodes {sorted(missing)}")
    if not selected:
        raise ValueError("--episodes did not select any episodes")
    return sorted(selected)


def prepare_chunk_directories(
    frames: list[np.ndarray], specs: list[ChunkSpec], root: Path
) -> dict[int, Path]:
    """Write each chunk as the numbered JPEG directory expected by SAM3."""
    shutil.rmtree(root, ignore_errors=True)
    directories = {}
    for spec in specs:
        directory = root / f"chunk-{spec.index:04d}-{spec.start:06d}-{spec.end:06d}"
        directory.mkdir(parents=True)
        for local_index, frame in enumerate(frames[spec.start : spec.end]):
            Image.fromarray(frame).save(
                directory / f"{local_index:06d}.jpg",
                quality=95,
                subsampling=0,
            )
        directories[spec.index] = directory
    return directories


def point_from_mask(mask: np.ndarray) -> tuple[int, int] | None:
    """Return an interior mask pixel nearest to the mask centroid."""
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return None
    center = coordinates.mean(axis=0)
    index = int(np.argmin(np.square(coordinates - center).sum(axis=1)))
    y, x = coordinates[index]
    return int(x), int(y)


def best_instance(outputs: dict[str, Any]) -> int | None:
    """Choose the highest-confidence text-grounded instance."""
    object_ids, masks = base.normalize_masks(outputs)
    if not len(object_ids):
        return None
    scores = outputs.get("out_probs", outputs.get("out_scores"))
    if scores is not None:
        score_array = np.asarray(scores).reshape(-1)
        if len(score_array) == len(object_ids):
            return int(object_ids[int(np.argmax(score_array))])
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    return int(object_ids[int(np.argmax(areas))])


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if not union:
        return 0.0
    return float(np.logical_and(left, right).sum() / union)


def select_tracking_mask_strict(
    outputs: dict[str, Any],
    preferred_object_id: int,
    previous_mask: np.ndarray,
    shape: tuple[int, int],
    minimum_iou: float,
) -> tuple[np.ndarray, list[int]]:
    """Select a tracked mask without switching to an unrelated instance."""
    object_ids, masks = base.normalize_masks(outputs)
    seen_ids = [int(value) for value in object_ids]
    if not len(masks):
        return np.zeros(shape, dtype=bool), seen_ids

    nonempty = np.flatnonzero(masks.reshape(len(masks), -1).any(axis=1))
    if not len(nonempty):
        return np.zeros(shape, dtype=bool), seen_ids

    if not np.any(previous_mask):
        matches = np.flatnonzero(object_ids == preferred_object_id)
        if len(matches):
            return masks[int(matches[0])], seen_ids
        return np.zeros(shape, dtype=bool), seen_ids

    ious = np.asarray([mask_iou(masks[index], previous_mask) for index in nonempty])
    best = int(nonempty[int(np.argmax(ious))])
    if float(ious.max()) < minimum_iou:
        return np.zeros(shape, dtype=bool), seen_ids
    return masks[best], seen_ids


def close_session(predictor: Any, session_id: str | None) -> None:
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


def start_prompted_session(
    predictor: Any,
    frame_directory: Path,
    entity: base.EntityPrompt,
    anchor_frame: int,
    point_xy: tuple[int, int] | None,
    frame_shape: tuple[int, int],
    output_prob_threshold: float,
) -> tuple[str, int, np.ndarray, dict[str, Any]]:
    """Start a chunk session and identify one target instance at its anchor."""
    response = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(frame_directory),
            "offload_video_to_cpu": True,
        }
    )
    session_id = str(response["session_id"])
    selected_id: int | None = None
    selected_query: str | None = None
    detected_ids: list[int] = []
    queries_tried: list[str] = []
    text_outputs: dict[str, Any] = {}
    try:
        for query in base.semantic_query_candidates(entity):
            queries_tried.append(query)
            prompt_response = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": anchor_frame,
                    "text": query,
                    "output_prob_thresh": output_prob_threshold,
                }
            )
            text_outputs = prompt_response["outputs"]
            detected_ids = [int(value) for value in base.normalize_masks(text_outputs)[0]]
            selected_id = (
                base.select_instance_at_point(text_outputs, point_xy)
                if point_xy is not None
                else best_instance(text_outputs)
            )
            if selected_id is not None:
                selected_query = query
                break

        mode = "text-selected by continuity point" if point_xy is not None else "text-best-instance"
        if selected_id is not None:
            for other_id in detected_ids:
                if other_id != selected_id:
                    predictor.handle_request(
                        {
                            "type": "remove_object",
                            "session_id": session_id,
                            "frame_index": anchor_frame,
                            "obj_id": other_id,
                        }
                    )
            prompt_mask = base.mask_for_object(text_outputs, selected_id, frame_shape)
        elif point_xy is not None:
            predictor.handle_request({"type": "reset_session", "session_id": session_id})
            x, y = point_xy
            height, width = frame_shape
            point_response = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": anchor_frame,
                    "points": [[x / width, y / height]],
                    "point_labels": [1],
                    "obj_id": 0,
                    "rel_coordinates": True,
                    "output_prob_thresh": output_prob_threshold,
                }
            )
            point_outputs = point_response["outputs"]
            selected_id = base.select_instance_at_point(point_outputs, point_xy)
            if selected_id is None:
                selected_id = best_instance(point_outputs)
            if selected_id is None:
                raise PromptUnavailable("point prompt returned no instance")
            prompt_mask = base.mask_for_object(point_outputs, selected_id, frame_shape)
            mode = "point-only fallback"
        else:
            raise PromptUnavailable("text grounding returned no instance")

        if not np.any(prompt_mask):
            raise PromptUnavailable("prompt returned an empty mask")
        diagnostics = {
            "mode": mode,
            "selected_query": selected_query,
            "queries_tried": queries_tried,
            "detected_ids": detected_ids,
            "selected_sam3_object_id": selected_id,
        }
        return session_id, selected_id, prompt_mask, diagnostics
    except Exception:
        close_session(predictor, session_id)
        raise


def expected_tracking_error(error: Exception) -> bool:
    message = str(error)
    return (
        "list index out of range" in message
        or "existing size (0)" in message
        or "out of bounds" in message
    )


def track_chunk(
    predictor: Any,
    frame_directory: Path,
    frame_count: int,
    entity: base.EntityPrompt,
    anchor_frame: int,
    point_xy: tuple[int, int] | None,
    frame_shape: tuple[int, int],
    output_prob_threshold: float,
    minimum_tracking_iou: float,
) -> ChunkResult:
    """Track one entity through every frame of one chunk."""
    masks = [np.zeros(frame_shape, dtype=bool) for _ in range(frame_count)]
    valid = np.zeros(frame_count, dtype=bool)
    directions = {
        "forward": frame_count - 1 - anchor_frame,
        "backward": anchor_frame,
    }
    direction_diagnostics = []
    if not any(distance > 0 for distance in directions.values()):
        directions = {"forward": 0}

    for direction, max_frames in directions.items():
        if max_frames < 1 and valid[anchor_frame]:
            continue
        session_id = None
        try:
            session_id, selected_id, prompt_mask, prompt_diagnostics = start_prompted_session(
                predictor,
                frame_directory,
                entity,
                anchor_frame,
                point_xy,
                frame_shape,
                output_prob_threshold,
            )
            if np.any(prompt_mask) or not valid[anchor_frame]:
                masks[anchor_frame] = prompt_mask
            valid[anchor_frame] = True
            diagnostic = {"direction": direction, **prompt_diagnostics, "warning": None}
            if max_frames > 0:
                previous_mask = prompt_mask
                try:
                    for propagation in predictor.handle_stream_request(
                        {
                            "type": "propagate_in_video",
                            "session_id": session_id,
                            "propagation_direction": direction,
                            "start_frame_index": anchor_frame,
                            "max_frame_num_to_track": max_frames,
                            "output_prob_thresh": output_prob_threshold,
                        }
                    ):
                        frame_index = int(propagation["frame_index"])
                        mask, _ = select_tracking_mask_strict(
                            propagation["outputs"],
                            selected_id,
                            previous_mask,
                            frame_shape,
                            minimum_tracking_iou,
                        )
                        if not np.any(mask):
                            continue
                        masks[frame_index] = mask
                        valid[frame_index] = True
                        previous_mask = mask
                except (IndexError, RuntimeError) as error:
                    if isinstance(error, torch.OutOfMemoryError) or not expected_tracking_error(error):
                        raise
                    diagnostic["warning"] = f"{type(error).__name__}: {error}"
            direction_diagnostics.append(diagnostic)
        except PromptUnavailable as error:
            direction_diagnostics.append(
                {
                    "direction": direction,
                    "mode": "unavailable",
                    "warning": str(error),
                }
            )
        finally:
            close_session(predictor, session_id)

    return ChunkResult(
        masks=masks,
        valid=valid,
        diagnostics={
            "anchor_frame": anchor_frame,
            "point_xy": list(point_xy) if point_xy is not None else None,
            "valid_fraction": float(valid.mean()),
            "directions": direction_diagnostics,
        },
    )


def continuity_anchor(
    current: ChunkSpec,
    neighbor: ChunkSpec,
    neighbor_result: ChunkResult,
) -> tuple[int, tuple[int, int]] | None:
    """Choose a nonempty overlap mask to seed an adjacent chunk."""
    overlap_start = max(current.start, neighbor.start)
    overlap_end = min(current.end, neighbor.end)
    if overlap_start >= overlap_end:
        return None
    midpoint = (overlap_start + overlap_end - 1) / 2
    candidates = []
    for global_frame in range(overlap_start, overlap_end):
        neighbor_local = global_frame - neighbor.start
        mask = neighbor_result.masks[neighbor_local]
        if neighbor_result.valid[neighbor_local] and np.any(mask):
            candidates.append((abs(global_frame - midpoint), global_frame, mask))
    if not candidates:
        return None
    _, global_frame, mask = min(candidates, key=lambda item: item[0])
    point = point_from_mask(mask)
    if point is None:
        return None
    return global_frame, point


def track_entity_chunked(
    predictor: Any,
    frames: list[np.ndarray],
    specs: list[ChunkSpec],
    directories: dict[int, Path],
    entity: base.EntityPrompt,
    frame_shape: tuple[int, int],
    output_prob_threshold: float,
    minimum_tracking_iou: float,
) -> EntityTrack:
    """Track one entity from its reviewed seed chunk toward both video ends."""
    frame_count = len(frames)
    if entity.not_visible:
        return EntityTrack(
            masks=[np.zeros(frame_shape, dtype=bool) for _ in frames],
            valid=np.ones(frame_count, dtype=bool),
            diagnostics={"mode": "reviewed_not_visible", "valid_fraction": 1.0, "chunks": []},
        )
    assert entity.frame_index is not None and entity.point_xy is not None

    containing = [spec for spec in specs if spec.contains(entity.frame_index)]
    seed = min(containing, key=lambda spec: abs(spec.center - entity.frame_index))
    order = [seed.index]
    order.extend(range(seed.index + 1, len(specs)))
    order.extend(range(seed.index - 1, -1, -1))
    chunk_results: dict[int, ChunkResult] = {}
    chunk_diagnostics = []

    for chunk_index in order:
        spec = specs[chunk_index]
        if chunk_index == seed.index:
            global_anchor = entity.frame_index
            point_xy = entity.point_xy
            anchor_source = entity.source
        else:
            neighbor_index = chunk_index - 1 if chunk_index > seed.index else chunk_index + 1
            continuity = continuity_anchor(
                spec,
                specs[neighbor_index],
                chunk_results[neighbor_index],
            )
            if continuity is not None:
                global_anchor, point_xy = continuity
                anchor_source = f"chunk {neighbor_index} overlap"
            elif spec.contains(entity.frame_index):
                global_anchor = entity.frame_index
                point_xy = entity.point_xy
                anchor_source = entity.source
            else:
                global_anchor = round(spec.center)
                point_xy = None
                anchor_source = "text-only reacquisition"

        result = track_chunk(
            predictor,
            directories[chunk_index],
            spec.length,
            entity,
            global_anchor - spec.start,
            point_xy,
            frame_shape,
            output_prob_threshold,
            minimum_tracking_iou,
        )
        chunk_results[chunk_index] = result
        chunk_diagnostics.append(
            {
                "chunk_index": chunk_index,
                "start": spec.start,
                "end": spec.end,
                "global_anchor": global_anchor,
                "anchor_source": anchor_source,
                **result.diagnostics,
            }
        )

    masks = [np.zeros(frame_shape, dtype=bool) for _ in frames]
    valid = np.zeros(frame_count, dtype=bool)
    centrality = np.full(frame_count, -1, dtype=np.int32)
    for spec in specs:
        result = chunk_results[spec.index]
        for local_frame in range(spec.length):
            if not result.valid[local_frame]:
                continue
            global_frame = spec.start + local_frame
            score = min(local_frame, spec.length - 1 - local_frame)
            if score > centrality[global_frame]:
                masks[global_frame] = result.masks[local_frame]
                valid[global_frame] = True
                centrality[global_frame] = score

    return EntityTrack(
        masks=masks,
        valid=valid,
        diagnostics={
            "mode": "chunked",
            "seed_chunk": seed.index,
            "valid_fraction": float(valid.mean()),
            "chunks": sorted(chunk_diagnostics, key=lambda item: item["chunk_index"]),
        },
    )


def save_outputs(
    sam_root: Path,
    output_root: Path,
    episode_index: int,
    camera: str,
    tracks: dict[int, EntityTrack],
    entities: list[base.EntityPrompt],
) -> np.ndarray:
    """Save per-entity masks and merged relevance masks with validity weights."""
    labels = {entity.object_id: entity.object_name for entity in entities}
    frame_count = len(next(iter(tracks.values())).masks)
    height, width = next(iter(tracks.values())).masks[0].shape
    object_rows = []
    for object_id, track in sorted(tracks.items()):
        for frame_index, (mask, valid) in enumerate(zip(track.masks, track.valid, strict=True)):
            object_rows.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "camera": camera,
                    "object_id": object_id,
                    "object_label": labels[object_id],
                    "height": height,
                    "width": width,
                    "mask_runs": base.encode_mask_runs(mask),
                    "valid": bool(valid),
                }
            )
    object_path = sam_root / "data" / f"episode-{episode_index:06d}" / f"camera-{camera}.parquet"
    base.write_parquet_atomic(object_path, pd.DataFrame(object_rows))

    merged_masks = []
    loss_weight = np.ones(frame_count, dtype=np.float32)
    valid_fraction = np.zeros(frame_count, dtype=np.float32)
    for frame_index in range(frame_count):
        frame_valid = [bool(track.valid[frame_index]) for track in tracks.values()]
        valid_fraction[frame_index] = float(np.mean(frame_valid))
        if not all(frame_valid):
            loss_weight[frame_index] = 0.0
        merged = np.zeros((height, width), dtype=bool)
        for track in tracks.values():
            if track.valid[frame_index]:
                merged |= track.masks[frame_index]
        merged_masks.append(merged)

    merged_table = pd.DataFrame(
        {
            "episode_index": np.full(frame_count, episode_index, dtype=np.int64),
            "frame_index": np.arange(frame_count, dtype=np.int64),
            "camera": np.full(frame_count, camera, dtype=object),
            "height": np.full(frame_count, height, dtype=np.int32),
            "width": np.full(frame_count, width, dtype=np.int32),
            "mask_runs": [base.encode_mask_runs(mask) for mask in merged_masks],
            "loss_weight": loss_weight,
            "valid_entity_fraction": valid_fraction,
        }
    )
    merged_path = output_root / "data" / f"episode-{episode_index:06d}" / f"camera-{camera}.parquet"
    base.write_parquet_atomic(merged_path, merged_table)
    return loss_weight


def outputs_complete(
    source: base.EpisodeVideoSource,
    sam_root: Path,
    output_root: Path,
    episode_index: int,
    camera: str,
    expected_object_ids: set[int],
) -> bool:
    expected_frames = int(source.episodes.loc[episode_index, "length"])
    object_path = sam_root / "data" / f"episode-{episode_index:06d}" / f"camera-{camera}.parquet"
    merged_path = output_root / "data" / f"episode-{episode_index:06d}" / f"camera-{camera}.parquet"
    if not object_path.is_file() or not merged_path.is_file():
        return False
    objects = pd.read_parquet(object_path, columns=["object_id", "frame_index", "valid"])
    merged = pd.read_parquet(merged_path, columns=["frame_index", "loss_weight"])
    sizes = objects.groupby("object_id").size()
    return (
        len(merged) == expected_frames
        and {int(object_id) for object_id in sizes.index} == expected_object_ids
        and bool((sizes == expected_frames).all())
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=base.DEFAULT_DATASET_ROOT)
    parser.add_argument("--yoloe-root", type=Path, default=base.DEFAULT_YOLOE_ROOT)
    parser.add_argument("--yoloe-review", type=Path, default=base.DEFAULT_YOLOE_REVIEW)
    parser.add_argument("--sam-root", type=Path, default=DEFAULT_SAM_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=base.DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--episodes",
        default="all",
        help="all or comma-separated indices/inclusive ranges, for example 0,3,8-12",
    )
    parser.add_argument("--cameras", nargs="+", choices=base.CAMERAS, default=list(base.CAMERAS))
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Frames loaded by each SAM3 session (default: 32)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=16,
        help="Shared frames used to carry identity between chunks (default: 16)",
    )
    parser.add_argument("--output-prob-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-tracking-iou", type=float, default=0.01)
    parser.add_argument("--object-ids", nargs="+", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-previews", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-chunk-frames", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit pairs for a smoke test; 0 means all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    if not 0.0 <= args.minimum_tracking_iou <= 1.0:
        raise ValueError("--minimum-tracking-iou must be in [0, 1]")
    make_chunk_specs(max(args.chunk_size + 1, 2), args.chunk_size, args.chunk_overlap)

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    source = base.EpisodeVideoSource(args.dataset_root)
    review = base.YOLOEReview(args.yoloe_root, args.yoloe_review)
    episode_indices = parse_episode_indices(args.episodes, source.episode_indices)
    pairs = [(episode, camera) for episode in episode_indices for camera in args.cameras]
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    sam_root = args.sam_root.resolve()
    output_root = args.output_root.resolve()
    temporary_root = sam_root / "chunk_frames"
    metadata = {
        "format_version": 1,
        "model": "facebook/sam3.1",
        "checkpoint": str(checkpoint),
        "base_dataset_root": str(source.root),
        "yoloe_root": str(review.root),
        "yoloe_review": str(review.review_path),
        "yoloe_review_sha256": review.review_sha256,
        "frame_stride": 1,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "output_prob_threshold": args.output_prob_threshold,
        "minimum_tracking_iou": args.minimum_tracking_iou,
        "invalid_frame_policy": "loss_weight=0; masks are never extrapolated",
    }
    base.write_json_atomic(
        sam_root / "meta" / "info.json", {**metadata, "output_root": str(output_root)}
    )
    base.write_json_atomic(
        output_root / "meta" / "info.json", {**metadata, "sam_root": str(sam_root)}
    )

    report_path = sam_root / "meta" / "sam3_chunked_report.json"
    report: dict[str, Any] = {"pairs": [], "failed": []}
    previous_pairs: dict[tuple[int, str], dict[str, Any]] = {}
    if report_path.is_file():
        previous = json.loads(report_path.read_text())
        previous_pairs = {
            (int(item["episode_index"]), str(item["camera"])): item
            for item in previous.get("pairs", [])
        }
        selected = set(pairs)
        report["pairs"] = [
            item
            for item in previous.get("pairs", [])
            if (int(item["episode_index"]), str(item["camera"])) not in selected
        ]
        report["failed"] = [
            item
            for item in previous.get("failed", [])
            if (int(item["episode_index"]), str(item["camera"])) not in selected
        ]

    predictor = None
    for pair_number, (episode_index, camera) in enumerate(pairs, start=1):
        pair = {"episode_index": episode_index, "camera": camera}
        entities = review.resolve(episode_index, camera)
        if args.object_ids:
            selected_ids = set(args.object_ids)
            entities = [entity for entity in entities if entity.object_id in selected_ids]
            missing = selected_ids - {entity.object_id for entity in entities}
            if missing:
                raise ValueError(f"Pair {pair} does not contain object IDs {sorted(missing)}")
        if not entities:
            raise ValueError(f"Pair {pair} does not contain any target entities")
        expected_object_ids = {entity.object_id for entity in entities}
        if (
            args.resume
            and not args.overwrite
            and outputs_complete(
                source,
                sam_root,
                output_root,
                episode_index,
                camera,
                expected_object_ids,
            )
        ):
            logger.info("[%d/%d] Skipping complete %s", pair_number, len(pairs), pair)
            existing = previous_pairs.get((episode_index, camera), {})
            report["pairs"].append({**existing, **pair, "status": "skipped_existing"})
            base.write_json_atomic(report_path, report)
            continue

        pair_temporary_root = temporary_root / f"episode-{episode_index:06d}-camera-{camera}"
        logger.info(
            "[%d/%d] Processing episode %d camera %s",
            pair_number,
            len(pairs),
            episode_index,
            camera,
        )
        try:
            frames = source.frames(episode_index, camera)
            frame_shape = frames[0].shape[:2]
            specs = make_chunk_specs(len(frames), args.chunk_size, args.chunk_overlap)
            directories = prepare_chunk_directories(frames, specs, pair_temporary_root)

            if predictor is None and any(not entity.not_visible for entity in entities):
                logger.info("Loading SAM3.1 from %s", checkpoint)
                predictor = base.build_sam3_predictor(
                    checkpoint_path=str(checkpoint),
                    version="sam3.1",
                    compile=False,
                    max_num_objects=16,
                    multiplex_count=16,
                    use_fa3=False,
                    use_rope_real=True,
                    async_loading_frames=False,
                )

            tracks = {}
            entity_diagnostics = []
            for entity_number, entity in enumerate(entities, start=1):
                logger.info(
                    "Episode %d %s entity %d/%d: %s",
                    episode_index,
                    camera,
                    entity_number,
                    len(entities),
                    entity.object_name,
                )
                track = track_entity_chunked(
                    predictor,
                    frames,
                    specs,
                    directories,
                    entity,
                    frame_shape,
                    args.output_prob_threshold,
                    args.minimum_tracking_iou,
                )
                tracks[entity.object_id] = track
                entity_diagnostics.append(
                    {
                        "object_id": entity.object_id,
                        "object_name": entity.object_name,
                        "object_role": entity.object_role,
                        **track.diagnostics,
                    }
                )

            loss_weight = save_outputs(
                sam_root,
                output_root,
                episode_index,
                camera,
                tracks,
                entities,
            )
            if args.write_previews:
                base.write_preview_video(
                    sam_root / "previews" / f"episode-{episode_index:06d}-camera-{camera}.mp4",
                    frames,
                    {object_id: track.masks for object_id, track in tracks.items()},
                    entities,
                    source.fps,
                )
            diagnostic = {
                **pair,
                "task": source.prompt(episode_index),
                "frame_count": len(frames),
                "frame_stride": 1,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "chunk_count": len(specs),
                "valid_loss_fraction": float(loss_weight.mean()),
                "entities": entity_diagnostics,
            }
            base.write_json_atomic(
                sam_root
                / "diagnostics"
                / f"episode-{episode_index:06d}-camera-{camera}.json",
                diagnostic,
            )
            report["pairs"].append({**pair, "status": "processed", **diagnostic})
        except Exception as error:
            logger.exception("SAM3 chunked conversion failed for %s", pair)
            report["failed"].append({**pair, "error": f"{type(error).__name__}: {error}"})
        finally:
            if not args.keep_chunk_frames:
                shutil.rmtree(pair_temporary_root, ignore_errors=True)
            base.write_json_atomic(report_path, report)

    logger.info(
        "SAM3 full-frame conversion finished: %d successful/skipped, %d failed. Report: %s",
        len(report["pairs"]),
        len(report["failed"]),
        report_path,
    )


if __name__ == "__main__":
    main()
