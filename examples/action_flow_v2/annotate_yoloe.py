#!/usr/bin/env python3
"""Initialize Action Flow V2 object tracks with YOLOE visual prompts.

The script reads task prompts from a LeRobot dataset, converts each prompt into
task-relevant entities, and detects only those entities using the visual
reference classes in ``yoloe_reference.yaml``. It derives one positive SAM2
point from each predicted segmentation mask, retains three ranked, temporally
separated candidate frames per entity by default, and writes an HTML inspection
page before any candidate is used to initialize SAM2.

Example:

python examples/action_flow_v2/annotate_yoloe.py \
    --dataset-config table_preparation \
    --episodes 0:3 \
    --max-sampled-frames 10 \
    --frame-stride 10 \
    --output-root runs/yoloe/rm65_single_table_preparation_and_bussing \
    --device 0 \
    --overwrite

or

python examples/action_flow_v2/annotate_yoloe.py \
    --dataset-config table_preparation \
    --episodes all \
    --all-frames \
    --output-root runs/yoloe/rm65_single_table_preparation_and_bussing \
    --device 0 \
    --overwrite \
    --conf 0.15 \
    --min-quality 0.3

python examples/action_flow_v2/annotate_yoloe.py \
    --dataset-config table_preparation \
    --episodes 36: \
    --all-frames \
    --output-root runs/yoloe/rm65_single_table_preparation_and_bussing_test \
    --device 0 \
    --overwrite \
    --conf 0.15 \
    --min-quality 0.3

Outputs default to ``runs/sam/<dataset-name>/yoloe``. Open
``inspection/index.html`` to review selected boxes, positive points, and all
three candidates.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    from .annotate_grounding import (
        Detection,
        EpisodeVideoSource,
        detection_row,
        missing_detection,
        parse_episode_selection,
        rebuild_summary,
        result_is_current,
        result_paths,
        retain_ranked_candidate,
        score_detection_view,
        setup_logging,
        write_candidate_contact_sheet,
        write_contact_sheet,
        write_json_atomic,
        write_parquet_atomic,
    )
    from .grounding_configs import (
        DATASET_CONFIGS,
        TABLE_PREPARATION_CONFIG,
        EntitySpec,
        GroundingDatasetConfig,
        get_dataset_config,
    )
except ImportError:
    from annotate_grounding import (  # type: ignore[no-redef]
        Detection,
        EpisodeVideoSource,
        detection_row,
        missing_detection,
        parse_episode_selection,
        rebuild_summary,
        result_is_current,
        result_paths,
        retain_ranked_candidate,
        score_detection_view,
        setup_logging,
        write_candidate_contact_sheet,
        write_contact_sheet,
        write_json_atomic,
        write_parquet_atomic,
    )
    from grounding_configs import (  # type: ignore[no-redef]
        DATASET_CONFIGS,
        TABLE_PREPARATION_CONFIG,
        EntitySpec,
        GroundingDatasetConfig,
        get_dataset_config,
    )


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_CONFIG = WORKSPACE / "examples" / "action_flow_v2" / "ref" / "yoloe_reference_realsense_viewer_with_collection_box.yaml"
# DEFAULT_REFERENCE_CONFIG = WORKSPACE / "examples" / "action_flow_v2" / "ref" / "yoloe_reference_lerobot.yaml"
DEFAULT_MODEL = WORKSPACE / ".cache" / "yoloe" / "weights" / "yoloe-26x-seg.pt"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReferenceObject:
    """One YOLOE visual-prompt class defined by a box in the reference image."""

    class_id: int
    name: str
    display_name: str
    bbox_xyxy: tuple[float, float, float, float]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class RawDetection:
    """One YOLOE box and optional segmentation mask before view scoring."""

    class_id: int
    score: float
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class PointDetection(Detection):
    """A ranked detection with one positive point in original-image coordinates."""

    point_x: int | None
    point_y: int | None
    point_source: str | None


def normalize_name(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


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


def box_center(
    box_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int]:
    x0, y0, x1, y1 = box_xyxy
    x = int(np.clip(round(0.5 * (x0 + x1)), 0, width - 1))
    y = int(np.clip(round(0.5 * (y0 + y1)), 0, height - 1))
    return x, y


def draw_positive_point(image: Image.Image, point_xy: tuple[int, int]) -> Image.Image:
    """Draw a high-contrast positive point that remains visible in review sheets."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    x, y = point_xy
    radius = 8
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=(255, 255, 255),
        outline=(220, 30, 30),
        width=3,
    )
    draw.line((x - 6, y, x + 6, y), fill=(0, 0, 0), width=2)
    draw.line((x, y - 6, x, y + 6), fill=(0, 0, 0), width=2)
    return annotated


def point_detection(
    detection: Detection,
    point_xy: tuple[int, int],
    point_source: str,
) -> PointDetection:
    return PointDetection(
        **asdict(detection),
        point_x=point_xy[0],
        point_y=point_xy[1],
        point_source=point_source,
    )


def missing_point_detection(width: int, height: int) -> PointDetection:
    return PointDetection(
        **asdict(missing_detection(width, height)),
        point_x=None,
        point_y=None,
        point_source=None,
    )


def point_detection_row(
    episode_index: int,
    camera: str,
    task: str,
    entity: EntitySpec,
    detection: PointDetection,
    model_path: str,
    frame_stride: int,
    sampled_frames: int,
    candidate_rank: int = 1,
    selected: bool | None = None,
) -> dict:
    row = detection_row(
        episode_index,
        camera,
        task,
        entity,
        detection,
        model_path,
        frame_stride,
        sampled_frames,
        candidate_rank,
        selected,
    )
    row.update(
        {
            "point_x": detection.point_x if detection.point_x is not None else np.nan,
            "point_y": detection.point_y if detection.point_y is not None else np.nan,
            "point_x_norm": (
                detection.point_x / detection.image_width
                if detection.point_x is not None and detection.image_width > 0
                else np.nan
            ),
            "point_y_norm": (
                detection.point_y / detection.image_height
                if detection.point_y is not None and detection.image_height > 0
                else np.nan
            ),
            "sam_label": np.int8(1) if detection.visible else np.nan,
            "point_source": detection.point_source,
        }
    )
    return row


def point_result_is_current(
    paths: tuple[Path, Path, Path, Path, Path],
    entities: list[EntitySpec],
) -> bool:
    """Reject old box-only outputs so point artifacts are regenerated."""
    if not result_is_current(paths, entities):
        return False
    required_columns = {"point_x", "point_y", "point_x_norm", "point_y_norm", "sam_label", "point_source"}
    try:
        return all(
            required_columns.issubset(pd.read_parquet(path).columns)
            for path in paths[:2]
        )
    except Exception:
        return False


def load_reference_config(config_path: Path) -> tuple[Path, list[ReferenceObject]]:
    """Load and validate the YOLOE visual-prompt image and class boxes."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read the YOLOE reference config") from exc

    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    config = yaml.safe_load(config_path.read_text())
    reference = config["reference"]
    reference_image = Path(reference["image"])
    if not reference_image.is_absolute():
        reference_image = config_path.parent / reference_image
    reference_image = reference_image.resolve()

    image = cv2.imread(str(reference_image))
    if image is None:
        raise RuntimeError(f"OpenCV could not read reference image: {reference_image}")
    image_height, image_width = image.shape[:2]
    configured_size = (int(reference["width"]), int(reference["height"]))
    if configured_size != (image_width, image_height):
        raise ValueError(
            f"Reference size mismatch: YAML={configured_size}, "
            f"image={(image_width, image_height)}"
        )

    objects = [
        ReferenceObject(
            class_id=int(item["class_id"]),
            name=str(item["name"]),
            display_name=str(item.get("display_name", item["name"])),
            bbox_xyxy=tuple(float(value) for value in item["bbox_xyxy"]),
            aliases=tuple(str(alias) for alias in item.get("aliases", ())),
        )
        for item in config["objects"]
    ]
    class_ids = [item.class_id for item in objects]
    if class_ids != list(range(len(objects))):
        raise ValueError(
            "YOLOE visual-prompt class IDs must be sequential and start at zero; "
            f"received {class_ids}"
        )

    seen_names: dict[str, int] = {}
    for item in objects:
        x0, y0, x1, y1 = item.bbox_xyxy
        if not (0 <= x0 < x1 <= image_width and 0 <= y0 < y1 <= image_height):
            raise ValueError(f"Invalid box for {item.name}: {item.bbox_xyxy}")
        for name in (item.name, item.display_name, *item.aliases):
            normalized = normalize_name(name)
            previous = seen_names.get(normalized)
            if previous is not None and previous != item.class_id:
                raise ValueError(
                    f"Reference name {name!r} maps to both class {previous} and "
                    f"class {item.class_id}"
                )
            seen_names[normalized] = item.class_id
    return reference_image, objects


class YOLOEVisualPromptDetector:
    """Run YOLOE after initializing its class embeddings from one reference image."""

    def __init__(
        self,
        model_path: str,
        reference_image: Path,
        objects: list[ReferenceObject],
        *,
        device: str,
        imgsz: int,
        confidence: float,
        iou: float,
    ) -> None:
        try:
            from ultralytics import YOLOE
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        except ImportError as exc:
            raise ImportError(
                "YOLOE requires Ultralytics. Install it with: "
                "python -m pip install -U ultralytics"
            ) from exc

        self.model = YOLOE(model_path)
        self.predictor_class = YOLOEVPSegPredictor
        self.reference_image = reference_image
        self.visual_prompts = {
            "bboxes": np.asarray([item.bbox_xyxy for item in objects], dtype=np.float32),
            "cls": np.asarray([item.class_id for item in objects], dtype=np.int64),
        }
        self.known_class_ids = {item.class_id for item in objects}
        self.device = device
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou = iou
        self.initialized = False

    def predict(self, images: list[Image.Image]) -> list[dict[int, list[RawDetection]]]:
        """Detect every configured visual-prompt class in a batch of RGB images."""
        if not images:
            return []
        bgr_images = [
            cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            for image in images
        ]
        prediction_args: dict[str, Any] = {
            "source": bgr_images,
            "imgsz": self.imgsz,
            "conf": self.confidence,
            "iou": self.iou,
            "device": self.device,
            "retina_masks": True,
            "verbose": False,
        }
        if not self.initialized:
            prediction_args.update(
                {
                    "refer_image": str(self.reference_image),
                    "visual_prompts": self.visual_prompts,
                    "predictor": self.predictor_class,
                }
            )

        results = self.model.predict(**prediction_args)
        self.initialized = True
        if len(results) != len(images):
            raise RuntimeError(
                f"YOLOE returned {len(results)} results for a batch of {len(images)} images"
            )

        batch_detections = []
        for result in results:
            detections: dict[int, list[RawDetection]] = {}
            if result.boxes is not None and len(result.boxes) > 0:
                class_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
                confidences = result.boxes.conf.detach().cpu().numpy()
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                masks = (
                    result.masks.data.detach().cpu().numpy()
                    if result.masks is not None
                    else None
                )
                raw_results = zip(
                    class_ids,
                    confidences,
                    boxes,
                    strict=True,
                )
                for detection_index, (class_id, score, box) in enumerate(raw_results):
                    class_id = int(class_id)
                    if class_id not in self.known_class_ids:
                        continue
                    detections.setdefault(class_id, []).append(
                        RawDetection(
                            class_id=class_id,
                            score=float(score),
                            box_xyxy=tuple(float(value) for value in box),
                            mask=(
                                masks[detection_index]
                                if masks is not None and detection_index < len(masks)
                                else None
                            ),
                        )
                    )
            for class_detections in detections.values():
                class_detections.sort(key=lambda item: item.score, reverse=True)
            batch_detections.append(detections)
        return batch_detections


class ReferenceResolver:
    """Map semantic task entities to YOLOE reference classes."""

    def __init__(
        self,
        objects: list[ReferenceObject],
        dataset_config: GroundingDatasetConfig,
    ) -> None:
        self.dataset_config = dataset_config
        self.class_id_by_name: dict[str, int] = {}
        for item in objects:
            for name in (item.name, item.display_name, *item.aliases):
                self.class_id_by_name[normalize_name(name)] = item.class_id

    def class_id(self, semantic_name: str) -> int | None:
        return self.class_id_by_name.get(normalize_name(semantic_name))

    def semantic_name_for_query(self, query: str) -> str:
        normalized_query = normalize_name(query)
        for semantic_name, detector_query in self.dataset_config.query_aliases.items():
            if normalize_name(detector_query) == normalized_query:
                return semantic_name
        return query

    def unresolved_names(self, entities: list[EntitySpec]) -> set[str]:
        names: set[str] = set()
        for entity in entities:
            if entity.relation_container_query is not None:
                relation_names = (
                    self.semantic_name_for_query(entity.relation_container_query),
                    *(
                        self.semantic_name_for_query(query)
                        for query in entity.relation_member_queries
                    ),
                )
                names.update(name for name in relation_names if self.class_id(name) is None)
            elif self.class_id(entity.object_name) is None:
                names.add(entity.object_name)
        return names

    @staticmethod
    def _ordered(
        detections: list[RawDetection],
        order: str,
    ) -> list[RawDetection]:
        if order == "confidence":
            return sorted(detections, key=lambda item: item.score, reverse=True)

        def center(detection: RawDetection) -> tuple[float, float]:
            x0, y0, x1, y1 = detection.box_xyxy
            return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

        if order == "top_to_bottom":
            return sorted(detections, key=lambda item: (center(item)[1], -item.score))
        if order == "bottom_to_top":
            return sorted(detections, key=lambda item: (-center(item)[1], -item.score))
        if order == "left_to_right":
            return sorted(detections, key=lambda item: (center(item)[0], -item.score))
        if order == "right_to_left":
            return sorted(detections, key=lambda item: (-center(item)[0], -item.score))
        raise ValueError(f"Unsupported candidate order: {order}")

    @staticmethod
    def _containment(inner: RawDetection, outer: RawDetection) -> float:
        ix0 = max(inner.box_xyxy[0], outer.box_xyxy[0])
        iy0 = max(inner.box_xyxy[1], outer.box_xyxy[1])
        ix1 = min(inner.box_xyxy[2], outer.box_xyxy[2])
        iy1 = min(inner.box_xyxy[3], outer.box_xyxy[3])
        intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        inner_area = max(1.0, inner.box_xyxy[2] - inner.box_xyxy[0]) * max(
            1.0, inner.box_xyxy[3] - inner.box_xyxy[1]
        )
        return intersection / inner_area

    def _resolve_relation(
        self,
        entity: EntitySpec,
        detections: dict[int, list[RawDetection]],
    ) -> RawDetection | None:
        container_name = self.semantic_name_for_query(entity.relation_container_query or "")
        container_id = self.class_id(container_name)
        if container_id is None or not detections.get(container_id):
            return None
        container = detections[container_id][0]

        members = []
        for query in entity.relation_member_queries:
            member_name = self.semantic_name_for_query(query)
            member_id = self.class_id(member_name)
            if member_id is not None:
                members.extend(detections.get(member_id, ()))
        if not members:
            return None

        ranked = [
            (self._containment(member, container), member.score, member)
            for member in members
        ]
        containment, _, selected = max(ranked, key=lambda item: (item[0], item[1]))
        return selected if containment >= 0.2 else None

    def resolve(
        self,
        entities: list[EntitySpec],
        detections: dict[int, list[RawDetection]],
    ) -> dict[int, RawDetection]:
        """Return at most one YOLOE detection for each task entity ID."""
        resolved: dict[int, RawDetection] = {}
        direct_groups: dict[
            tuple[str, str, str, int],
            list[EntitySpec],
        ] = {}
        for entity in entities:
            if entity.relation_container_query is not None:
                detection = self._resolve_relation(entity, detections)
                if detection is not None:
                    resolved[entity.object_id] = detection
                continue
            group_key = (
                entity.object_name,
                entity.cardinality,
                entity.candidate_order,
                entity.task_step if entity.cardinality == "single" else 0,
            )
            direct_groups.setdefault(group_key, []).append(entity)

        for group_entities in direct_groups.values():
            representative = group_entities[0]
            class_id = self.class_id(representative.object_name)
            if class_id is None:
                continue
            ordered = self._ordered(
                detections.get(class_id, []),
                representative.candidate_order,
            )
            for entity in sorted(group_entities, key=lambda item: item.instance_index):
                if entity.instance_index < len(ordered):
                    resolved[entity.object_id] = ordered[entity.instance_index]
        return resolved


def annotate_episode_camera(
    source: EpisodeVideoSource,
    episode_index: int,
    camera: str,
    entities: list[EntitySpec],
    detector: YOLOEVisualPromptDetector,
    resolver: ReferenceResolver,
    *,
    frame_stride: int,
    max_sampled_frames: int,
    batch_size: int,
    top_k: int,
    min_quality: float,
    mask_threshold: float,
) -> tuple[
    list[PointDetection],
    dict[int, Image.Image],
    dict[int, list[tuple[PointDetection, Image.Image]]],
    Image.Image | None,
    int,
]:
    """Scan one episode-camera and rank point-annotated candidates per entity."""
    ranked_candidates: dict[int, list[tuple[PointDetection, Image.Image]]] = {
        entity.object_id: [] for entity in entities
    }
    entities_by_id = {entity.object_id: entity for entity in entities}
    episode_length = int(source.row(episode_index)["length"])
    minimum_frame_gap = max(frame_stride * 2, 1)
    fallback_frame: Image.Image | None = None
    sampled_count = 0
    pending: list[tuple[int, float, Image.Image]] = []

    def process_batch(batch: list[tuple[int, float, Image.Image]]) -> None:
        nonlocal sampled_count, fallback_frame
        predictions = detector.predict([item[2] for item in batch])
        for (frame_index, timestamp_s, image), frame_predictions in zip(
            batch,
            predictions,
            strict=True,
        ):
            sampled_count += 1
            if fallback_frame is None:
                fallback_frame = image.copy()
            resolved = resolver.resolve(entities, frame_predictions)
            for object_id, raw_detection in resolved.items():
                point_xy = None
                point_source = "box_center"
                if raw_detection.mask is not None:
                    binary_mask = resize_binary_mask(
                        raw_detection.mask,
                        image.width,
                        image.height,
                        mask_threshold,
                    )
                    point_xy = mask_interior_point(binary_mask)
                    if point_xy is not None:
                        point_source = "mask_interior"
                if point_xy is None:
                    point_xy = box_center(
                        raw_detection.box_xyxy,
                        image.width,
                        image.height,
                    )
                detection = point_detection(
                    score_detection_view(
                        image,
                        frame_index,
                        timestamp_s,
                        raw_detection.score,
                        raw_detection.box_xyxy,
                        entities_by_id[object_id],
                        camera,
                        episode_length,
                    ),
                    point_xy,
                    point_source,
                )
                retain_ranked_candidate(
                    ranked_candidates[object_id],
                    detection,
                    draw_positive_point(image, point_xy),
                    top_k,
                    minimum_frame_gap,
                )

    for sample in source.sampled_frames(
        episode_index,
        camera,
        frame_stride,
        max_sampled_frames,
    ):
        pending.append(sample)
        if len(pending) == batch_size:
            process_batch(pending)
            pending = []
    if pending:
        process_batch(pending)

    width = fallback_frame.width if fallback_frame is not None else 0
    height = fallback_frame.height if fallback_frame is not None else 0
    accepted_candidates = {
        entity.object_id: next(
            (
                candidate
                for candidate in ranked_candidates[entity.object_id]
                if candidate[0].quality_score is not None
                and candidate[0].quality_score >= min_quality
            ),
            None,
        )
        for entity in entities
    }
    detections = [
        accepted_candidates[entity.object_id][0]
        if accepted_candidates[entity.object_id] is not None
        else missing_point_detection(width, height)
        for entity in entities
    ]
    detection_frames = {
        entity.object_id: accepted_candidates[entity.object_id][1]
        for entity in entities
        if accepted_candidates[entity.object_id] is not None
    }
    return detections, detection_frames, ranked_candidates, fallback_frame, sampled_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config",
        choices=sorted(DATASET_CONFIGS),
        default=TABLE_PREPARATION_CONFIG.name,
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--reference-config", type=Path, default=DEFAULT_REFERENCE_CONFIG)
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Local YOLOE checkpoint or an Ultralytics model name.",
    )
    parser.add_argument(
        "--episodes",
        default="all",
        help="all, comma-separated indices, or end-exclusive ranges such as 0:20,156",
    )
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Detect every episode frame instead of sampling with --frame-stride.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-sampled-frames", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--min-quality", type=float, default=0.4)
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="YOLOE mask probability threshold used to derive positive points.",
    )
    parser.add_argument("--device", default="0", help="Ultralytics device, such as 0 or cpu.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_sampled_frames < 0:
        raise ValueError("--max-sampled-frames must be non-negative")
    if args.all_frames and args.max_sampled_frames > 0:
        raise ValueError("--all-frames requires --max-sampled-frames=0")
    if args.imgsz < 1:
        raise ValueError("--imgsz must be positive")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("--conf and --iou must be between 0 and 1")
    if not 0 <= args.mask_threshold <= 1:
        raise ValueError("--mask-threshold must be between 0 and 1")

    dataset_config = get_dataset_config(args.dataset_config)
    dataset_root = (args.dataset_root or dataset_config.default_dataset_root).resolve()
    source = EpisodeVideoSource(dataset_root)
    cameras = tuple(args.cameras) if args.cameras is not None else dataset_config.cameras
    missing_cameras = sorted(set(cameras).difference(source.camera_names))
    if missing_cameras:
        raise ValueError(
            f"Dataset does not contain cameras {missing_cameras}; "
            f"available cameras: {source.camera_names}"
        )
    episode_indices = parse_episode_selection(args.episodes, source.episode_indices)
    task_entities = {
        episode_index: dataset_config.entities_from_task(source.task(episode_index))
        for episode_index in episode_indices
    }

    reference_config = args.reference_config.resolve()
    reference_image, reference_objects = load_reference_config(reference_config)
    resolver = ReferenceResolver(reference_objects, dataset_config)
    unresolved = sorted(
        {
            name
            for entities in task_entities.values()
            for name in resolver.unresolved_names(entities)
        }
    )

    output_root = (
        args.output_root
        if args.output_root is not None
        else WORKSPACE / "runs" / "sam" / source.root.name / "yoloe"
    ).resolve()
    setup_logging(output_root, "yoloe.log")
    if unresolved:
        logger.warning(
            "Reference config has no visual prompt for: %s. "
            "These entities will appear as NOT FOUND in review outputs.",
            ", ".join(unresolved),
        )

    effective_frame_stride = 1 if args.all_frames else args.frame_stride
    model_path = str(Path(args.model).expanduser().resolve()) if Path(args.model).expanduser().is_file() else args.model
    metadata = {
        "format_version": 2,
        "role": "YOLOE visual-prompt box and positive-point proposals for SAM2",
        "dataset_config": dataset_config.name,
        "dataset_root": str(source.root),
        "model": model_path,
        "reference_config": str(reference_config),
        "reference_image": str(reference_image),
        "reference_classes": [
            {
                "class_id": item.class_id,
                "name": item.name,
                "display_name": item.display_name,
            }
            for item in reference_objects
        ],
        "unresolved_entities": unresolved,
        "episodes": episode_indices,
        "cameras": cameras,
        "frame_selection": "all" if args.all_frames else "sampled",
        "frame_stride": effective_frame_stride,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "max_sampled_frames": args.max_sampled_frames,
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "min_quality": args.min_quality,
        "mask_threshold": args.mask_threshold,
        "point_selection": (
            "pixel farthest from the YOLOE mask boundary; box center is used "
            "when no valid mask is available"
        ),
        "object_id_scope": "episode; IDs are identical across cameras for the same episode",
        "selection": "highest complete-view quality score across processed frames",
        "quality_score": (
            "View quality combines confidence, visible area, border clearance, and sharpness; "
            "source entities receive a later-frame penalty"
        ),
    }
    write_json_atomic(output_root / "meta" / "info.json", metadata)

    if args.all_frames:
        logger.info("Frame selection: all episode frames")
    else:
        logger.info("Frame selection: every %d frame(s)", effective_frame_stride)
    logger.info("Loading YOLOE from %s on device %s", model_path, args.device)
    detector = YOLOEVisualPromptDetector(
        model_path,
        reference_image,
        reference_objects,
        device=args.device,
        imgsz=args.imgsz,
        confidence=args.conf,
        iou=args.iou,
    )

    total_jobs = len(episode_indices) * len(cameras)
    completed_jobs = 0
    for episode_index in episode_indices:
        task = source.task(episode_index)
        for camera in cameras:
            entities = dataset_config.entities_for_camera(task_entities[episode_index], camera)
            paths = result_paths(output_root, episode_index, camera)
            if point_result_is_current(paths, entities) and not args.overwrite:
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
                "[%d/%d] YOLOE episode %d camera %s: %s",
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
            ) = annotate_episode_camera(
                source,
                episode_index,
                camera,
                entities,
                detector,
                resolver,
                frame_stride=effective_frame_stride,
                max_sampled_frames=args.max_sampled_frames,
                batch_size=args.batch_size,
                top_k=args.top_k,
                min_quality=args.min_quality,
                mask_threshold=args.mask_threshold,
            )

            records = []
            candidate_records = []
            for entity, selected_detection in zip(entities, detections, strict=True):
                candidates = ranked_candidates[entity.object_id]
                selected_rank = next(
                    (
                        rank
                        for rank, (candidate, _) in enumerate(candidates, start=1)
                        if candidate == selected_detection
                    ),
                    1,
                )
                records.append(
                    point_detection_row(
                        episode_index,
                        camera,
                        task,
                        entity,
                        selected_detection,
                        model_path,
                        effective_frame_stride,
                        sampled_count,
                        selected_rank,
                        selected=selected_detection.visible,
                    )
                )
                ranked_detections = [item[0] for item in candidates] or [selected_detection]
                for candidate_rank, detection in enumerate(ranked_detections, start=1):
                    candidate_records.append(
                        point_detection_row(
                            episode_index,
                            camera,
                            task,
                            entity,
                            detection,
                            model_path,
                            effective_frame_stride,
                            sampled_count,
                            candidate_rank,
                            selected=(
                                selected_detection.visible
                                and detection == selected_detection
                            ),
                        )
                    )

            parquet_path, candidates_path, json_path, preview_path, candidate_preview_path = paths
            write_parquet_atomic(parquet_path, pd.DataFrame(records))
            write_parquet_atomic(candidates_path, pd.DataFrame(candidate_records))
            write_json_atomic(
                json_path,
                {
                    "selected": records,
                    "candidates": candidate_records,
                },
            )
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
                "[%d/%d] Retained up to %d candidates/object and selected %d/%d entities "
                "after %d processed frames",
                completed_jobs,
                total_jobs,
                args.top_k,
                found_count,
                len(detections),
                sampled_count,
            )

    rebuild_summary(output_root, detector_name="YOLOE")
    logger.info("Detection table: %s", output_root / "meta" / "detections.parquet")
    logger.info("Candidate table: %s", output_root / "meta" / "candidates.parquet")
    logger.info("Inspection index: %s", output_root / "inspection" / "index.html")


if __name__ == "__main__":
    main()
