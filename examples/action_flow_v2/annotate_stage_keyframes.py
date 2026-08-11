#!/usr/bin/env python3
"""Manually annotate synchronized stage keyframes with SAM2 point prompts.

The reviewed planning sidecar defines each ``grasp`` and ``put`` interval and
its active target. This UI initializes every stage at its first frame, lets the
reviewer select another frame inside that stage, and always displays the top
and wrist cameras at the same robot timestep.

Brief user manual:

1. Select a stage. The shared frame starts at ``start_frame``.
2. Move the frame slider and press ``Load selected frame`` until the active
   target is represented well in both synchronized camera views.
3. Click positive points inside the target and optional negative points on
   nearby distractors. Select the best SAM2 candidate independently per camera.
4. Use ``Not visible`` only when the target is absent from that camera at the
   selected timestep.
5. Press ``Save stage and next`` after both cameras have a mask or a
   ``Not visible`` decision. Every saved stage contains one shared frame index.

The script excludes ``complete`` stages. It saves source frames, binary masks,
previews, provenance JSON, and a consolidated Parquet table under
``<output-root>``. Existing saved stages are loaded and can be revised.

Example:

    HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 \
      python examples/action_flow_v2/annotate_stage_keyframes.py \
        --dataset-root runs/dataset/rm65_single_table_preparation_and_bussing \
        --planning-root runs/planning/rm65_single_table_preparation_and_bussing \
        --output-root runs/sam/rm65_single_table_preparation_and_bussing_stage_keyframes \
        --episodes all \
        --device cuda \
        --port 7869
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from annotate_insid3_boxes import LeRobotVideoSource, parse_episode_selection
from sam2_image import DEFAULT_SAM2_MODEL, Sam2PointSegmenter


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = WORKSPACE / "runs/dataset/rm65_single_table_preparation_and_bussing"
DEFAULT_PLANNING_ROOT = WORKSPACE / "runs/planning/rm65_single_table_preparation_and_bussing"
DEFAULT_OUTPUT_ROOT = WORKSPACE / "runs/sam/rm65_single_table_preparation_and_bussing_stage_keyframes"
MASK_COLOR = np.array([20, 210, 120], dtype=np.float32)
logger = logging.getLogger(__name__)


def mask_to_xyxy(mask: np.ndarray | None) -> list[int] | None:
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def stage_target(stage: str, source_entity: str, destination_entity: str) -> str:
    if stage == "grasp":
        return source_entity.strip()
    if stage == "put":
        return destination_entity.strip()
    raise ValueError(f"Stage {stage!r} does not have a grounding target")


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result or "target"


@dataclass(frozen=True)
class StageItem:
    episode_index: int
    segment_index: int
    cycle_index: int
    start_frame: int
    end_frame: int
    stage: str
    source_entity: str
    destination_entity: str
    next_step: str
    segment_uid: str
    target_entity: str

    @property
    def key(self) -> str:
        return f"episode-{self.episode_index:06d}-segment-{self.segment_index:03d}"


@dataclass
class CameraDraft:
    points: list[tuple[int, int, int]] = field(default_factory=list)
    candidate_masks: list[np.ndarray] = field(default_factory=list)
    candidate_scores: list[float] = field(default_factory=list)
    selected_candidate: int = 0
    not_visible: bool = False

    def selected_mask(self) -> np.ndarray | None:
        if not self.candidate_masks:
            return None
        return self.candidate_masks[self.selected_candidate]

    def clear(self) -> None:
        self.points.clear()
        self.candidate_masks.clear()
        self.candidate_scores.clear()
        self.selected_candidate = 0
        self.not_visible = False


class StageKeyframeAnnotator:
    """Manage synchronized frame selection and per-camera SAM2 labels."""

    def __init__(
        self,
        dataset_root: Path,
        planning_root: Path,
        output_root: Path,
        episodes: str,
        cameras: list[str],
        model_name: str,
        device: str,
    ):
        if len(cameras) != 2 or len(set(cameras)) != 2:
            raise ValueError("Exactly two distinct --cameras are required")
        self.dataset_root = dataset_root.resolve()
        self.planning_root = planning_root.resolve()
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.cameras = tuple(cameras)
        self.source = LeRobotVideoSource(self.dataset_root)
        missing_cameras = sorted(set(cameras).difference(self.source.camera_keys))
        if missing_cameras:
            raise ValueError(f"Dataset does not contain cameras: {missing_cameras}")

        planning_path = self.planning_root / "review/segments.parquet"
        if not planning_path.is_file():
            raise FileNotFoundError(f"Reviewed planning segments are missing: {planning_path}")
        planning = pd.read_parquet(planning_path).sort_values(
            ["episode_index", "segment_index"]
        )
        available = sorted(planning["episode_index"].astype(int).unique().tolist())
        selected_episodes = set(parse_episode_selection(episodes, available))
        planning = planning.loc[
            planning["episode_index"].isin(selected_episodes)
            & planning["stage"].isin(["grasp", "put"])
        ]
        self.items = [self._make_item(row) for row in planning.to_dict("records")]
        if not self.items:
            raise ValueError("No grasp or put stages matched the requested episodes")

        self.annotations_path = self.output_root / "review/annotations.json"
        self.table_path = self.output_root / "review/boxes.parquet"
        self.annotations = self._load_annotations()
        self.segmenter = Sam2PointSegmenter(model_name, device)
        self.lock = threading.RLock()
        self.active_index: int | None = None
        self.frame_index = 0
        self.drafts = {camera: CameraDraft() for camera in self.cameras}
        self._save_outputs()

    @staticmethod
    def _make_item(row: dict[str, Any]) -> StageItem:
        stage = str(row["stage"])
        source = str(row["source_entity"])
        destination = str(row["destination_entity"])
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        if start >= end:
            raise ValueError(f"Invalid stage interval [{start}, {end})")
        return StageItem(
            episode_index=int(row["episode_index"]),
            segment_index=int(row["segment_index"]),
            cycle_index=int(row["cycle_index"]),
            start_frame=start,
            end_frame=end,
            stage=stage,
            source_entity=source,
            destination_entity=destination,
            next_step=str(row["next_step"]),
            segment_uid=str(row["segment_uid"]),
            target_entity=stage_target(stage, source, destination),
        )

    def _load_annotations(self) -> dict[str, dict[str, Any]]:
        if not self.annotations_path.is_file():
            return {}
        payload = json.loads(self.annotations_path.read_text())
        if int(payload.get("format_version", 0)) != 1:
            raise ValueError(f"Unsupported annotation format in {self.annotations_path}")
        saved_dataset = Path(payload["dataset_root"]).resolve()
        saved_planning = Path(payload["planning_root"]).resolve()
        if saved_dataset != self.dataset_root or saved_planning != self.planning_root:
            raise ValueError("Saved annotations belong to a different dataset or planning sidecar")
        saved_cameras = tuple(payload["cameras"])
        if saved_cameras != self.cameras:
            raise ValueError(f"Saved cameras are {saved_cameras}, requested {self.cameras}")
        return {str(entry["stage_key"]): entry for entry in payload.get("annotations", [])}

    def _item(self, item_index: float | int) -> StageItem:
        index = int(item_index)
        if not 0 <= index < len(self.items):
            raise ValueError(f"Stage {index} is outside 0..{len(self.items) - 1}")
        return self.items[index]

    def _reset_drafts(self) -> None:
        for draft in self.drafts.values():
            draft.clear()
        self.segmenter.clear_image()

    def _activate(self, item_index: float | int) -> StageItem:
        item = self._item(item_index)
        index = int(item_index)
        if self.active_index == index:
            return item
        self.active_index = index
        self._reset_drafts()
        saved = self.annotations.get(item.key)
        self.frame_index = int(saved["frame_index"]) if saved else item.start_frame
        self.frame_index = int(np.clip(self.frame_index, item.start_frame, item.end_frame - 1))
        if saved:
            for camera in self.cameras:
                camera_saved = saved["cameras"][camera]
                draft = self.drafts[camera]
                draft.not_visible = not bool(camera_saved["visible"])
                draft.points = [
                    (
                        int(point["x"]),
                        int(point["y"]),
                        1 if point["label"] == "positive" else 0,
                    )
                    for point in camera_saved.get("points", [])
                ]
                mask_path = camera_saved.get("mask_path")
                if bool(camera_saved["visible"]) and mask_path:
                    mask = np.asarray(
                        Image.open(self.output_root / mask_path).convert("L"), dtype=np.uint8
                    ) > 0
                    draft.candidate_masks = [mask]
                    draft.candidate_scores = [float(camera_saved.get("sam_predicted_iou", 1.0))]
        return item

    def _frame(self, item: StageItem, camera: str) -> np.ndarray:
        return np.asarray(
            self.source.frame(item.episode_index, camera, self.frame_index).convert("RGB"),
            dtype=np.uint8,
        )

    def _task_prompt(self, item: StageItem) -> str:
        tasks = self.source.row(item.episode_index)["tasks"]
        if isinstance(tasks, str):
            return tasks
        return "; ".join(str(task) for task in tasks)

    @staticmethod
    def _instruction(item: StageItem) -> str:
        role = "source object to grasp" if item.stage == "grasp" else "placement destination"
        return (
            f"## Segment only: `{item.target_entity}`\n"
            f"**Stage:** `{item.stage}` | **Role:** {role} | "
            f"**Episode:** {item.episode_index} | **Segment:** {item.segment_index} | "
            f"**Interval:** `[{item.start_frame}, {item.end_frame})`"
        )

    def _run_sam2(self, item: StageItem, camera: str) -> None:
        draft = self.drafts[camera]
        if not any(label == 1 for _, _, label in draft.points):
            draft.candidate_masks = []
            draft.candidate_scores = []
            draft.selected_candidate = 0
            return
        draft.candidate_masks, draft.candidate_scores = self.segmenter.segment(
            self._frame(item, camera),
            draft.points,
            image_key=(item.key, self.frame_index, camera),
        )
        draft.selected_candidate = int(np.argmax(draft.candidate_scores))

    def render(self, item_index: float | int, camera: str) -> np.ndarray:
        item = self._activate(item_index)
        draft = self.drafts[camera]
        frame = self._frame(item, camera).copy()
        mask = draft.selected_mask()
        if mask is not None and np.any(mask):
            pixels = frame[mask].astype(np.float32)
            frame[mask] = np.clip(0.55 * pixels + 0.45 * MASK_COLOR, 0, 255).astype(np.uint8)
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        if mask is not None:
            box = mask_to_xyxy(mask)
            if box is not None:
                draw.rectangle(
                    (box[0], box[1], box[2] - 1, box[3] - 1),
                    outline="#10d890",
                    width=3,
                )
        if draft.not_visible:
            draw.text((12, 12), "NOT VISIBLE", fill="#ffd33d", stroke_width=2, stroke_fill="black")
        for x, y, label in draft.points:
            color = "#00ff80" if label == 1 else "#ff304f"
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=1)
        return np.asarray(image)

    def camera_status(self, item: StageItem, camera: str) -> str:
        draft = self.drafts[camera]
        if draft.not_visible:
            label = "ready: not visible"
        elif draft.selected_mask() is None:
            label = "pending: add a positive point or mark not visible"
        else:
            mask = draft.selected_mask()
            assert mask is not None
            score = draft.candidate_scores[draft.selected_candidate]
            label = (
                f"ready: candidate {draft.selected_candidate + 1}, "
                f"score={score:.3f}, area={float(mask.mean()):.3%}"
            )
        return (
            f"{camera} | frame {self.frame_index} | {label} | "
            f"positive={sum(label == 1 for _, _, label in draft.points)}, "
            f"negative={sum(label == 0 for _, _, label in draft.points)}"
        )

    def progress(self) -> str:
        reviewed = len(self.annotations)
        return f"{reviewed}/{len(self.items)} stages saved | {len(self.items) - reviewed} pending"

    def load_item(self, item_index: float | int) -> dict[str, Any]:
        with self.lock:
            item = self._activate(item_index)
            return {
                "instruction": self._instruction(item),
                "task": self._task_prompt(item),
                "next_step": item.next_step,
                "frame_index": self.frame_index,
                "frame_minimum": item.start_frame,
                "frame_maximum": item.end_frame - 1,
                "images": {camera: self.render(item_index, camera) for camera in self.cameras},
                "candidates": {
                    camera: str(self.drafts[camera].selected_candidate + 1)
                    for camera in self.cameras
                },
                "statuses": {
                    camera: self.camera_status(item, camera) for camera in self.cameras
                },
                "progress": self.progress(),
            }

    def select_frame(self, item_index: float | int, frame_index: float | int) -> dict[str, Any]:
        with self.lock:
            item = self._activate(item_index)
            selected = int(np.clip(int(frame_index), item.start_frame, item.end_frame - 1))
            if selected != self.frame_index:
                self.frame_index = selected
                self._reset_drafts()
            return self.load_item(item_index)

    def shift_frame(self, item_index: float | int, frame_index: float | int, delta: int) -> dict[str, Any]:
        return self.select_frame(item_index, int(frame_index) + delta)

    def add_point(
        self,
        item_index: float | int,
        camera: str,
        point_mode: str,
        event: Any,
    ) -> tuple[np.ndarray, str, str]:
        with self.lock:
            item = self._activate(item_index)
            frame = self._frame(item, camera)
            x, y = (int(round(value)) for value in event.index[:2])
            height, width = frame.shape[:2]
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Click {(x, y)} is outside {width}x{height}")
            draft = self.drafts[camera]
            draft.not_visible = False
            draft.points.append((x, y, 1 if point_mode == "Positive" else 0))
            self._run_sam2(item, camera)
            return (
                self.render(item_index, camera),
                self.camera_status(item, camera),
                str(draft.selected_candidate + 1),
            )

    def choose_candidate(
        self, item_index: float | int, camera: str, candidate: str
    ) -> tuple[np.ndarray, str]:
        with self.lock:
            item = self._activate(item_index)
            draft = self.drafts[camera]
            if draft.candidate_masks:
                draft.selected_candidate = int(
                    np.clip(int(candidate) - 1, 0, len(draft.candidate_masks) - 1)
                )
            return self.render(item_index, camera), self.camera_status(item, camera)

    def undo_point(self, item_index: float | int, camera: str) -> tuple[np.ndarray, str, str]:
        with self.lock:
            item = self._activate(item_index)
            draft = self.drafts[camera]
            if draft.points:
                draft.points.pop()
            self._run_sam2(item, camera)
            return (
                self.render(item_index, camera),
                self.camera_status(item, camera),
                str(draft.selected_candidate + 1),
            )

    def clear_camera(self, item_index: float | int, camera: str) -> tuple[np.ndarray, str, str]:
        with self.lock:
            item = self._activate(item_index)
            self.drafts[camera].clear()
            return self.render(item_index, camera), self.camera_status(item, camera), "1"

    def mark_not_visible(
        self, item_index: float | int, camera: str
    ) -> tuple[np.ndarray, str, str]:
        with self.lock:
            item = self._activate(item_index)
            draft = self.drafts[camera]
            draft.clear()
            draft.not_visible = True
            return self.render(item_index, camera), self.camera_status(item, camera), "1"

    def _stage_paths(self, item: StageItem, camera: str) -> tuple[Path, Path, Path]:
        filename = f"{item.key}-{slug(item.target_entity)}.png"
        return (
            self.output_root / "images" / camera / filename,
            self.output_root / "masks" / camera / filename,
            self.output_root / "previews" / camera / filename,
        )

    def _save_preview(self, item_index: int, camera: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.render(item_index, camera)).save(path)

    def _camera_record(self, item: StageItem, item_index: int, camera: str) -> dict[str, Any]:
        draft = self.drafts[camera]
        image_path, mask_path, preview_path = self._stage_paths(item, camera)
        for path in (image_path, mask_path, preview_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        frame = self._frame(item, camera)
        Image.fromarray(frame).save(image_path)
        self._save_preview(item_index, camera, preview_path)
        if draft.not_visible:
            mask_path.unlink(missing_ok=True)
            return {
                "visible": False,
                "bbox_xyxy": None,
                "mask_path": None,
                "image_path": str(image_path.relative_to(self.output_root)),
                "preview_path": str(preview_path.relative_to(self.output_root)),
                "points": [],
                "sam_candidate": None,
                "sam_predicted_iou": None,
                "foreground_fraction": 0.0,
            }
        mask = draft.selected_mask()
        box = mask_to_xyxy(mask) if mask is not None else None
        if mask is None or box is None:
            raise RuntimeError(f"{camera}: add a valid SAM2 mask or mark the target not visible")
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        return {
            "visible": True,
            "bbox_xyxy": box,
            "mask_path": str(mask_path.relative_to(self.output_root)),
            "image_path": str(image_path.relative_to(self.output_root)),
            "preview_path": str(preview_path.relative_to(self.output_root)),
            "points": [
                {"x": x, "y": y, "label": "positive" if label == 1 else "negative"}
                for x, y, label in draft.points
            ],
            "sam_candidate": draft.selected_candidate + 1,
            "sam_predicted_iou": float(draft.candidate_scores[draft.selected_candidate]),
            "foreground_fraction": float(mask.mean()),
        }

    def save_stage(self, item_index: float | int) -> int:
        with self.lock:
            index = int(item_index)
            item = self._activate(index)
            for camera in self.cameras:
                draft = self.drafts[camera]
                if not draft.not_visible and mask_to_xyxy(draft.selected_mask()) is None:
                    raise RuntimeError(
                        f"{camera}: add a valid SAM2 mask or mark the target not visible"
                    )
            camera_records = {
                camera: self._camera_record(item, index, camera) for camera in self.cameras
            }
            self.annotations[item.key] = {
                "stage_key": item.key,
                "episode_index": item.episode_index,
                "segment_index": item.segment_index,
                "cycle_index": item.cycle_index,
                "start_frame": item.start_frame,
                "end_frame": item.end_frame,
                "frame_index": self.frame_index,
                "stage": item.stage,
                "source_entity": item.source_entity,
                "destination_entity": item.destination_entity,
                "target_entity": item.target_entity,
                "next_step": item.next_step,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "cameras": camera_records,
            }
            self._save_outputs()
            return self.next_pending(index)

    def clear_saved_stage(self, item_index: float | int) -> int:
        with self.lock:
            index = int(item_index)
            item = self._activate(index)
            self.annotations.pop(item.key, None)
            for camera in self.cameras:
                for path in self._stage_paths(item, camera):
                    path.unlink(missing_ok=True)
            self._save_outputs()
            self.active_index = None
            return index

    def next_pending(self, item_index: float | int) -> int:
        start = int(item_index)
        for offset in range(1, len(self.items) + 1):
            candidate = (start + offset) % len(self.items)
            if self.items[candidate].key not in self.annotations:
                return candidate
        return start

    def _save_outputs(self) -> None:
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "rm65_synchronized_stage_sam2_keyframes",
            "format_version": 1,
            "dataset_root": str(self.dataset_root),
            "planning_root": str(self.planning_root),
            "cameras": list(self.cameras),
            "frame_semantics": "one shared robot frame index for all cameras in a stage",
            "bbox_format": "xyxy_inclusive_exclusive",
            "mask_format": "binary_png_0_255",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "annotations": sorted(
                self.annotations.values(),
                key=lambda entry: (int(entry["episode_index"]), int(entry["segment_index"])),
            ),
        }
        temporary_json = self.annotations_path.with_suffix(".json.tmp")
        temporary_json.write_text(json.dumps(payload, indent=2) + "\n")
        temporary_json.replace(self.annotations_path)

        rows: list[dict[str, Any]] = []
        for item in self.items:
            saved = self.annotations.get(item.key)
            for camera in self.cameras:
                camera_saved = saved["cameras"][camera] if saved else None
                rows.append(
                    {
                        "episode_index": item.episode_index,
                        "segment_index": item.segment_index,
                        "cycle_index": item.cycle_index,
                        "start_frame": item.start_frame,
                        "end_frame": item.end_frame,
                        "frame_index": int(saved["frame_index"]) if saved else item.start_frame,
                        "stage": item.stage,
                        "source_entity": item.source_entity,
                        "destination_entity": item.destination_entity,
                        "target_entity": item.target_entity,
                        "next_step": item.next_step,
                        "camera": camera,
                        "reviewed": saved is not None,
                        "visible": camera_saved["visible"] if camera_saved else None,
                        "bbox_xyxy": camera_saved["bbox_xyxy"] if camera_saved else None,
                        "mask_path": camera_saved["mask_path"] if camera_saved else None,
                        "image_path": camera_saved["image_path"] if camera_saved else None,
                        "preview_path": camera_saved["preview_path"] if camera_saved else None,
                        "sam_predicted_iou": (
                            camera_saved["sam_predicted_iou"] if camera_saved else None
                        ),
                        "foreground_fraction": (
                            camera_saved["foreground_fraction"] if camera_saved else None
                        ),
                    }
                )
        table = pd.DataFrame(rows)
        temporary_parquet = self.table_path.with_suffix(".parquet.tmp")
        table.to_parquet(temporary_parquet, index=False)
        temporary_parquet.replace(self.table_path)


def sanitize_proxy_environment() -> None:
    for name in ("ALL_PROXY", "all_proxy"):
        if os.environ.get(name, "").lower().startswith("socks://"):
            os.environ.pop(name, None)


def build_app(annotator: StageKeyframeAnnotator, initial_item: int) -> Any:
    sanitize_proxy_environment()
    import gradio as gr

    camera_a, camera_b = annotator.cameras

    def payload_outputs(payload: dict[str, Any]) -> tuple[Any, ...]:
        return (
            payload["instruction"],
            payload["task"],
            payload["next_step"],
            gr.update(
                value=payload["frame_index"],
                minimum=payload["frame_minimum"],
                maximum=payload["frame_maximum"],
            ),
            payload["images"][camera_a],
            payload["candidates"][camera_a],
            payload["statuses"][camera_a],
            payload["images"][camera_b],
            payload["candidates"][camera_b],
            payload["statuses"][camera_b],
            payload["progress"],
        )

    def load_stage(item_value: float) -> tuple[Any, ...]:
        return payload_outputs(annotator.load_item(item_value))

    def load_frame(item_value: float, frame_value: float) -> tuple[Any, ...]:
        return payload_outputs(annotator.select_frame(item_value, frame_value))

    def shift_frame(item_value: float, frame_value: float, delta: int) -> tuple[Any, ...]:
        return payload_outputs(annotator.shift_frame(item_value, frame_value, delta))

    with gr.Blocks(title="Synchronized Stage Keyframe Annotation") as app:
        with gr.Row():
            previous_button = gr.Button("Previous")
            item_index = gr.Slider(
                0,
                len(annotator.items) - 1,
                value=initial_item,
                step=1,
                label="Stage",
            )
            next_button = gr.Button("Next")
            next_pending_button = gr.Button("Next pending")

        instruction = gr.Markdown()
        task = gr.Textbox(label="Task prompt", interactive=False)
        next_step = gr.Textbox(label="Current stage instruction", interactive=False)

        with gr.Row():
            frame_slider = gr.Slider(0, 1, value=0, step=1, label="Shared frame index")
            minus_ten = gr.Button("-10")
            minus_one = gr.Button("-1")
            plus_one = gr.Button("+1")
            plus_ten = gr.Button("+10")
            load_frame_button = gr.Button("Load selected frame", variant="primary")

        point_mode = gr.Radio(["Positive", "Negative"], value="Positive", label="Click type")

        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### `{camera_a}` camera")
                image_a = gr.Image(type="numpy", label=f"{camera_a} synchronized frame", height=520)
                candidate_a = gr.Radio(["1", "2", "3"], value="1", label="SAM2 candidate")
                with gr.Row():
                    undo_a = gr.Button("Undo point")
                    clear_a = gr.Button("Clear")
                    not_visible_a = gr.Button("Not visible")
                status_a = gr.Textbox(label=f"{camera_a} status", interactive=False, lines=2)

            with gr.Column():
                gr.Markdown(f"### `{camera_b}` camera")
                image_b = gr.Image(type="numpy", label=f"{camera_b} synchronized frame", height=520)
                candidate_b = gr.Radio(["1", "2", "3"], value="1", label="SAM2 candidate")
                with gr.Row():
                    undo_b = gr.Button("Undo point")
                    clear_b = gr.Button("Clear")
                    not_visible_b = gr.Button("Not visible")
                status_b = gr.Textbox(label=f"{camera_b} status", interactive=False, lines=2)

        with gr.Row():
            save_button = gr.Button("Save stage and next", variant="primary")
            clear_saved_button = gr.Button("Clear saved stage")
        progress = gr.Textbox(label="Progress", interactive=False)

        stage_outputs = [
            instruction,
            task,
            next_step,
            frame_slider,
            image_a,
            candidate_a,
            status_a,
            image_b,
            candidate_b,
            status_b,
            progress,
        ]

        def add_point_a(item_value: float, mode: str, event: Any):
            return annotator.add_point(item_value, camera_a, mode, event)

        def add_point_b(item_value: float, mode: str, event: Any):
            return annotator.add_point(item_value, camera_b, mode, event)

        add_point_a.__annotations__["event"] = gr.SelectData
        add_point_b.__annotations__["event"] = gr.SelectData

        def save_and_load(item_value: float) -> tuple[Any, ...]:
            next_index = annotator.save_stage(item_value)
            return gr.update(value=next_index), *load_stage(next_index)

        def clear_and_load(item_value: float) -> tuple[Any, ...]:
            index = annotator.clear_saved_stage(item_value)
            return gr.update(value=index), *load_stage(index)

        app.load(load_stage, inputs=item_index, outputs=stage_outputs)
        item_index.change(
            load_stage,
            inputs=item_index,
            outputs=stage_outputs,
            show_progress="hidden",
        )
        load_frame_button.click(
            load_frame,
            inputs=[item_index, frame_slider],
            outputs=stage_outputs,
            concurrency_limit=1,
        )
        def make_shift_handler(delta: int):
            def handler(item: float, frame: float):
                return shift_frame(item, frame, delta)

            return handler

        for button, delta in ((minus_ten, -10), (minus_one, -1), (plus_one, 1), (plus_ten, 10)):
            button.click(
                make_shift_handler(delta),
                inputs=[item_index, frame_slider],
                outputs=stage_outputs,
                concurrency_limit=1,
            )

        image_a.select(
            add_point_a,
            inputs=[item_index, point_mode],
            outputs=[image_a, status_a, candidate_a],
            concurrency_limit=1,
        )
        image_b.select(
            add_point_b,
            inputs=[item_index, point_mode],
            outputs=[image_b, status_b, candidate_b],
            concurrency_limit=1,
        )
        candidate_a.change(
            lambda item, candidate: annotator.choose_candidate(item, camera_a, candidate),
            inputs=[item_index, candidate_a],
            outputs=[image_a, status_a],
            show_progress="hidden",
        )
        candidate_b.change(
            lambda item, candidate: annotator.choose_candidate(item, camera_b, candidate),
            inputs=[item_index, candidate_b],
            outputs=[image_b, status_b],
            show_progress="hidden",
        )
        undo_a.click(
            lambda item: annotator.undo_point(item, camera_a),
            inputs=item_index,
            outputs=[image_a, status_a, candidate_a],
            concurrency_limit=1,
        )
        undo_b.click(
            lambda item: annotator.undo_point(item, camera_b),
            inputs=item_index,
            outputs=[image_b, status_b, candidate_b],
            concurrency_limit=1,
        )
        clear_a.click(
            lambda item: annotator.clear_camera(item, camera_a),
            inputs=item_index,
            outputs=[image_a, status_a, candidate_a],
        )
        clear_b.click(
            lambda item: annotator.clear_camera(item, camera_b),
            inputs=item_index,
            outputs=[image_b, status_b, candidate_b],
        )
        not_visible_a.click(
            lambda item: annotator.mark_not_visible(item, camera_a),
            inputs=item_index,
            outputs=[image_a, status_a, candidate_a],
        )
        not_visible_b.click(
            lambda item: annotator.mark_not_visible(item, camera_b),
            inputs=item_index,
            outputs=[image_b, status_b, candidate_b],
        )

        action_outputs = [item_index, *stage_outputs]
        save_button.click(
            save_and_load,
            inputs=item_index,
            outputs=action_outputs,
            concurrency_limit=1,
        )
        clear_saved_button.click(
            clear_and_load,
            inputs=item_index,
            outputs=action_outputs,
        )
        previous_button.click(
            lambda value: max(0, int(value) - 1),
            inputs=item_index,
            outputs=item_index,
            show_progress="hidden",
        )
        next_button.click(
            lambda value: min(len(annotator.items) - 1, int(value) + 1),
            inputs=item_index,
            outputs=item_index,
            show_progress="hidden",
        )
        next_pending_button.click(
            annotator.next_pending,
            inputs=item_index,
            outputs=item_index,
            show_progress="hidden",
        )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--planning-root", type=Path, default=DEFAULT_PLANNING_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episodes", default="all")
    parser.add_argument("--cameras", nargs=2, default=["top", "wrist"])
    parser.add_argument("--model-name", default=DEFAULT_SAM2_MODEL)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, or cpu")
    parser.add_argument("--initial-item", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7869)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    annotator = StageKeyframeAnnotator(
        dataset_root=args.dataset_root,
        planning_root=args.planning_root,
        output_root=args.output_root,
        episodes=args.episodes,
        cameras=args.cameras,
        model_name=args.model_name,
        device=args.device,
    )
    if not 0 <= args.initial_item < len(annotator.items):
        raise ValueError(f"--initial-item must be in 0..{len(annotator.items) - 1}")
    logger.info("Annotations: %s", annotator.annotations_path)
    logger.info("Consolidated labels: %s", annotator.table_path)
    app = build_app(annotator, args.initial_item)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
