#!/usr/bin/env python3
"""Review INSID3 keyframe boxes and correct them with SAM2 or manual boxes.

Every decision is saved immediately under ``<insid3-root>/review``. Pending
INSID3 proposals remain visibly available but are marked ``reviewed=False`` in
the consolidated Parquet and must not be used as verified training labels.

Controls:

- ``Accept INSID3``: verify the red proposal box.
- Positive/negative image clicks: refine one target with SAM2.
- ``Accept SAM2``: save the selected SAM2 candidate's bounding box and mask.
- ``Box corner`` clicks: select two opposite corners, then save the manual box.
- ``Not visible``: save a valid negative label with no box.

Example:

    HF_HUB_OFFLINE=1 python examples/action_flow_v2/review_insid3_boxes.py \
        --insid3-root \
        runs/insid3/rm65_single_table_preparation_and_bussing_sam2_stratified_pilot \
        --device cuda \
        --port 7869
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from annotate_insid3_boxes import LeRobotVideoSource
from sam2_image import DEFAULT_SAM2_MODEL, Sam2PointSegmenter


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_INSID3_ROOT = (
    WORKSPACE
    / "runs/insid3/rm65_single_table_preparation_and_bussing_sam2_stratified_pilot"
)
MASK_COLOR = np.array([20, 210, 120], dtype=np.float32)
logger = logging.getLogger(__name__)


def normalize_box(value: Any) -> list[int] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(array) != 4 or not np.isfinite(array).all():
        return None
    x0, y0, x1, y1 = (int(round(number)) for number in array)
    return [x0, y0, x1, y1] if x0 < x1 and y0 < y1 else None


def mask_to_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


@dataclass(frozen=True)
class ReviewItem:
    row_index: int
    episode_index: int
    segment_index: int
    cycle_index: int
    stage: str
    camera: str
    frame_index: int
    target_entity: str
    next_step: str
    proposal_box: list[int] | None
    proposal_status: str
    proposal_review_required: bool

    @property
    def key(self) -> str:
        return (
            f"episode-{self.episode_index:06d}-segment-{self.segment_index:03d}-"
            f"camera-{self.camera}"
        )

    @property
    def label(self) -> str:
        return (
            f"{self.target_entity} | episode {self.episode_index} | "
            f"segment {self.segment_index} ({self.stage}) | {self.camera} | "
            f"frame {self.frame_index}"
        )


class INSID3BoxReviewer:
    """Render INSID3 proposals and persist verified box decisions."""

    def __init__(
        self,
        insid3_root: Path,
        model_name: str,
        device: str,
        review_path: Path | None = None,
    ):
        self.root = insid3_root.resolve()
        info_path = self.root / "meta/info.json"
        boxes_path = self.root / "data/boxes.parquet"
        if not info_path.is_file() or not boxes_path.is_file():
            raise FileNotFoundError(f"INSID3 metadata or boxes are missing under {self.root}")
        self.info = json.loads(info_path.read_text())
        self.dataset_root = Path(self.info["dataset_root"]).resolve()
        self.source = LeRobotVideoSource(self.dataset_root)
        self.table = pd.read_parquet(boxes_path).sort_values(
            ["episode_index", "segment_index", "camera"]
        ).reset_index(drop=True)
        required = {
            "episode_index",
            "segment_index",
            "cycle_index",
            "stage",
            "camera",
            "frame_index",
            "target_entity",
            "next_step",
            "bbox_xyxy",
            "status",
            "review_required",
        }
        missing = required.difference(self.table.columns)
        if missing:
            raise ValueError(f"INSID3 box table is missing columns: {sorted(missing)}")
        self.items = [
            ReviewItem(
                row_index=index,
                episode_index=int(row.episode_index),
                segment_index=int(row.segment_index),
                cycle_index=int(row.cycle_index),
                stage=str(row.stage),
                camera=str(row.camera),
                frame_index=int(row.frame_index),
                target_entity=str(row.target_entity),
                next_step=str(row.next_step),
                proposal_box=normalize_box(row.bbox_xyxy),
                proposal_status=str(row.status),
                proposal_review_required=bool(row.review_required),
            )
            for index, row in enumerate(self.table.itertuples(index=False))
        ]
        if not self.items:
            raise ValueError(f"No box review items found under {self.root}")

        self.review_path = (
            review_path.resolve()
            if review_path is not None
            else self.root / "review/decisions.json"
        )
        self.review_table_path = self.root / "review/boxes.parquet"
        self.sam_mask_root = self.root / "review/sam_masks"
        self.decisions = self._load_decisions()
        self.segmenter = Sam2PointSegmenter(model_name, device)
        self.lock = threading.RLock()
        self.active_index: int | None = None
        self.points: list[tuple[int, int, int]] = []
        self.manual_corners: list[tuple[int, int]] = []
        self.candidate_masks: list[np.ndarray] = []
        self.candidate_scores: list[float] = []
        self.selected_candidate = 0
        self._save_outputs()

    def _load_decisions(self) -> dict[str, dict[str, Any]]:
        if not self.review_path.is_file():
            return {}
        payload = json.loads(self.review_path.read_text())
        if int(payload.get("format_version", 0)) != 1:
            raise ValueError(f"Unsupported review format in {self.review_path}")
        saved_root = Path(payload["insid3_root"]).resolve()
        if saved_root != self.root:
            raise ValueError(f"Review belongs to {saved_root}, not {self.root}")
        return {str(item["item_key"]): item for item in payload.get("decisions", [])}

    def _save_outputs(self) -> None:
        self.review_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "role": "verified INSID3 keyframe boxes",
            "insid3_root": str(self.root),
            "dataset_root": str(self.dataset_root),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "default": "Items without a decision are pending and are not verified labels.",
            "decisions": sorted(
                self.decisions.values(),
                key=lambda item: (
                    int(item["episode_index"]),
                    int(item["segment_index"]),
                    str(item["camera"]),
                ),
            ),
        }
        temporary_json = self.review_path.with_suffix(".json.tmp")
        temporary_json.write_text(json.dumps(payload, indent=2) + "\n")
        temporary_json.replace(self.review_path)

        rows: list[dict[str, Any]] = []
        for item, source_row in zip(self.items, self.table.to_dict("records"), strict=True):
            row = dict(source_row)
            proposal_box = normalize_box(row.get("bbox_xyxy"))
            decision = self.decisions.get(item.key)
            row["insid3_bbox_xyxy"] = proposal_box
            row["insid3_visible"] = bool(row.get("visible", proposal_box is not None))
            if decision is None:
                row["bbox_xyxy"] = proposal_box
                row["reviewed"] = False
                row["review_status"] = "pending"
                row["box_source"] = "insid3_unreviewed"
                row["reviewed_at"] = None
            else:
                row["bbox_xyxy"] = normalize_box(decision.get("bbox_xyxy"))
                row["visible"] = bool(decision["visible"])
                row["reviewed"] = True
                row["review_status"] = str(decision["status"])
                row["box_source"] = str(decision["box_source"])
                row["reviewed_at"] = str(decision["reviewed_at"])
            rows.append(row)
        reviewed = pd.DataFrame(rows)
        temporary_parquet = self.review_table_path.with_suffix(".parquet.tmp")
        reviewed.to_parquet(temporary_parquet, index=False)
        temporary_parquet.replace(self.review_table_path)

    def _item(self, item_index: float | int) -> ReviewItem:
        index = int(item_index)
        if not 0 <= index < len(self.items):
            raise ValueError(f"Review item {index} is outside 0..{len(self.items) - 1}")
        return self.items[index]

    def _activate(self, item_index: float | int) -> ReviewItem:
        item = self._item(item_index)
        if self.active_index != int(item_index):
            self.active_index = int(item_index)
            self.points = []
            self.manual_corners = []
            self.candidate_masks = []
            self.candidate_scores = []
            self.selected_candidate = 0
            self.segmenter.clear_image()
        return item

    def _frame(self, item: ReviewItem) -> np.ndarray:
        return np.asarray(
            self.source.frame(item.episode_index, item.camera, item.frame_index).convert("RGB"),
            dtype=np.uint8,
        )

    def _task_prompt(self, item: ReviewItem) -> str:
        tasks = self.source.row(item.episode_index)["tasks"]
        if isinstance(tasks, str):
            return tasks
        return "; ".join(str(task) for task in tasks)

    @staticmethod
    def _review_instruction(item: ReviewItem) -> str:
        stage_roles = {
            "grasp": "source object to grasp",
            "put": "destination where the held object must be placed",
            "complete": "no target; mark the item not visible",
        }
        role = stage_roles.get(item.stage, "target relevant to the current stage")
        return (
            f"## Segment only: `{item.target_entity}`\n"
            f"**Stage:** `{item.stage}` | **Expected role:** {role} | "
            f"**Camera:** `{item.camera}`"
        )

    def _selected_mask(self) -> np.ndarray | None:
        if not self.candidate_masks:
            return None
        return self.candidate_masks[self.selected_candidate]

    @staticmethod
    def _draw_box(draw: ImageDraw.ImageDraw, box: list[int], color: str, width: int) -> None:
        draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=color, width=width)

    def render(self, item_index: float | int) -> np.ndarray:
        item = self._activate(item_index)
        frame = self._frame(item).copy()
        mask = self._selected_mask()
        if mask is not None and np.any(mask):
            pixels = frame[mask].astype(np.float32)
            frame[mask] = np.clip(0.55 * pixels + 0.45 * MASK_COLOR, 0, 255).astype(np.uint8)
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        if item.proposal_box is not None:
            self._draw_box(draw, item.proposal_box, "#ff4050", 2)
        if mask is not None:
            sam_box = mask_to_xyxy(mask)
            if sam_box is not None:
                self._draw_box(draw, sam_box, "#10d890", 3)
        if len(self.manual_corners) == 2:
            manual_box = self._manual_box(item)
            if manual_box is not None:
                self._draw_box(draw, manual_box, "#00bfff", 3)
        decision = self.decisions.get(item.key)
        if decision is not None:
            reviewed_box = normalize_box(decision.get("bbox_xyxy"))
            if reviewed_box is not None:
                self._draw_box(draw, reviewed_box, "#ffd33d", 4)
            elif not decision["visible"]:
                draw.text((12, 12), "NOT VISIBLE", fill="#ffd33d", stroke_width=2, stroke_fill="black")
        for x, y, label in self.points:
            color = "#00ff80" if label == 1 else "#ff304f"
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=1)
        for index, (x, y) in enumerate(self.manual_corners, start=1):
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#00bfff", outline="white")
            draw.text((x + 7, y - 8), str(index), fill="#00bfff", stroke_width=1, stroke_fill="black")
        return np.asarray(image)

    def _manual_box(self, item: ReviewItem) -> list[int] | None:
        if len(self.manual_corners) != 2:
            return None
        frame = self._frame(item)
        height, width = frame.shape[:2]
        (x0, y0), (x1, y1) = self.manual_corners
        left, right = sorted((int(np.clip(x0, 0, width - 1)), int(np.clip(x1, 0, width - 1))))
        top, bottom = sorted((int(np.clip(y0, 0, height - 1)), int(np.clip(y1, 0, height - 1))))
        return [left, top, right + 1, bottom + 1] if left < right and top < bottom else None

    def _progress(self) -> str:
        reviewed = len(self.decisions)
        return f"{reviewed}/{len(self.items)} reviewed | {len(self.items) - reviewed} pending"

    def _status(self, item: ReviewItem) -> str:
        decision = self.decisions.get(item.key)
        decision_text = "pending" if decision is None else decision["status"]
        proposal = item.proposal_box if item.proposal_box is not None else "none"
        sam_text = "none"
        if self.candidate_scores:
            sam_text = (
                f"candidate {self.selected_candidate + 1}, "
                f"score {self.candidate_scores[self.selected_candidate]:.3f}"
            )
        manual = self._manual_box(item)
        return (
            f"Decision: {decision_text}\n"
            f"INSID3 proposal: {proposal} | automatic review flag: "
            f"{item.proposal_review_required}\n"
            f"SAM2: {sam_text} | points: {len(self.points)}\n"
            f"Manual box: {manual if manual is not None else 'click two opposite corners'}\n"
            f"{self._progress()}"
        )

    def load_item(self, item_index: float | int) -> tuple[str, str, str, np.ndarray, str, str, str]:
        with self.lock:
            item = self._activate(item_index)
            return (
                self._review_instruction(item),
                self._task_prompt(item),
                item.next_step,
                self.render(item_index),
                "1",
                self._status(item),
                self._progress(),
            )

    def _run_sam2(self, item: ReviewItem) -> None:
        if not any(label == 1 for _, _, label in self.points):
            self.candidate_masks = []
            self.candidate_scores = []
            return
        self.candidate_masks, self.candidate_scores = self.segmenter.segment(
            self._frame(item),
            self.points,
            image_key=item.key,
        )
        self.selected_candidate = int(np.argmax(self.candidate_scores))

    def add_click(
        self,
        item_index: float | int,
        click_mode: str,
        event: Any,
    ) -> tuple[np.ndarray, str, str]:
        with self.lock:
            item = self._activate(item_index)
            frame = self._frame(item)
            x, y = (int(round(value)) for value in event.index[:2])
            height, width = frame.shape[:2]
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Click {(x, y)} is outside {width}x{height}")
            if click_mode == "Box corner":
                if len(self.manual_corners) == 2:
                    self.manual_corners = []
                self.manual_corners.append((x, y))
            else:
                label = 1 if click_mode == "Positive" else 0
                self.points.append((x, y, label))
                self._run_sam2(item)
            return self.render(item_index), self._status(item), str(self.selected_candidate + 1)

    def choose_candidate(self, item_index: float | int, candidate: str) -> tuple[np.ndarray, str]:
        with self.lock:
            item = self._activate(item_index)
            if self.candidate_masks:
                self.selected_candidate = int(
                    np.clip(int(candidate) - 1, 0, len(self.candidate_masks) - 1)
                )
            return self.render(item_index), self._status(item)

    def undo_point(self, item_index: float | int) -> tuple[np.ndarray, str, str]:
        with self.lock:
            item = self._activate(item_index)
            if self.points:
                self.points.pop()
            self._run_sam2(item)
            return self.render(item_index), self._status(item), str(self.selected_candidate + 1)

    def clear_correction(self, item_index: float | int) -> tuple[np.ndarray, str, str]:
        with self.lock:
            item = self._activate(item_index)
            self.points = []
            self.manual_corners = []
            self.candidate_masks = []
            self.candidate_scores = []
            self.selected_candidate = 0
            return self.render(item_index), self._status(item), "1"

    def _base_decision(self, item: ReviewItem) -> dict[str, Any]:
        return {
            "item_key": item.key,
            "episode_index": item.episode_index,
            "segment_index": item.segment_index,
            "cycle_index": item.cycle_index,
            "stage": item.stage,
            "camera": item.camera,
            "frame_index": item.frame_index,
            "target_entity": item.target_entity,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _record_decision(
        self,
        item: ReviewItem,
        *,
        status: str,
        box_source: str,
        visible: bool,
        box: list[int] | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.decisions[item.key] = {
            **self._base_decision(item),
            "status": status,
            "box_source": box_source,
            "visible": visible,
            "bbox_xyxy": box,
            **(extra or {}),
        }
        self._save_outputs()

    def accept_insid3(self, item_index: float | int) -> int:
        with self.lock:
            item = self._activate(item_index)
            if item.proposal_box is None:
                raise RuntimeError("This item has no INSID3 box to accept")
            self._record_decision(
                item,
                status="accepted_insid3",
                box_source="insid3_verified",
                visible=True,
                box=item.proposal_box,
            )
            return self.next_pending(item_index)

    def accept_sam2(self, item_index: float | int) -> int:
        with self.lock:
            item = self._activate(item_index)
            mask = self._selected_mask()
            box = mask_to_xyxy(mask) if mask is not None else None
            if mask is None or box is None:
                raise RuntimeError("Create and select a non-empty SAM2 mask first")
            mask_path = self.sam_mask_root / f"{item.key}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            self._record_decision(
                item,
                status="accepted_sam2",
                box_source="sam2_manual_points",
                visible=True,
                box=box,
                extra={
                    "sam_mask_path": str(mask_path.relative_to(self.root)),
                    "sam_candidate": self.selected_candidate + 1,
                    "sam_predicted_iou": float(self.candidate_scores[self.selected_candidate]),
                    "points": [
                        {"x": x, "y": y, "label": "positive" if label == 1 else "negative"}
                        for x, y, label in self.points
                    ],
                },
            )
            return self.next_pending(item_index)

    def accept_manual_box(self, item_index: float | int) -> int:
        with self.lock:
            item = self._activate(item_index)
            box = self._manual_box(item)
            if box is None:
                raise RuntimeError("Click two distinct opposite box corners first")
            self._record_decision(
                item,
                status="manual_box",
                box_source="manual_box",
                visible=True,
                box=box,
                extra={"corners_xy": [list(point) for point in self.manual_corners]},
            )
            return self.next_pending(item_index)

    def mark_not_visible(self, item_index: float | int) -> int:
        with self.lock:
            item = self._activate(item_index)
            self._record_decision(
                item,
                status="not_visible",
                box_source="reviewed_not_visible",
                visible=False,
                box=None,
            )
            return self.next_pending(item_index)

    def clear_decision(self, item_index: float | int) -> int:
        with self.lock:
            item = self._activate(item_index)
            self.decisions.pop(item.key, None)
            self._save_outputs()
            return int(item_index)

    def next_pending(self, item_index: float | int) -> int:
        start = int(item_index)
        for offset in range(1, len(self.items) + 1):
            candidate = (start + offset) % len(self.items)
            if self.items[candidate].key not in self.decisions:
                return candidate
        return start


def sanitize_proxy_environment() -> None:
    for name in ("ALL_PROXY", "all_proxy"):
        if os.environ.get(name, "").lower().startswith("socks://"):
            os.environ.pop(name, None)


def build_app(reviewer: INSID3BoxReviewer, initial_item: int) -> Any:
    sanitize_proxy_environment()
    import gradio as gr

    with gr.Blocks(title="INSID3 Box Review") as app:
        with gr.Row():
            previous_button = gr.Button("Previous")
            item_index = gr.Slider(
                0,
                len(reviewer.items) - 1,
                value=initial_item,
                step=1,
                label="Review item",
            )
            next_button = gr.Button("Next")
            next_pending_button = gr.Button("Next pending")

        instruction = gr.Markdown()
        task = gr.Textbox(label="Task prompt", interactive=False)
        next_step = gr.Textbox(label="Current stage instruction", interactive=False)
        image = gr.Image(type="numpy", label="Keyframe and box proposals", interactive=False, height=600)

        with gr.Row():
            click_mode = gr.Radio(
                ["Positive", "Negative", "Box corner"],
                value="Positive",
                label="Click mode",
            )
            candidate = gr.Radio(["1", "2", "3"], value="1", label="SAM2 candidate")
            undo_button = gr.Button("Undo SAM point")
            clear_correction_button = gr.Button("Clear correction")

        with gr.Row():
            accept_insid3_button = gr.Button("Accept INSID3", variant="primary")
            accept_sam2_button = gr.Button("Accept SAM2")
            accept_manual_button = gr.Button("Save manual box")
            not_visible_button = gr.Button("Not visible")
            clear_decision_button = gr.Button("Clear decision")

        status = gr.Textbox(label="Review status", interactive=False, lines=5)
        progress = gr.Textbox(label="Progress", interactive=False)
        item_outputs = [instruction, task, next_step, image, candidate, status, progress]

        def add_click(item_value: float, mode: str, event: Any):
            return reviewer.add_click(item_value, mode, event)

        add_click.__annotations__["event"] = gr.SelectData

        def save_and_load(action: str, item_value: float):
            next_index = int(getattr(reviewer, action)(item_value))
            return gr.update(value=next_index), *reviewer.load_item(next_index)

        app.load(reviewer.load_item, inputs=item_index, outputs=item_outputs)
        item_index.change(
            reviewer.load_item,
            inputs=item_index,
            outputs=item_outputs,
            show_progress="hidden",
        )
        image.select(
            add_click,
            inputs=[item_index, click_mode],
            outputs=[image, status, candidate],
            concurrency_limit=1,
        )
        candidate.change(
            reviewer.choose_candidate,
            inputs=[item_index, candidate],
            outputs=[image, status],
            show_progress="hidden",
        )
        undo_button.click(
            reviewer.undo_point,
            inputs=item_index,
            outputs=[image, status, candidate],
            concurrency_limit=1,
        )
        clear_correction_button.click(
            reviewer.clear_correction,
            inputs=item_index,
            outputs=[image, status, candidate],
        )

        action_outputs = [item_index, *item_outputs]
        accept_insid3_button.click(
            lambda value: save_and_load("accept_insid3", value),
            inputs=item_index,
            outputs=action_outputs,
        )
        accept_sam2_button.click(
            lambda value: save_and_load("accept_sam2", value),
            inputs=item_index,
            outputs=action_outputs,
            concurrency_limit=1,
        )
        accept_manual_button.click(
            lambda value: save_and_load("accept_manual_box", value),
            inputs=item_index,
            outputs=action_outputs,
        )
        not_visible_button.click(
            lambda value: save_and_load("mark_not_visible", value),
            inputs=item_index,
            outputs=action_outputs,
        )
        clear_decision_button.click(
            lambda value: save_and_load("clear_decision", value),
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
            lambda value: min(len(reviewer.items) - 1, int(value) + 1),
            inputs=item_index,
            outputs=item_index,
            show_progress="hidden",
        )
        next_pending_button.click(
            reviewer.next_pending,
            inputs=item_index,
            outputs=item_index,
            show_progress="hidden",
        )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--insid3-root", type=Path, default=DEFAULT_INSID3_ROOT)
    parser.add_argument("--review-path", type=Path)
    parser.add_argument("--model-name", default=DEFAULT_SAM2_MODEL)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, or cpu")
    parser.add_argument("--initial-item", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7869)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    reviewer = INSID3BoxReviewer(
        args.insid3_root,
        args.model_name,
        args.device,
        args.review_path,
    )
    if not 0 <= args.initial_item < len(reviewer.items):
        raise ValueError(f"--initial-item must be in 0..{len(reviewer.items) - 1}")
    logger.info("Review decisions: %s", reviewer.review_path)
    logger.info("Reviewed boxes: %s", reviewer.review_table_path)
    app = build_app(reviewer, args.initial_item)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
