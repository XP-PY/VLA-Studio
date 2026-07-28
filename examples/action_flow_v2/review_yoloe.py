#!/usr/bin/env python3
"""Review YOLOE points and save corrections for automatic SAM2 processing.

The application treats YOLOE's selected point as the default. For an incorrect
result, either choose another candidate and accept its point, or click the
correct target directly in that candidate frame. The frame slider can be used
when the target is not visible in any retained candidate. Every decision is
saved immediately to ``<yoloe-root>/review/decisions.json``.

Example:

python examples/action_flow_v2/review_yoloe.py \
    --yoloe-root runs/yoloe/rm65_single_table_preparation_and_bussing \
    --host 127.0.0.1 \
    --port 7867
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import pandas as pd

try:
    from .annotate_grounding import EpisodeVideoSource
except ImportError:
    from annotate_grounding import EpisodeVideoSource  # type: ignore[no-redef]


DEFAULT_YOLOE_ROOT = Path("runs/yoloe/rm65_single_table_preparation_and_bussing")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReviewItem:
    episode_index: int
    camera: str
    object_id: int
    object_name: str
    object_role: str
    task: str

    @property
    def key(self) -> str:
        return f"{self.episode_index}/{self.camera}/{self.object_id}"

    @property
    def label(self) -> str:
        return (
            f"Episode {self.episode_index} | {self.camera} | "
            f"ID {self.object_id} | {self.object_name}"
        )


class YOLOEPointReviewer:
    """Load YOLOE proposals, render source frames, and persist point overrides."""

    def __init__(self, yoloe_root: Path, review_path: Path | None = None):
        self.root = yoloe_root.resolve()
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(info_path)
        self.info = json.loads(info_path.read_text())
        self.dataset_root = Path(self.info["dataset_root"]).resolve()
        self.source = EpisodeVideoSource(self.dataset_root)
        self.review_path = (
            review_path.resolve()
            if review_path is not None
            else self.root / "review" / "decisions.json"
        )
        self.lock = threading.RLock()

        selected_paths = sorted((self.root / "data").glob("episode-*/*.parquet"))
        candidate_paths = sorted((self.root / "candidates").glob("episode-*/*.parquet"))
        if not selected_paths or not candidate_paths:
            raise FileNotFoundError(
                f"YOLOE selected or candidate tables are missing under {self.root}"
            )
        self.selected = pd.concat(
            (pd.read_parquet(path) for path in selected_paths),
            ignore_index=True,
        )
        self.candidates = pd.concat(
            (pd.read_parquet(path) for path in candidate_paths),
            ignore_index=True,
        )
        required = {
            "episode_index",
            "camera",
            "object_id",
            "object_name",
            "frame_index",
            "visible",
            "point_x",
            "point_y",
            "candidate_rank",
        }
        for name, table in (("selected", self.selected), ("candidates", self.candidates)):
            missing = required.difference(table.columns)
            if missing:
                raise ValueError(f"{name} table is missing columns: {sorted(missing)}")

        self.selected = self.selected.sort_values(
            ["episode_index", "camera", "object_id"]
        ).reset_index(drop=True)
        self.candidates = self.candidates.sort_values(
            ["episode_index", "camera", "object_id", "candidate_rank"]
        ).reset_index(drop=True)
        self.items = [
            ReviewItem(
                episode_index=int(row.episode_index),
                camera=str(row.camera),
                object_id=int(row.object_id),
                object_name=str(row.object_name),
                object_role=str(row.object_role),
                task=str(row.task),
            )
            for row in self.selected.itertuples(index=False)
        ]
        if not self.items:
            raise ValueError(f"No YOLOE review items found under {self.root}")

        self.decisions = self._load_decisions()
        if not self.review_path.is_file():
            self._save_decisions()
        self.cached_video_key: tuple[int, str] | None = None
        self.cached_frames: list[np.ndarray] = []

    def _load_decisions(self) -> dict[str, dict[str, Any]]:
        if not self.review_path.is_file():
            return {}
        payload = json.loads(self.review_path.read_text())
        if int(payload.get("format_version", 0)) != 1:
            raise ValueError(f"Unsupported review format in {self.review_path}")
        saved_root = Path(payload["yoloe_root"]).resolve()
        if saved_root != self.root:
            raise ValueError(
                f"Review belongs to {saved_root}, but --yoloe-root is {self.root}"
            )
        return {
            f"{int(item['episode_index'])}/{item['camera']}/{int(item['object_id'])}": item
            for item in payload.get("decisions", [])
        }

    def _save_decisions(self) -> None:
        payload = {
            "format_version": 1,
            "role": "reviewed YOLOE positive-point prompts for SAM2",
            "yoloe_root": str(self.root),
            "dataset_root": str(self.dataset_root),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "default": "Use the selected YOLOE point when no decision is recorded.",
            "decisions": sorted(
                self.decisions.values(),
                key=lambda item: (
                    int(item["episode_index"]),
                    str(item["camera"]),
                    int(item["object_id"]),
                ),
            ),
        }
        self.review_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.review_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(self.review_path)

    def _item(self, item_index: float | int) -> ReviewItem:
        index = int(item_index)
        if not 0 <= index < len(self.items):
            raise gr.Error(f"Review item {index} is out of range")
        return self.items[index]

    def _rows_for_item(self, table: pd.DataFrame, item: ReviewItem) -> pd.DataFrame:
        return table.loc[
            (table["episode_index"] == item.episode_index)
            & (table["camera"] == item.camera)
            & (table["object_id"] == item.object_id)
        ].sort_values("candidate_rank")

    def _selected_row(self, item: ReviewItem) -> pd.Series:
        rows = self._rows_for_item(self.selected, item)
        if len(rows) != 1:
            raise RuntimeError(f"Expected one selected row for {item.key}, got {len(rows)}")
        return rows.iloc[0]

    def _candidate_row(self, item: ReviewItem, candidate_rank: int) -> pd.Series | None:
        rows = self._rows_for_item(self.candidates, item)
        rows = rows.loc[rows["candidate_rank"] == int(candidate_rank)]
        if len(rows) != 1:
            return None
        return rows.iloc[0]

    @staticmethod
    def _row_has_point(row: pd.Series | None) -> bool:
        return bool(
            row is not None
            and row["visible"]
            and int(row["frame_index"]) >= 0
            and np.isfinite([row["point_x"], row["point_y"]]).all()
        )

    def _load_frames(self, item: ReviewItem) -> list[np.ndarray]:
        key = (item.episode_index, item.camera)
        if key == self.cached_video_key:
            return self.cached_frames
        logger.info("Decoding episode %d camera %s", *key)
        frames = [
            np.asarray(frame.convert("RGB"), dtype=np.uint8)
            for _, _, frame in self.source.sampled_frames(
                item.episode_index,
                item.camera,
                frame_stride=1,
                max_sampled_frames=0,
            )
        ]
        expected = int(self.source.row(item.episode_index)["length"])
        if len(frames) != expected:
            raise RuntimeError(
                f"Decoded {len(frames)} frames for {key}, expected {expected}"
            )
        self.cached_video_key = key
        self.cached_frames = frames
        return frames

    def _default_rank(self, item: ReviewItem) -> int:
        decision = self.decisions.get(item.key)
        if decision is not None and decision.get("candidate_rank") is not None:
            return int(decision["candidate_rank"])
        row = self._selected_row(item)
        return int(row["candidate_rank"]) if np.isfinite(row["candidate_rank"]) else 1

    def _default_frame(self, item: ReviewItem, candidate_rank: int) -> int:
        decision = self.decisions.get(item.key)
        if decision is not None and decision.get("frame_index") is not None:
            return int(decision["frame_index"])
        row = self._candidate_row(item, candidate_rank)
        if row is None:
            row = self._selected_row(item)
        return max(0, int(row["frame_index"])) if bool(row["visible"]) else 0

    @staticmethod
    def _draw_cross(
        image: np.ndarray,
        point: tuple[int, int],
        color: tuple[int, int, int],
        radius: int = 9,
    ) -> None:
        x, y = point
        cv2.circle(image, (x, y), radius, color, 2, lineType=cv2.LINE_AA)
        cv2.line(image, (x - radius, y), (x + radius, y), color, 2, cv2.LINE_AA)
        cv2.line(image, (x, y - radius), (x, y + radius), color, 2, cv2.LINE_AA)

    def render(
        self,
        item_index: float | int,
        candidate_rank: float | int,
        frame_index: float | int,
    ) -> np.ndarray:
        item = self._item(item_index)
        frames = self._load_frames(item)
        frame_index = int(np.clip(int(frame_index), 0, len(frames) - 1))
        image = frames[frame_index].copy()

        row = self._candidate_row(item, int(candidate_rank))
        if row is not None and bool(row["visible"]) and int(row["frame_index"]) == frame_index:
            coordinates = [row[f"box_{axis}"] for axis in ("x0", "y0", "x1", "y1")]
            if np.isfinite(coordinates).all():
                x0, y0, x1, y1 = (int(round(value)) for value in coordinates)
                cv2.rectangle(image, (x0, y0), (x1, y1), (255, 190, 0), 2)
            if self._row_has_point(row):
                self._draw_cross(
                    image,
                    (int(round(row["point_x"])), int(round(row["point_y"]))),
                    (255, 190, 0),
                )

        decision = self.decisions.get(item.key)
        if (
            decision is not None
            and decision.get("status") in {"manual_point", "candidate_point"}
            and int(decision["frame_index"]) == frame_index
        ):
            self._draw_cross(
                image,
                (int(decision["point_x"]), int(decision["point_y"])),
                (0, 255, 80),
                radius=12,
            )
        return image

    def _progress(self) -> str:
        decided = len(self.decisions)
        return f"{decided}/{len(self.items)} explicit review decisions saved"

    def _status(self, item: ReviewItem, candidate_rank: int, frame_index: int) -> str:
        decision = self.decisions.get(item.key)
        row = self._candidate_row(item, candidate_rank)
        candidate_text = (
            f"candidate {candidate_rank}: frame {int(row['frame_index'])}, "
            f"confidence {float(row['score']):.3f}"
            if self._row_has_point(row)
            else f"candidate {candidate_rank}: unavailable"
        )
        if decision is None:
            decision_text = "No override: SAM2 will use the selected YOLOE point."
        elif decision["status"] == "not_visible":
            decision_text = "Saved as not visible: SAM2 will create an all-zero track."
        else:
            decision_text = (
                f"Saved {decision['status']} at frame {decision['frame_index']}, "
                f"point ({decision['point_x']}, {decision['point_y']})."
            )
        return (
            f"{item.label}\n{candidate_text}\n"
            f"showing frame {frame_index}\n{decision_text}\n{self._progress()}"
        )

    def load_item(
        self,
        item_index: float | int,
    ) -> tuple[str, str, int, Any, np.ndarray, str]:
        with self.lock:
            item = self._item(item_index)
            frames = self._load_frames(item)
            rank = self._default_rank(item)
            frame_index = self._default_frame(item, rank)
            return (
                item.label,
                item.task,
                rank,
                gr.update(value=frame_index, maximum=len(frames) - 1),
                self.render(item_index, rank, frame_index),
                self._status(item, rank, frame_index),
            )

    def select_candidate(
        self,
        item_index: float | int,
        candidate_rank: float | int,
    ) -> tuple[Any, np.ndarray, str]:
        with self.lock:
            item = self._item(item_index)
            rank = int(candidate_rank)
            frames = self._load_frames(item)
            frame_index = self._default_frame(item, rank)
            frame_index = int(np.clip(frame_index, 0, len(frames) - 1))
            return (
                gr.update(value=frame_index, maximum=len(frames) - 1),
                self.render(item_index, rank, frame_index),
                self._status(item, rank, frame_index),
            )

    def show_frame(
        self,
        item_index: float | int,
        candidate_rank: float | int,
        frame_index: float | int,
    ) -> tuple[np.ndarray, str]:
        with self.lock:
            item = self._item(item_index)
            frame_index = int(frame_index)
            return (
                self.render(item_index, candidate_rank, frame_index),
                self._status(item, int(candidate_rank), frame_index),
            )

    def _base_decision(self, item: ReviewItem) -> dict[str, Any]:
        return {
            "episode_index": item.episode_index,
            "camera": item.camera,
            "object_id": item.object_id,
            "object_name": item.object_name,
            "object_role": item.object_role,
        }

    def use_candidate_point(
        self,
        item_index: float | int,
        candidate_rank: float | int,
    ) -> tuple[Any, np.ndarray, str]:
        with self.lock:
            item = self._item(item_index)
            rank = int(candidate_rank)
            row = self._candidate_row(item, rank)
            if not self._row_has_point(row):
                raise gr.Error(f"Candidate {rank} has no usable YOLOE point")
            assert row is not None
            frame_index = int(row["frame_index"])
            self.decisions[item.key] = {
                **self._base_decision(item),
                "status": "candidate_point",
                "candidate_rank": rank,
                "frame_index": frame_index,
                "point_x": int(round(row["point_x"])),
                "point_y": int(round(row["point_y"])),
                "point_label": 1,
            }
            self._save_decisions()
            return (
                gr.update(value=frame_index),
                self.render(item_index, rank, frame_index),
                self._status(item, rank, frame_index),
            )

    def save_clicked_point(
        self,
        item_index: float | int,
        candidate_rank: float | int,
        frame_index: float | int,
        event: gr.SelectData,
    ) -> tuple[np.ndarray, str]:
        with self.lock:
            item = self._item(item_index)
            frame_index = int(frame_index)
            x, y = (int(value) for value in event.index[:2])
            frame = self._load_frames(item)[frame_index]
            height, width = frame.shape[:2]
            if not (0 <= x < width and 0 <= y < height):
                raise gr.Error(f"Point ({x}, {y}) is outside the {width}x{height} frame")
            self.decisions[item.key] = {
                **self._base_decision(item),
                "status": "manual_point",
                "candidate_rank": int(candidate_rank),
                "frame_index": frame_index,
                "point_x": x,
                "point_y": y,
                "point_label": 1,
            }
            self._save_decisions()
            return (
                self.render(item_index, candidate_rank, frame_index),
                self._status(item, int(candidate_rank), frame_index),
            )

    def mark_not_visible(
        self,
        item_index: float | int,
        candidate_rank: float | int,
        frame_index: float | int,
    ) -> tuple[np.ndarray, str]:
        with self.lock:
            item = self._item(item_index)
            self.decisions[item.key] = {
                **self._base_decision(item),
                "status": "not_visible",
                "candidate_rank": None,
                "frame_index": None,
                "point_x": None,
                "point_y": None,
                "point_label": None,
            }
            self._save_decisions()
            return (
                self.render(item_index, candidate_rank, frame_index),
                self._status(item, int(candidate_rank), int(frame_index)),
            )

    def clear_decision(
        self,
        item_index: float | int,
        candidate_rank: float | int,
        frame_index: float | int,
    ) -> tuple[np.ndarray, str]:
        with self.lock:
            item = self._item(item_index)
            self.decisions.pop(item.key, None)
            self._save_decisions()
            return (
                self.render(item_index, candidate_rank, frame_index),
                self._status(item, int(candidate_rank), int(frame_index)),
            )


def build_app(reviewer: YOLOEPointReviewer, initial_item: int) -> gr.Blocks:
    with gr.Blocks(title="YOLOE Point Review") as app:
        with gr.Row():
            previous_button = gr.Button("Previous")
            item_index = gr.Slider(
                minimum=0,
                maximum=len(reviewer.items) - 1,
                value=initial_item,
                step=1,
                label="Review item",
            )
            next_button = gr.Button("Next")

        item_label = gr.Textbox(label="Target", interactive=False)
        task = gr.Textbox(label="Task prompt", interactive=False)
        with gr.Row():
            candidate_rank = gr.Radio(
                choices=[1, 2, 3],
                value=1,
                label="Candidate",
            )
            frame_index = gr.Slider(
                minimum=0,
                maximum=1,
                value=0,
                step=1,
                label="Frame",
            )

        image = gr.Image(
            type="numpy",
            label="Click the correct target to save a positive SAM2 point",
            interactive=False,
            height=600,
        )
        with gr.Row():
            use_candidate_button = gr.Button("Use YOLOE point", variant="primary")
            not_visible_button = gr.Button("Not visible")
            clear_button = gr.Button("Clear override")
        status = gr.Textbox(label="Review status", interactive=False, lines=5)

        outputs = [item_label, task, candidate_rank, frame_index, image, status]
        app.load(reviewer.load_item, inputs=[item_index], outputs=outputs)
        item_index.change(
            reviewer.load_item,
            inputs=[item_index],
            outputs=outputs,
            show_progress="hidden",
        )
        candidate_rank.change(
            reviewer.select_candidate,
            inputs=[item_index, candidate_rank],
            outputs=[frame_index, image, status],
            show_progress="hidden",
        )
        frame_index.change(
            reviewer.show_frame,
            inputs=[item_index, candidate_rank, frame_index],
            outputs=[image, status],
            show_progress="hidden",
        )
        image.select(
            reviewer.save_clicked_point,
            inputs=[item_index, candidate_rank, frame_index],
            outputs=[image, status],
        )
        use_candidate_button.click(
            reviewer.use_candidate_point,
            inputs=[item_index, candidate_rank],
            outputs=[frame_index, image, status],
        )
        not_visible_button.click(
            reviewer.mark_not_visible,
            inputs=[item_index, candidate_rank, frame_index],
            outputs=[image, status],
        )
        clear_button.click(
            reviewer.clear_decision,
            inputs=[item_index, candidate_rank, frame_index],
            outputs=[image, status],
        )
        previous_button.click(
            lambda value: max(0, int(value) - 1),
            inputs=[item_index],
            outputs=[item_index],
            show_progress="hidden",
        )
        next_button.click(
            lambda value: min(len(reviewer.items) - 1, int(value) + 1),
            inputs=[item_index],
            outputs=[item_index],
            show_progress="hidden",
        )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yoloe-root", type=Path, default=DEFAULT_YOLOE_ROOT)
    parser.add_argument(
        "--review-path",
        type=Path,
        help="Defaults to <yoloe-root>/review/decisions.json.",
    )
    parser.add_argument("--initial-item", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7867)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    reviewer = YOLOEPointReviewer(args.yoloe_root, args.review_path)
    if not 0 <= args.initial_item < len(reviewer.items):
        raise ValueError(
            f"--initial-item must be between 0 and {len(reviewer.items) - 1}"
        )
    logger.info("Review decisions: %s", reviewer.review_path)
    app = build_app(reviewer, args.initial_item)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
