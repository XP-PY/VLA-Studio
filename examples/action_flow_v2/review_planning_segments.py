#!/usr/bin/env python3
"""Review and correct automatically generated next-step planning segments.

The application shows the top and wrist views together with the gripper trace.
It can edit segment boundaries and plan text, split or delete segments, and fill
the ``{target}`` placeholder used by bussing episodes. Every edit is saved
immediately under ``<planning-root>/review``; the automatic proposal under
``data`` and ``meta`` is never modified.

Example:

    python examples/action_flow_v2/review_planning_segments.py \
        --planning-root runs/planning/rm65_single_table_preparation_and_bussing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import av

# httpx rejects the nonstandard ``socks://`` scheme before Gradio starts. The
# configured HTTP/HTTPS proxies remain available when this fallback is removed.
for proxy_variable in ("ALL_PROXY", "all_proxy"):
    if os.environ.get(proxy_variable, "").lower().startswith("socks://"):
        os.environ.pop(proxy_variable)

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .annotate_planning_segments import (
        DEFAULT_OUTPUT_ROOT,
        LeRobotPlanningSource,
        smooth_gripper,
        validate_segments,
        write_parquet_atomic,
    )
except ImportError:
    from annotate_planning_segments import (  # type: ignore[no-redef]
        DEFAULT_OUTPUT_ROOT,
        LeRobotPlanningSource,
        smooth_gripper,
        validate_segments,
        write_parquet_atomic,
    )


logger = logging.getLogger(__name__)
STAGES = ["grasp", "put", "complete"]
SEGMENT_COLUMNS = [
    "episode_index",
    "segment_index",
    "cycle_index",
    "start_frame",
    "end_frame",
    "stage",
    "source_entity",
    "destination_entity",
    "next_step",
    "segment_uid",
]


class PlanningSegmentReviewer:
    """Serve dataset frames and persist corrected planning annotations."""

    def __init__(self, planning_root: Path):
        self.root = planning_root.resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.dataset_root = Path(self.info["dataset_root"]).resolve()
        gripper_name = str(self.info["parameters"]["gripper_name"])
        self.source = LeRobotPlanningSource(self.dataset_root, gripper_name)
        self.fps = int(self.info["fps"])
        self.smoothing_window = int(self.info["parameters"]["smoothing_window"])
        self.lock = threading.RLock()

        baseline_segments = pd.read_parquet(self.root / "data/segments.parquet")
        baseline_episodes = pd.read_parquet(self.root / "meta/episodes.parquet")
        self.review_segments_path = self.root / "review/segments.parquet"
        self.review_episodes_path = self.root / "review/episodes.parquet"

        if self.review_segments_path.is_file():
            self.segments = pd.read_parquet(self.review_segments_path)
        else:
            self.segments = baseline_segments.copy()
            write_parquet_atomic(self.review_segments_path, self.segments)

        if self.review_episodes_path.is_file():
            self.episodes = pd.read_parquet(self.review_episodes_path)
        else:
            self.episodes = baseline_episodes.copy()
            self.episodes["reviewed"] = False
            self.episodes["review_notes"] = ""
            self.episodes["updated_at"] = ""
            write_parquet_atomic(self.review_episodes_path, self.episodes)

        self.segments = self.segments.sort_values(
            ["episode_index", "segment_index"]
        ).reset_index(drop=True)
        self.episodes = self.episodes.sort_values("episode_index").reset_index(drop=True)
        self.episode_indices = self.episodes["episode_index"].astype(int).tolist()
        if not self.episode_indices:
            raise ValueError(f"No episodes found under {self.root}")

        segment_episodes = set(self.segments["episode_index"].astype(int))
        if segment_episodes != set(self.episode_indices):
            raise ValueError("Review segment and episode tables cover different episodes")

        self.camera_keys = [
            key
            for key, feature in self.source.info["features"].items()
            if feature.get("dtype") == "video"
        ]
        self.camera_by_name = {key.rsplit(".", 1)[-1]: key for key in self.camera_keys}
        if "top" not in self.camera_by_name or "wrist" not in self.camera_by_name:
            raise ValueError(
                "The reviewer expects observation.images.top and observation.images.wrist"
            )

    def _episode_index(self, episode_position: float | int) -> int:
        position = int(episode_position)
        if not 0 <= position < len(self.episode_indices):
            raise gr.Error(f"Episode position {position} is out of range")
        return self.episode_indices[position]

    def _episode_row(self, episode_index: int) -> pd.Series:
        rows = self.episodes.loc[self.episodes["episode_index"] == episode_index]
        if len(rows) != 1:
            raise RuntimeError(f"Expected one metadata row for episode {episode_index}")
        return rows.iloc[0]

    def _segment_rows(self, episode_index: int) -> pd.DataFrame:
        return self.segments.loc[
            self.segments["episode_index"] == episode_index
        ].sort_values("segment_index").reset_index(drop=True)

    def _segment_row(self, episode_index: int, segment_index: float | int) -> pd.Series:
        rows = self._segment_rows(episode_index)
        index = int(segment_index)
        if not 0 <= index < len(rows):
            raise gr.Error(f"Segment {index} is out of range for episode {episode_index}")
        return rows.iloc[index]

    def _video_path_and_time(
        self, episode_index: int, camera_name: str, frame_index: int
    ) -> tuple[Path, float]:
        camera_key = self.camera_by_name[camera_name]
        row = self.source.row(episode_index)
        prefix = f"videos/{camera_key}"
        path = self.source.info["video_path"].format(
            video_key=camera_key,
            chunk_index=int(row[f"{prefix}/chunk_index"]),
            file_index=int(row[f"{prefix}/file_index"]),
        )
        start_time = float(row[f"{prefix}/from_timestamp"])
        return self.dataset_root / path, start_time + frame_index / self.fps

    @lru_cache(maxsize=256)
    def frame(self, episode_index: int, camera_name: str, frame_index: int) -> np.ndarray:
        """Decode one episode-local video frame without retaining whole videos."""

        length = int(self._episode_row(episode_index)["length"])
        frame_index = int(np.clip(frame_index, 0, length - 1))
        path, target_time = self._video_path_and_time(
            episode_index, camera_name, frame_index
        )
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            seek_time = max(0.0, target_time - 1.0)
            container.seek(
                int(seek_time / time_base),
                stream=stream,
                backward=True,
                any_frame=False,
            )
            nearest: tuple[float, np.ndarray] | None = None
            for video_frame in container.decode(stream):
                if video_frame.pts is None:
                    continue
                timestamp = float(video_frame.pts * stream.time_base)
                distance = abs(timestamp - target_time)
                image = video_frame.to_ndarray(format="rgb24")
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, image)
                if timestamp >= target_time:
                    break
        if nearest is None:
            raise RuntimeError(f"Could not decode frame {frame_index} from {path}")
        return nearest[1]

    def _save_segments(self) -> None:
        self.segments = self.segments.sort_values(
            ["episode_index", "segment_index"]
        ).reset_index(drop=True)
        write_parquet_atomic(self.review_segments_path, self.segments)

    def _save_episodes(self) -> None:
        write_parquet_atomic(self.review_episodes_path, self.episodes)

    @staticmethod
    def _renumber(rows: list[dict[str, Any]], episode_index: int) -> None:
        for index, row in enumerate(rows):
            row["episode_index"] = episode_index
            row["segment_index"] = index
            row["segment_uid"] = f"episode-{episode_index:06d}-segment-{index:03d}"

    def _replace_episode_segments(
        self, episode_index: int, rows: list[dict[str, Any]]
    ) -> None:
        length = int(self._episode_row(episode_index)["length"])
        self._renumber(rows, episode_index)
        validate_segments(rows, length)
        remaining = self.segments.loc[self.segments["episode_index"] != episode_index]
        replacement = pd.DataFrame(rows, columns=SEGMENT_COLUMNS)
        self.segments = pd.concat([remaining, replacement], ignore_index=True)
        self._save_segments()

    def _render_trace(
        self, episode_index: int, segment_index: int, frame_index: int
    ) -> Any:
        raw = self.source.gripper_values(episode_index)
        smoothed = smooth_gripper(raw, self.smoothing_window)
        rows = self._segment_rows(episode_index)
        colors = {"grasp": "#d64545", "put": "#2d70b3", "complete": "#3b8f59"}
        figure, axis = plt.subplots(figsize=(11, 3.0), constrained_layout=True)
        axis.plot(raw, color="#a9afb8", linewidth=0.8, label="raw")
        axis.plot(smoothed, color="#20242a", linewidth=1.3, label="smoothed")
        for row in rows.itertuples(index=False):
            alpha = 0.25 if int(row.segment_index) == segment_index else 0.09
            axis.axvspan(
                int(row.start_frame),
                int(row.end_frame),
                color=colors.get(str(row.stage), "#777777"),
                alpha=alpha,
            )
        axis.axvline(frame_index, color="#111111", linestyle="--", linewidth=1)
        axis.set(xlabel="Episode-local frame", ylabel="Gripper position")
        axis.legend(loc="upper right", ncols=2, frameon=False)
        return figure

    def _status(self, episode_index: int, segment_index: int, frame_index: int) -> str:
        episode = self._episode_row(episode_index)
        rows = self._segment_rows(episode_index)
        reviewed_count = int(self.episodes["reviewed"].fillna(False).astype(bool).sum())
        return (
            f"Episode {episode_index}: segment {segment_index + 1}/{len(rows)}, "
            f"frame {frame_index}/{int(episode['length']) - 1}. "
            f"Reviewed {reviewed_count}/{len(self.episodes)} episodes. "
            f"Edits: {self.review_segments_path}"
        )

    def _visual_outputs(
        self, episode_index: int, segment_index: int, frame_index: int
    ) -> tuple[np.ndarray, np.ndarray, Any, str]:
        return (
            self.frame(episode_index, "top", frame_index),
            self.frame(episode_index, "wrist", frame_index),
            self._render_trace(episode_index, segment_index, frame_index),
            self._status(episode_index, segment_index, frame_index),
        )

    def load_episode(self, episode_position: float | int) -> tuple[Any, ...]:
        with self.lock:
            episode_index = self._episode_index(episode_position)
            episode = self._episode_row(episode_index)
            rows = self._segment_rows(episode_index)
            segment = rows.iloc[0]
            frame_index = (int(segment["start_frame"]) + int(segment["end_frame"]) - 1) // 2
            top, wrist, trace, status = self._visual_outputs(episode_index, 0, frame_index)
            diagnostics = json.loads(str(episode["diagnostics"]))
            return (
                f"Episode {episode_index} ({int(episode_position) + 1}/{len(self.episodes)})",
                str(episode["task"]),
                f"{episode['detection_status']}: {', '.join(diagnostics) if diagnostics else 'none'}",
                gr.update(value=0, maximum=len(rows) - 1),
                gr.update(value=frame_index, maximum=int(episode["length"]) - 1),
                top,
                wrist,
                trace,
                int(segment["start_frame"]),
                int(segment["end_frame"]),
                int(segment["cycle_index"]),
                str(segment["stage"]),
                str(segment["source_entity"]),
                str(segment["destination_entity"]),
                str(segment["next_step"]),
                bool(episode.get("reviewed", False)),
                str(episode.get("review_notes", "")),
                status,
            )

    def load_segment(
        self, episode_position: float | int, segment_index: float | int
    ) -> tuple[Any, ...]:
        with self.lock:
            episode_index = self._episode_index(episode_position)
            segment_index = int(segment_index)
            segment = self._segment_row(episode_index, segment_index)
            frame_index = (int(segment["start_frame"]) + int(segment["end_frame"]) - 1) // 2
            top, wrist, trace, status = self._visual_outputs(
                episode_index, segment_index, frame_index
            )
            return (
                gr.update(value=frame_index),
                top,
                wrist,
                trace,
                int(segment["start_frame"]),
                int(segment["end_frame"]),
                int(segment["cycle_index"]),
                str(segment["stage"]),
                str(segment["source_entity"]),
                str(segment["destination_entity"]),
                str(segment["next_step"]),
                status,
            )

    def show_frame(
        self,
        episode_position: float | int,
        segment_index: float | int,
        frame_index: float | int,
    ) -> tuple[np.ndarray, np.ndarray, Any, str]:
        with self.lock:
            episode_index = self._episode_index(episode_position)
            return self._visual_outputs(episode_index, int(segment_index), int(frame_index))

    def save_segment(
        self,
        episode_position: float | int,
        segment_index: float | int,
        start_frame: float | int,
        end_frame: float | int,
        cycle_index: float | int,
        stage: str,
        source_entity: str,
        destination_entity: str,
        next_step: str,
    ) -> tuple[Any, ...]:
        with self.lock:
            episode_index = self._episode_index(episode_position)
            index = int(segment_index)
            rows = self._segment_rows(episode_index).to_dict("records")
            length = int(self._episode_row(episode_index)["length"])
            start = int(start_frame)
            end = int(end_frame)
            if start >= end:
                raise gr.Error("Start frame must be smaller than end frame")

            if index == 0:
                start = 0
            else:
                minimum_start = int(rows[index - 1]["start_frame"]) + 1
                if minimum_start >= end:
                    raise gr.Error("The edited range would make the previous segment empty")
                start = int(np.clip(start, minimum_start, end - 1))
                rows[index - 1]["end_frame"] = start
            if index == len(rows) - 1:
                end = length
            else:
                maximum_end = int(rows[index + 1]["end_frame"]) - 1
                if start + 1 > maximum_end:
                    raise gr.Error("The edited range would make the next segment empty")
                end = int(np.clip(end, start + 1, maximum_end))
                rows[index + 1]["start_frame"] = end

            row = rows[index]
            row.update(
                start_frame=start,
                end_frame=end,
                cycle_index=int(cycle_index),
                stage=str(stage),
                source_entity=str(source_entity).strip(),
                destination_entity=str(destination_entity).strip(),
                next_step=str(next_step).strip(),
            )
            self._replace_episode_segments(episode_index, rows)
            return self.load_segment(episode_position, index)

    def split_segment(
        self,
        episode_position: float | int,
        segment_index: float | int,
        frame_index: float | int,
    ) -> tuple[Any, ...]:
        with self.lock:
            episode_index = self._episode_index(episode_position)
            index = int(segment_index)
            split = int(frame_index)
            rows = self._segment_rows(episode_index).to_dict("records")
            start = int(rows[index]["start_frame"])
            end = int(rows[index]["end_frame"])
            if not start < split < end:
                raise gr.Error(f"Split frame must be inside [{start}, {end})")
            second = dict(rows[index])
            rows[index]["end_frame"] = split
            second["start_frame"] = split
            rows.insert(index + 1, second)
            self._replace_episode_segments(episode_index, rows)
            outputs = self.load_segment(episode_position, index + 1)
            return (gr.update(value=index + 1, maximum=len(rows) - 1), *outputs)

    def delete_segment(
        self, episode_position: float | int, segment_index: float | int
    ) -> tuple[Any, ...]:
        with self.lock:
            episode_index = self._episode_index(episode_position)
            index = int(segment_index)
            rows = self._segment_rows(episode_index).to_dict("records")
            if len(rows) == 1:
                raise gr.Error("The only segment in an episode cannot be deleted")
            deleted = rows.pop(index)
            if index > 0:
                rows[index - 1]["end_frame"] = int(deleted["end_frame"])
                new_index = index - 1
            else:
                rows[0]["start_frame"] = 0
                new_index = 0
            self._replace_episode_segments(episode_index, rows)
            outputs = self.load_segment(episode_position, new_index)
            return (gr.update(value=new_index, maximum=len(rows) - 1), *outputs)

    def set_cycle_target(
        self,
        episode_position: float | int,
        segment_index: float | int,
        target: str,
    ) -> tuple[Any, ...]:
        with self.lock:
            target = str(target).strip()
            if not target:
                raise gr.Error("Enter the bussing target name first")
            episode_index = self._episode_index(episode_position)
            current = self._segment_row(episode_index, segment_index)
            cycle_index = int(current["cycle_index"])
            if cycle_index < 0:
                raise gr.Error("The complete segment does not belong to a manipulation cycle")
            rows = self._segment_rows(episode_index).to_dict("records")
            article_target = target if target.startswith("{") else f"the {target}"
            for row in rows:
                if int(row["cycle_index"]) != cycle_index:
                    continue
                row["source_entity"] = target
                if row["stage"] == "grasp":
                    row["destination_entity"] = ""
                    row["next_step"] = f"Grasp {article_target}."
                elif row["stage"] == "put":
                    row["destination_entity"] = "collection box"
                    row["next_step"] = f"Put {article_target} in the collection box."
            self._replace_episode_segments(episode_index, rows)
            return self.load_segment(episode_position, int(segment_index))

    def save_episode_review(
        self,
        episode_position: float | int,
        reviewed: bool,
        notes: str,
        segment_index: float | int,
        frame_index: float | int,
    ) -> str:
        with self.lock:
            episode_index = self._episode_index(episode_position)
            mask = self.episodes["episode_index"] == episode_index
            self.episodes.loc[mask, "reviewed"] = bool(reviewed)
            self.episodes.loc[mask, "review_notes"] = str(notes).strip()
            self.episodes.loc[mask, "updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save_episodes()
            return self._status(episode_index, int(segment_index), int(frame_index))


def build_app(reviewer: PlanningSegmentReviewer, initial_episode: int) -> gr.Blocks:
    with gr.Blocks(title="RM65 Planning Segment Review") as app:
        with gr.Row():
            previous_episode = gr.Button("Previous episode")
            episode_position = gr.Slider(
                0,
                len(reviewer.episode_indices) - 1,
                value=initial_episode,
                step=1,
                label="Episode",
            )
            next_episode = gr.Button("Next episode")

        episode_label = gr.Textbox(label="Episode", interactive=False)
        task = gr.Textbox(label="Task prompt", interactive=False)
        diagnostics = gr.Textbox(label="Automatic detection", interactive=False)

        with gr.Row():
            previous_segment = gr.Button("Previous segment")
            segment_index = gr.Slider(0, 1, value=0, step=1, label="Segment")
            next_segment = gr.Button("Next segment")
        frame_index = gr.Slider(0, 1, value=0, step=1, label="Preview frame")

        with gr.Row():
            top_image = gr.Image(label="Top camera", type="numpy", height=420)
            wrist_image = gr.Image(label="Wrist camera", type="numpy", height=420)
        gripper_trace = gr.Plot(label="Gripper motion and planning ranges")

        with gr.Row():
            start_frame = gr.Number(label="Start frame (inclusive)", precision=0)
            end_frame = gr.Number(label="End frame (exclusive)", precision=0)
            cycle_index = gr.Number(label="Cycle index", precision=0)
            stage = gr.Dropdown(STAGES, label="Stage")
        with gr.Row():
            source_entity = gr.Textbox(label="Source entity")
            destination_entity = gr.Textbox(label="Destination entity")
        next_step = gr.Textbox(label="Next-step plan")

        with gr.Row():
            save_segment = gr.Button("Save segment", variant="primary")
            split_segment = gr.Button("Split at preview frame")
            delete_segment = gr.Button("Delete segment")
        with gr.Row():
            cycle_target = gr.Textbox(label="Bussing cycle target")
            set_cycle_target = gr.Button("Apply target to grasp and put")

        with gr.Row():
            reviewed = gr.Checkbox(label="Episode reviewed")
            review_notes = gr.Textbox(label="Review notes")
            save_review = gr.Button("Save episode review")
        status = gr.Textbox(label="Status", interactive=False)

        episode_outputs = [
            episode_label,
            task,
            diagnostics,
            segment_index,
            frame_index,
            top_image,
            wrist_image,
            gripper_trace,
            start_frame,
            end_frame,
            cycle_index,
            stage,
            source_entity,
            destination_entity,
            next_step,
            reviewed,
            review_notes,
            status,
        ]
        segment_outputs = [
            frame_index,
            top_image,
            wrist_image,
            gripper_trace,
            start_frame,
            end_frame,
            cycle_index,
            stage,
            source_entity,
            destination_entity,
            next_step,
            status,
        ]
        structure_outputs = [segment_index, *segment_outputs]

        app.load(reviewer.load_episode, [episode_position], episode_outputs)
        episode_position.change(
            reviewer.load_episode,
            [episode_position],
            episode_outputs,
            show_progress="hidden",
        )
        segment_index.change(
            reviewer.load_segment,
            [episode_position, segment_index],
            segment_outputs,
            show_progress="hidden",
        )
        frame_index.change(
            reviewer.show_frame,
            [episode_position, segment_index, frame_index],
            [top_image, wrist_image, gripper_trace, status],
            show_progress="hidden",
        )
        save_segment.click(
            reviewer.save_segment,
            [
                episode_position,
                segment_index,
                start_frame,
                end_frame,
                cycle_index,
                stage,
                source_entity,
                destination_entity,
                next_step,
            ],
            segment_outputs,
        )
        split_segment.click(
            reviewer.split_segment,
            [episode_position, segment_index, frame_index],
            structure_outputs,
        )
        delete_segment.click(
            reviewer.delete_segment,
            [episode_position, segment_index],
            structure_outputs,
        )
        set_cycle_target.click(
            reviewer.set_cycle_target,
            [episode_position, segment_index, cycle_target],
            segment_outputs,
        )
        save_review.click(
            reviewer.save_episode_review,
            [episode_position, reviewed, review_notes, segment_index, frame_index],
            [status],
        )

        previous_episode.click(
            lambda value: max(0, int(value) - 1),
            [episode_position],
            [episode_position],
            show_progress="hidden",
        )
        next_episode.click(
            lambda value: min(len(reviewer.episode_indices) - 1, int(value) + 1),
            [episode_position],
            [episode_position],
            show_progress="hidden",
        )
        previous_segment.click(
            lambda value: max(0, int(value) - 1),
            [segment_index],
            [segment_index],
            show_progress="hidden",
        )
        next_segment.click(
            lambda episode, value: min(
                len(reviewer._segment_rows(reviewer._episode_index(episode))) - 1,
                int(value) + 1,
            ),
            [episode_position, segment_index],
            [segment_index],
            show_progress="hidden",
        )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--initial-episode", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7868)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    reviewer = PlanningSegmentReviewer(args.planning_root)
    if not 0 <= args.initial_episode < len(reviewer.episode_indices):
        raise ValueError(
            f"--initial-episode must be between 0 and {len(reviewer.episode_indices) - 1}"
        )
    logger.info("Reviewed segments: %s", reviewer.review_segments_path)
    app = build_app(reviewer, args.initial_episode)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
