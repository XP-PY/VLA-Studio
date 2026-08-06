#!/usr/bin/env python3
"""Generate coarse next-step planning segments from RM65 gripper motion.

The source LeRobot dataset is never modified. This script detects sustained
gripper closing/opening transitions, pairs them into manipulation cycles, and
writes a sidecar planning dataset under runs/planning. Each cycle contains two
macro stages, "grasp" and "put"; only the interval after the final release is
labeled "complete".

The generated Parquet files are proposals. Review and correction are performed
with review_planning_segments.py, which stores edits separately under the
output root's review directory.

Example:

    python examples/action_flow_v2/annotate_planning_segments.py \
        --dataset-root runs/dataset/rm65_single_table_preparation_and_bussing \
        --output-root runs/planning/rm65_single_table_preparation_and_bussing \
        --episodes all
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = WORKSPACE / "runs/dataset/rm65_single_table_preparation_and_bussing"
DEFAULT_OUTPUT_ROOT = WORKSPACE / "runs/planning/rm65_single_table_preparation_and_bussing"

PLACE_PATTERN = re.compile(
    r"place the (tomato|garlic|corn|cucumber) on the (pink|yellow|blue) plate",
    flags=re.IGNORECASE,
)
CLEAR_PATTERN = re.compile(
    r"put the vegetable from the (pink|yellow|blue) plate into the collection box",
    flags=re.IGNORECASE,
)
BUS_TASK = "put all vegetables into the collection box"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransitionEvent:
    """One large monotonic gripper transition in episode-local frame indices."""

    start_frame: int
    end_frame: int
    direction: str
    delta: float


@dataclass(frozen=True)
class DetectedCycle:
    """One close/hold/open pair converted into planning boundaries."""

    hold_start_frame: int
    release_frame: int
    close_event: TransitionEvent
    open_event: TransitionEvent


@dataclass(frozen=True)
class CyclePlan:
    """Semantic source and destination for one manipulation cycle."""

    source_entity: str
    action_object: str
    destination_entity: str
    preposition: str


def parse_episode_selection(value: str, available: list[int]) -> list[int]:
    """Parse "all", comma-separated indices, and half-open start:stop ranges."""

    if value.strip().lower() == "all":
        return available
    selected: set[int] = set()
    available_set = set(available)
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            parts = token.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid episode range {token!r}; use start:stop")
            start = int(parts[0]) if parts[0] else min(available)
            stop = int(parts[1]) if parts[1] else max(available) + 1
            selected.update(range(start, stop))
        else:
            selected.add(int(token))
    missing = sorted(selected.difference(available_set))
    if missing:
        raise ValueError(f"Requested episodes are absent from the dataset: {missing}")
    return sorted(selected)


def write_parquet_atomic(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


class LeRobotPlanningSource:
    """Read episode metadata and gripper trajectories from a LeRobot v3 dataset."""

    def __init__(self, root: Path, gripper_name: str):
        self.root = root.resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        episode_paths = sorted((self.root / "meta/episodes").glob("chunk-*/*.parquet"))
        if not episode_paths:
            raise FileNotFoundError(f"No episode metadata found under {self.root}")
        episodes = pd.concat((pd.read_parquet(path) for path in episode_paths), ignore_index=True)
        self.episodes = episodes.set_index("episode_index", drop=False).sort_index()
        self.fps = int(self.info["fps"])

        state_names = list(self.info["features"]["observation.state"]["names"])
        if gripper_name not in state_names:
            raise ValueError(
                f"Gripper feature {gripper_name!r} is absent from observation.state: {state_names}"
            )
        self.gripper_index = state_names.index(gripper_name)
        self.gripper_name = gripper_name
        self._cached_data_path: Path | None = None
        self._cached_data: pd.DataFrame | None = None

    @property
    def episode_indices(self) -> list[int]:
        return [int(index) for index in self.episodes.index]

    def row(self, episode_index: int) -> pd.Series:
        return self.episodes.loc[episode_index]

    def task(self, episode_index: int) -> str:
        tasks = self.row(episode_index)["tasks"]
        if isinstance(tasks, str):
            return tasks
        if len(tasks) != 1:
            raise ValueError(f"Episode {episode_index} has {len(tasks)} tasks; expected one")
        return str(tasks[0])

    def _data_table(self, episode_index: int) -> pd.DataFrame:
        row = self.row(episode_index)
        relative_path = self.info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        path = self.root / relative_path
        if path != self._cached_data_path:
            self._cached_data = pd.read_parquet(
                path,
                columns=["episode_index", "frame_index", "observation.state"],
            )
            self._cached_data_path = path
        assert self._cached_data is not None
        return self._cached_data

    def gripper_values(self, episode_index: int) -> np.ndarray:
        data = self._data_table(episode_index)
        episode = data.loc[data["episode_index"] == episode_index].sort_values("frame_index")
        expected_length = int(self.row(episode_index)["length"])
        if len(episode) != expected_length:
            raise RuntimeError(
                f"Episode {episode_index} has {len(episode)} data rows; expected {expected_length}"
            )
        frame_indices = episode["frame_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(frame_indices, np.arange(expected_length)):
            raise RuntimeError(f"Episode {episode_index} frame indices are not contiguous")
        return np.asarray(
            [state[self.gripper_index] for state in episode["observation.state"]],
            dtype=np.float64,
        )


def smooth_gripper(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0 or window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")
    return (
        pd.Series(values)
        .rolling(window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float64)
    )


def detect_transition_events(
    smoothed: np.ndarray,
    *,
    minimum_rate: float,
    transition_gap_frames: int,
    minimum_amplitude: float,
    same_direction_merge_gap: int,
) -> list[TransitionEvent]:
    """Detect and merge large signed transitions in a smoothed gripper trace."""

    derivatives = np.diff(smoothed)
    active = np.flatnonzero(np.abs(derivatives) >= minimum_rate)
    groups: list[tuple[int, int]] = []
    if len(active):
        start = previous = int(active[0])
        for raw_index in active[1:]:
            index = int(raw_index)
            if index - previous > transition_gap_frames:
                groups.append((start, previous + 1))
                start = index
            previous = index
        groups.append((start, previous + 1))

    events: list[TransitionEvent] = []
    for start, end in groups:
        delta = float(smoothed[end] - smoothed[start])
        if abs(delta) < minimum_amplitude:
            continue
        direction = "close" if delta < 0 else "open"
        event = TransitionEvent(start, end, direction, delta)
        if (
            events
            and events[-1].direction == direction
            and event.start_frame - events[-1].end_frame <= same_direction_merge_gap
        ):
            previous = events[-1]
            events[-1] = TransitionEvent(
                start_frame=previous.start_frame,
                end_frame=event.end_frame,
                direction=direction,
                delta=float(smoothed[event.end_frame] - smoothed[previous.start_frame]),
            )
        else:
            events.append(event)
    return events


def pair_manipulation_cycles(
    events: list[TransitionEvent],
    *,
    episode_length: int,
    minimum_hold_frames: int,
) -> tuple[list[DetectedCycle], list[str]]:
    """Pair close/open transitions and return cycle-level review diagnostics."""

    cycles: list[DetectedCycle] = []
    diagnostics: list[str] = []
    pending_close: TransitionEvent | None = None
    for event in events:
        if event.direction == "close":
            if pending_close is not None:
                diagnostics.append("consecutive_close_events")
            pending_close = event
            continue
        if pending_close is None:
            continue

        hold_start = min(episode_length, pending_close.end_frame + 1)
        release_frame = min(episode_length, event.end_frame + 1)
        hold_frames = event.start_frame - hold_start
        if hold_frames < minimum_hold_frames:
            diagnostics.append("short_hold_interval")
        if hold_start >= release_frame:
            diagnostics.append("invalid_cycle_boundaries")
        else:
            cycles.append(
                DetectedCycle(
                    hold_start_frame=hold_start,
                    release_frame=release_frame,
                    close_event=pending_close,
                    open_event=event,
                )
            )
        pending_close = None

    if pending_close is not None:
        diagnostics.append("unmatched_close_event")
    return cycles, diagnostics


def parse_cycle_plans(task: str, detected_cycles: int) -> tuple[str, list[CyclePlan], int | None]:
    """Map a task prompt to ordered cycle semantics and expected cycle count."""

    placements = PLACE_PATTERN.findall(task)
    if placements:
        plans = [
            CyclePlan(
                source_entity=vegetable.lower(),
                action_object=vegetable.lower(),
                destination_entity=f"{color.lower()} plate",
                preposition="on",
            )
            for vegetable, color in placements
        ]
        return ("arrange" if len(plans) > 1 else "place", plans, len(plans))

    clear_match = CLEAR_PATTERN.fullmatch(task.rstrip("."))
    if clear_match:
        plate = f"{clear_match.group(1).lower()} plate"
        return (
            "clear",
            [CyclePlan(f"vegetable on the {plate}", "vegetable", "collection box", "in")],
            1,
        )

    if task.rstrip(".").lower() == BUS_TASK:
        plans = [
            CyclePlan("{target}", "{target}", "collection box", "in")
            for _ in range(detected_cycles)
        ]
        return "bus", plans, None

    raise ValueError(f"Unsupported table-preparation task prompt: {task!r}")


def with_article(entity: str) -> str:
    return entity if entity.startswith("{") else f"the {entity}"


def grasp_plan(plan: CyclePlan) -> str:
    return f"Grasp {with_article(plan.source_entity)}."


def put_plan(plan: CyclePlan) -> str:
    return (
        f"Put {with_article(plan.action_object)} {plan.preposition} "
        f"{with_article(plan.destination_entity)}."
    )


def build_segments(
    *,
    episode_index: int,
    episode_length: int,
    cycles: list[DetectedCycle],
    plans: list[CyclePlan],
) -> list[dict[str, Any]]:
    """Build contiguous half-open grasp/put/complete intervals."""

    rows: list[dict[str, Any]] = []
    cursor = 0
    for cycle_index, cycle in enumerate(cycles):
        plan = (
            plans[cycle_index]
            if cycle_index < len(plans)
            else CyclePlan("{target}", "{target}", "{destination}", "on")
        )
        hold_start = max(cursor + 1, cycle.hold_start_frame)
        release_frame = max(hold_start + 1, cycle.release_frame)
        release_frame = min(release_frame, episode_length)
        rows.append(
            {
                "episode_index": episode_index,
                "segment_index": len(rows),
                "cycle_index": cycle_index,
                "start_frame": cursor,
                "end_frame": hold_start,
                "stage": "grasp",
                "source_entity": plan.source_entity,
                "destination_entity": "",
                "next_step": grasp_plan(plan),
            }
        )
        rows.append(
            {
                "episode_index": episode_index,
                "segment_index": len(rows),
                "cycle_index": cycle_index,
                "start_frame": hold_start,
                "end_frame": release_frame,
                "stage": "put",
                "source_entity": plan.source_entity,
                "destination_entity": plan.destination_entity,
                "next_step": put_plan(plan),
            }
        )
        cursor = release_frame

    if cursor < episode_length:
        rows.append(
            {
                "episode_index": episode_index,
                "segment_index": len(rows),
                "cycle_index": -1,
                "start_frame": cursor,
                "end_frame": episode_length,
                "stage": "complete",
                "source_entity": "",
                "destination_entity": "",
                "next_step": "Task complete.",
            }
        )
    for row in rows:
        row["segment_uid"] = (
            f"episode-{episode_index:06d}-segment-{int(row['segment_index']):03d}"
        )
    return rows


def validate_segments(rows: list[dict[str, Any]], episode_length: int) -> None:
    if not rows:
        raise ValueError("An episode must contain at least one planning segment")
    if int(rows[0]["start_frame"]) != 0:
        raise ValueError("The first segment must start at frame 0")
    if int(rows[-1]["end_frame"]) != episode_length:
        raise ValueError("The final segment must end at the episode length")
    for index, row in enumerate(rows):
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        if not 0 <= start < end <= episode_length:
            raise ValueError(f"Invalid segment interval [{start}, {end})")
        if index and int(rows[index - 1]["end_frame"]) != start:
            raise ValueError("Planning segments must be contiguous")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episodes", default="all")
    parser.add_argument("--gripper-name", default="gripper.pos")
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--minimum-rate", type=float, default=2.0)
    parser.add_argument("--transition-gap-frames", type=int, default=12)
    parser.add_argument("--minimum-amplitude", type=float, default=100.0)
    parser.add_argument("--same-direction-merge-gap", type=int, default=30)
    parser.add_argument("--minimum-hold-frames", type=int, default=10)
    parser.add_argument("--maximum-bus-cycles", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.minimum_rate <= 0 or args.minimum_amplitude <= 0:
        raise ValueError("Transition rate and amplitude must be positive")
    if args.transition_gap_frames < 0 or args.same_direction_merge_gap < 0:
        raise ValueError("Transition gap values must be non-negative")
    if args.minimum_hold_frames < 1:
        raise ValueError("--minimum-hold-frames must be positive")

    output_root = args.output_root.resolve()
    baseline_path = output_root / "data/segments.parquet"
    if baseline_path.exists() and not args.overwrite:
        raise FileExistsError(f"{baseline_path} exists; pass --overwrite to regenerate it")

    source = LeRobotPlanningSource(args.dataset_root, args.gripper_name)
    episode_indices = parse_episode_selection(args.episodes, source.episode_indices)
    segment_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []

    for ordinal, episode_index in enumerate(episode_indices, start=1):
        task = source.task(episode_index)
        gripper = source.gripper_values(episode_index)
        smoothed = smooth_gripper(gripper, args.smoothing_window)
        events = detect_transition_events(
            smoothed,
            minimum_rate=args.minimum_rate,
            transition_gap_frames=args.transition_gap_frames,
            minimum_amplitude=args.minimum_amplitude,
            same_direction_merge_gap=args.same_direction_merge_gap,
        )
        cycles, diagnostics = pair_manipulation_cycles(
            events,
            episode_length=len(gripper),
            minimum_hold_frames=args.minimum_hold_frames,
        )
        task_kind, plans, expected_cycles = parse_cycle_plans(task, len(cycles))
        if expected_cycles is not None and len(cycles) != expected_cycles:
            diagnostics.append(f"expected_{expected_cycles}_cycles_detected_{len(cycles)}")
        if task_kind == "bus" and not 1 <= len(cycles) <= args.maximum_bus_cycles:
            diagnostics.append(f"bus_detected_{len(cycles)}_cycles")
        if not cycles:
            diagnostics.append("no_complete_cycles")

        rows = build_segments(
            episode_index=episode_index,
            episode_length=len(gripper),
            cycles=cycles,
            plans=plans,
        )
        validate_segments(rows, len(gripper))
        segment_rows.extend(rows)
        for event_index, event in enumerate(events):
            transition_rows.append(
                {
                    "episode_index": episode_index,
                    "event_index": event_index,
                    **asdict(event),
                }
            )
        episode_rows.append(
            {
                "episode_index": episode_index,
                "task": task,
                "task_kind": task_kind,
                "length": len(gripper),
                "expected_cycles": expected_cycles,
                "detected_cycles": len(cycles),
                "detection_status": "OK" if not diagnostics else "REVIEW",
                "diagnostics": json.dumps(sorted(set(diagnostics))),
                "gripper_min": float(gripper.min()),
                "gripper_max": float(gripper.max()),
            }
        )
        if ordinal % 50 == 0 or ordinal == len(episode_indices):
            logger.info("Processed %d/%d episodes", ordinal, len(episode_indices))

    segments = pd.DataFrame(segment_rows)
    episodes = pd.DataFrame(episode_rows)
    transitions = pd.DataFrame(transition_rows)
    write_parquet_atomic(baseline_path, segments)
    write_parquet_atomic(output_root / "meta/episodes.parquet", episodes)
    write_parquet_atomic(output_root / "data/transitions.parquet", transitions)

    parameters = {
        "gripper_name": args.gripper_name,
        "smoothing_window": args.smoothing_window,
        "minimum_rate": args.minimum_rate,
        "transition_gap_frames": args.transition_gap_frames,
        "minimum_amplitude": args.minimum_amplitude,
        "same_direction_merge_gap": args.same_direction_merge_gap,
        "minimum_hold_frames": args.minimum_hold_frames,
        "maximum_bus_cycles": args.maximum_bus_cycles,
    }
    info = {
        "format": "rm65_next_step_planning_segments",
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(source.root),
        "fps": source.fps,
        "interval_semantics": "start_frame inclusive, end_frame exclusive",
        "episode_count": len(episodes),
        "segment_count": len(segments),
        "review_episode_count": int((episodes["detection_status"] != "OK").sum()),
        "parameters": parameters,
        "baseline_segments": "data/segments.parquet",
        "episode_diagnostics": "meta/episodes.parquet",
        "transition_events": "data/transitions.parquet",
    }
    write_json_atomic(output_root / "meta/info.json", info)
    logger.info(
        "Wrote %d segments for %d episodes to %s (%d flagged for review).",
        len(segments),
        len(episodes),
        output_root,
        info["review_episode_count"],
    )


if __name__ == "__main__":
    main()
