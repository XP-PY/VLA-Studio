#!/usr/bin/env python3
"""Annotate task-relevance masks in LeRobot videos with reviewed DINO boxes and SAM2.

Brief user manual:
1. Start the app and open the printed local URL in a browser.
2. Load an episode and camera. Reviewed Grounding DINO boxes are imported
   automatically when no saved annotation workspace exists.
3. Inspect each box at its initialization frame. Add positive/negative clicks
   to refine it, or clear that object's prompts and initialize it manually.
4. For the wrist camera, optionally set an object's final visible frame with
   "Set object end". Resolve every item reported as manual or missing.
5. Select "Propagate and save". Inspect the resulting mask with the frame
   slider and add correction clicks before propagating again if tracking drifts.

Use ``--mode batch --episodes all --cameras top wrist`` to process all fully
resolved episode-camera pairs. Batch mode skips unresolved pairs and writes a
report under the SAM metadata directory for subsequent manual correction.

Per-object SAM2 tracks, labels, clicks, and previews are saved under
``runs/sam`` for later experiments. Training rows under ``runs/dataset``
contain one merged binary relevance mask and one loss weight per frame and
camera. An all-zero mask is a valid negative label; excluded frames use a zero
loss weight.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from transformers import Sam2VideoModel, Sam2VideoProcessor


DEFAULT_DATASET_ROOT = Path("runs/dataset/rm65_single_table_preparation_and_bussing")
DEFAULT_OUTPUT_ROOT = Path("runs/dataset/rm65_single_table_preparation_and_bussing_afv2")
DEFAULT_SAM_ROOT = Path("runs/sam/rm65_single_table_preparation_and_bussing")
DEFAULT_MODEL_PATH = Path("models/sam2.1-hiera-small")
DEFAULT_GROUNDING_ROOT = DEFAULT_SAM_ROOT / "grounding_dino"
DEFAULT_GROUNDING_REVIEW = DEFAULT_SAM_ROOT / "grounding_dino_review.md"
CAMERAS = ("top", "wrist")
OVERLAY_COLOR = np.array([0, 210, 170], dtype=np.float32)

logger = logging.getLogger(__name__)


def encode_mask_runs(mask: np.ndarray) -> bytes:
    """Encode a binary mask as compressed uint32 ``(start, length)`` pairs."""
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    padded = np.concatenate((np.zeros(1, dtype=np.uint8), flat, np.zeros(1, dtype=np.uint8)))
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    runs = transitions.reshape(-1, 2).astype(np.uint32)
    runs[:, 1] -= runs[:, 0]
    return zlib.compress(runs.tobytes())


def decode_mask_runs(payload: bytes, height: int, width: int) -> np.ndarray:
    """Decode a mask produced by :func:`encode_mask_runs`."""
    mask = np.zeros(height * width, dtype=bool)
    if payload:
        runs = np.frombuffer(zlib.decompress(payload), dtype=np.uint32).reshape(-1, 2)
        for start, length in runs:
            mask[int(start) : int(start + length)] = True
    return mask.reshape(height, width)


def overlay_mask(frame: np.ndarray, mask: np.ndarray | None, alpha: float = 0.45) -> np.ndarray:
    """Return an RGB frame with a translucent relevance-mask overlay."""
    image = np.asarray(frame, dtype=np.uint8).copy()
    if mask is None or not np.any(mask):
        return image
    pixels = image[mask].astype(np.float32)
    image[mask] = np.clip((1.0 - alpha) * pixels + alpha * OVERLAY_COLOR, 0, 255).astype(np.uint8)
    return image


class EpisodeVideoSource:
    """Read an episode's camera segment directly from LeRobot v3 video metadata."""

    def __init__(self, dataset_root: Path):
        self.root = dataset_root.resolve()
        with (self.root / "meta" / "info.json").open() as file:
            self.info = json.load(file)

        episode_files = sorted((self.root / "meta" / "episodes").glob("chunk-*/*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"No episode metadata found under {self.root}")
        episodes = pd.concat((pd.read_parquet(path) for path in episode_files), ignore_index=True)
        self.episodes = episodes.set_index("episode_index", drop=False)
        self.fps = int(self.info["fps"])

    @property
    def episode_indices(self) -> list[int]:
        return sorted(int(index) for index in self.episodes.index)

    def prompt(self, episode_index: int) -> str:
        tasks = self.episodes.loc[episode_index, "tasks"]
        if isinstance(tasks, str):
            return tasks
        if len(tasks) != 1:
            raise ValueError(f"Episode {episode_index} has {len(tasks)} tasks; expected exactly one")
        return str(tasks[0])

    def frames(self, episode_index: int, camera: str) -> list[np.ndarray]:
        if camera not in CAMERAS:
            raise ValueError(f"Unsupported camera {camera!r}; expected one of {CAMERAS}")
        row = self.episodes.loc[episode_index]
        prefix = f"videos/observation.images.{camera}"
        chunk_index = int(row[f"{prefix}/chunk_index"])
        file_index = int(row[f"{prefix}/file_index"])
        start_s = float(row[f"{prefix}/from_timestamp"])
        expected_frames = int(row["length"])
        video_key = f"observation.images.{camera}"
        relative_path = self.info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        video_path = self.root / relative_path
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        decoded: list[np.ndarray] = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if start_s > 0:
                seek_pts = max(0, int((start_s - 1.0 / self.fps) / float(stream.time_base)))
                container.seek(seek_pts, stream=stream, backward=True)

            tolerance_s = 0.5 / self.fps
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                timestamp_s = float(frame.pts * stream.time_base)
                if timestamp_s < start_s - tolerance_s:
                    continue
                decoded.append(frame.to_ndarray(format="rgb24"))
                if len(decoded) == expected_frames:
                    break

        if len(decoded) != expected_frames:
            raise RuntimeError(
                f"Decoded {len(decoded)} frames for episode {episode_index} camera {camera}, "
                f"but metadata expects {expected_frames}"
            )
        return decoded


@dataclass
class ObjectBounds:
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class GroundingDecision:
    """One optional review override for a DINO proposal."""

    status: str
    candidate_rank: int | None
    manual_frame_index: int | None


@dataclass(frozen=True)
class GroundingBoxPrompt:
    """One reviewed box prompt ready for a SAM2 video session."""

    object_id: int
    object_name: str
    object_role: str
    frame_index: int
    box_xyxy: tuple[float, float, float, float]
    source: str


@dataclass(frozen=True)
class GroundingResolution:
    """Resolved prompts and items requiring manual intervention."""

    prompts: tuple[GroundingBoxPrompt, ...]
    unresolved: dict[int, str]
    intentionally_not_visible: dict[int, str]


class GroundingDINOReview:
    """Resolve selected DINO boxes using candidate choices from the review table."""

    VALID_STATUSES = {"accept", "use_candidate", "manual", "not_visible"}

    def __init__(self, grounding_root: Path, review_path: Path):
        self.root = grounding_root.resolve()
        self.review_path = review_path.resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if not self.review_path.is_file():
            raise FileNotFoundError(self.review_path)
        self.decisions = self._load_decisions()

    @staticmethod
    def _optional_int(value: str) -> int | None:
        value = value.strip()
        return int(value) if value else None

    def _load_decisions(self) -> dict[tuple[int, str, int], GroundingDecision]:
        text = self.review_path.read_text()
        if "## Review Results" not in text:
            raise ValueError(f"Review Results section is missing from {self.review_path}")
        results = text.split("## Review Results", 1)[1]
        results = results.split("## Dataset Reindexing Note", 1)[0]
        decisions: dict[tuple[int, str, int], GroundingDecision] = {}
        for line in results.splitlines():
            if not re.match(r"^\|\s*\d+\s*\|", line):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 9:
                raise ValueError(f"Invalid review row with {len(cells)} columns: {line}")
            episode_index = int(cells[0])
            camera = cells[1]
            object_id = int(cells[2])
            status = cells[4]
            if camera not in CAMERAS:
                raise ValueError(f"Invalid review camera {camera!r}: {line}")
            if status not in self.VALID_STATUSES:
                raise ValueError(f"Invalid review status {status!r}: {line}")
            decision = GroundingDecision(
                status=status,
                candidate_rank=self._optional_int(cells[5]),
                manual_frame_index=self._optional_int(cells[6]),
            )
            key = (episode_index, camera, object_id)
            if key in decisions:
                logger.warning("Duplicate review decision for %s; using the last row", key)
            decisions[key] = decision
        return decisions

    def _paths(self, episode_index: int, camera: str) -> tuple[Path, Path]:
        episode_dir = f"episode-{episode_index:06d}"
        return (
            self.root / "data" / episode_dir / f"camera-{camera}.parquet",
            self.root / "candidates" / episode_dir / f"camera-{camera}.parquet",
        )

    @staticmethod
    def _box_prompt(row: pd.Series, source: str) -> GroundingBoxPrompt | None:
        coordinates = tuple(float(row[f"box_{axis}"]) for axis in ("x0", "y0", "x1", "y1"))
        if not bool(row["visible"]) or int(row["frame_index"]) < 0 or not np.isfinite(coordinates).all():
            return None
        x0, y0, x1, y1 = coordinates
        if x1 <= x0 or y1 <= y0:
            return None
        return GroundingBoxPrompt(
            object_id=int(row["object_id"]),
            object_name=str(row["object_name"]),
            object_role=str(row["object_role"]),
            frame_index=int(row["frame_index"]),
            box_xyxy=coordinates,
            source=source,
        )

    def resolve(self, episode_index: int, camera: str) -> GroundingResolution:
        """Resolve one episode-camera pair into reviewed SAM2 box prompts."""
        selected_path, candidates_path = self._paths(episode_index, camera)
        if not selected_path.is_file() or not candidates_path.is_file():
            raise FileNotFoundError(f"Grounding artifacts are missing for episode {episode_index} {camera}")
        selected = pd.read_parquet(selected_path).sort_values("object_id")
        candidates = pd.read_parquet(candidates_path).sort_values(["object_id", "candidate_rank"])

        prompts = []
        unresolved = {}
        intentionally_not_visible = {}
        for row in selected.itertuples(index=False):
            object_id = int(row.object_id)
            object_name = str(row.object_name)
            key = (episode_index, camera, object_id)
            decision = self.decisions.get(key)
            if decision is not None and decision.status == "not_visible":
                intentionally_not_visible[object_id] = f"{object_name}: marked not_visible in review"
                continue
            if decision is not None and decision.status == "manual":
                frame_hint = (
                    f" at frame {decision.manual_frame_index}"
                    if decision.manual_frame_index is not None
                    else ""
                )
                unresolved[object_id] = f"{object_name}: manual initialization required{frame_hint}"
                continue

            selected_row = selected.loc[selected["object_id"] == object_id].iloc[0]
            source = "selected"
            proposal_row = selected_row
            if decision is not None and decision.status == "use_candidate":
                if decision.candidate_rank is None:
                    unresolved[object_id] = f"{object_name}: use_candidate has no candidate rank"
                    continue
                matches = candidates.loc[
                    (candidates["object_id"] == object_id)
                    & (candidates["candidate_rank"] == decision.candidate_rank)
                ]
                if len(matches) != 1:
                    unresolved[object_id] = (
                        f"{object_name}: candidate {decision.candidate_rank} is unavailable"
                    )
                    continue
                proposal_row = matches.iloc[0]
                source = f"reviewed candidate {decision.candidate_rank}"

            prompt = self._box_prompt(proposal_row, source)
            if prompt is None:
                unresolved[object_id] = (
                    f"{object_name}: selected DINO proposal is unavailable; add a manual click"
                )
                continue
            prompts.append(prompt)

        return GroundingResolution(tuple(prompts), unresolved, intentionally_not_visible)


class TaskRelevanceAnnotator:
    """Own the SAM2 session and persist per-object tracks plus merged masks."""

    def __init__(
        self,
        dataset_root: Path,
        output_root: Path,
        sam_root: Path,
        model_path: Path,
        device: str,
        grounding_root: Path | None = None,
        grounding_review: Path | None = None,
        auto_grounding: bool = True,
    ):
        self.source = EpisodeVideoSource(dataset_root)
        self.output_root = output_root.resolve()
        self.sam_root = sam_root.resolve()
        self.model_path = model_path.resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(self.model_path)
        self.grounding_review = None
        if auto_grounding:
            if grounding_root is None or grounding_review is None:
                raise ValueError("grounding_root and grounding_review are required with auto grounding")
            self.grounding_review = GroundingDINOReview(grounding_root, grounding_review)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float32
        if self.device.type == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model: Sam2VideoModel | None = None
        self.processor: Sam2VideoProcessor | None = None
        self.session: Any | None = None
        self.lock = threading.RLock()

        self.episode_index: int | None = None
        self.camera: str | None = None
        self.task_prompt = ""
        self.video_frames: list[np.ndarray] = []
        self.prompts: dict[int, dict[int, list[tuple[int, int, int]]]] = {}
        self.boxes: dict[int, dict[int, tuple[float, float, float, float]]] = {}
        self.box_sources: dict[int, dict[int, str]] = {}
        self.object_labels: dict[int, str] = {}
        self.bounds: dict[int, ObjectBounds] = {}
        self.grounding_unresolved: dict[int, str] = {}
        self.grounding_not_visible: dict[int, str] = {}
        self.excluded_frames: set[int] = set()
        self.propagated_object_masks: dict[int, list[np.ndarray]] | None = None
        self.propagated_masks: list[np.ndarray] | None = None
        self.preview_mask: np.ndarray | None = None

        self._write_metadata()

    def _write_metadata(self) -> None:
        merged_info = {
            "format_version": 1,
            "base_dataset_root": str(self.source.root),
            "sam_annotation_root": str(self.sam_root),
            "grounding_root": str(self.grounding_review.root) if self.grounding_review else None,
            "grounding_review": (
                str(self.grounding_review.review_path) if self.grounding_review else None
            ),
            "fps": self.source.fps,
            "cameras": list(CAMERAS),
            "mask_encoding": "zlib-compressed uint32 start-length pairs in row-major order",
            "data_path": "data/episode-{episode_index:06d}/camera-{camera}.parquet",
            "training_fields": ["episode_index", "frame_index", "camera", "mask_runs", "loss_weight"],
        }
        sam_info = {
            "format_version": 1,
            "base_dataset_root": str(self.source.root),
            "merged_dataset_root": str(self.output_root),
            "sam_model_path": str(self.model_path),
            "grounding_root": str(self.grounding_review.root) if self.grounding_review else None,
            "grounding_review": (
                str(self.grounding_review.review_path) if self.grounding_review else None
            ),
            "fps": self.source.fps,
            "cameras": list(CAMERAS),
            "mask_encoding": "zlib-compressed uint32 start-length pairs in row-major order",
            "data_path": "data/episode-{episode_index:06d}/camera-{camera}.parquet",
            "track_fields": [
                "episode_index",
                "frame_index",
                "camera",
                "object_id",
                "object_label",
                "mask_runs",
            ],
        }
        self._write_json_atomic(self.output_root / "meta" / "info.json", merged_info)
        self._write_json_atomic(self.sam_root / "meta" / "info.json", sam_info)

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)

    def _load_model(self) -> None:
        if self.model is not None:
            return
        logger.info("Loading SAM2 video model from %s on %s", self.model_path, self.device)
        self.processor = Sam2VideoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = Sam2VideoModel.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def _workspace_path(self) -> Path:
        assert self.episode_index is not None and self.camera is not None
        return self.sam_root / "work" / f"episode-{self.episode_index:06d}-camera-{self.camera}.json"

    def _legacy_workspace_path(self) -> Path:
        assert self.episode_index is not None and self.camera is not None
        return self.output_root / "work" / f"episode-{self.episode_index:06d}-camera-{self.camera}.json"

    def _data_path(self) -> Path:
        assert self.episode_index is not None and self.camera is not None
        return (
            self.output_root
            / "data"
            / f"episode-{self.episode_index:06d}"
            / f"camera-{self.camera}.parquet"
        )

    def _sam_data_path(self) -> Path:
        assert self.episode_index is not None and self.camera is not None
        return self.sam_root / "data" / f"episode-{self.episode_index:06d}" / f"camera-{self.camera}.parquet"

    def _save_workspace(self) -> None:
        path = self._workspace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "episode_index": self.episode_index,
            "camera": self.camera,
            "prompts": {
                str(object_id): {
                    str(frame_index): [list(point) for point in points]
                    for frame_index, points in sorted(frame_prompts.items())
                }
                for object_id, frame_prompts in sorted(self.prompts.items())
            },
            "boxes": {
                str(object_id): {
                    str(frame_index): list(box)
                    for frame_index, box in sorted(frame_boxes.items())
                }
                for object_id, frame_boxes in sorted(self.boxes.items())
            },
            "box_sources": {
                str(object_id): {
                    str(frame_index): source
                    for frame_index, source in sorted(frame_sources.items())
                }
                for object_id, frame_sources in sorted(self.box_sources.items())
            },
            "bounds": {
                str(object_id): {"start": bounds.start, "end": bounds.end}
                for object_id, bounds in sorted(self.bounds.items())
            },
            "object_labels": {str(object_id): label for object_id, label in sorted(self.object_labels.items())},
            "grounding_unresolved": {
                str(object_id): reason for object_id, reason in sorted(self.grounding_unresolved.items())
            },
            "grounding_not_visible": {
                str(object_id): reason for object_id, reason in sorted(self.grounding_not_visible.items())
            },
            "excluded_frames": sorted(self.excluded_frames),
        }
        self._write_json_atomic(path, payload)

    def _load_workspace(self) -> bool:
        self.prompts = {}
        self.boxes = {}
        self.box_sources = {}
        self.object_labels = {}
        self.bounds = {}
        self.grounding_unresolved = {}
        self.grounding_not_visible = {}
        self.excluded_frames = set()
        path = self._workspace_path()
        if not path.is_file() and self._legacy_workspace_path().is_file():
            path = self._legacy_workspace_path()
            logger.info("Loading legacy annotation workspace from %s", path)
        if not path.is_file():
            return False
        payload = json.loads(path.read_text())
        self.prompts = {
            int(object_id): {
                int(frame_index): [tuple(int(value) for value in point) for point in points]
                for frame_index, points in frame_prompts.items()
            }
            for object_id, frame_prompts in payload.get("prompts", {}).items()
        }
        self.boxes = {
            int(object_id): {
                int(frame_index): tuple(float(value) for value in box)
                for frame_index, box in frame_boxes.items()
            }
            for object_id, frame_boxes in payload.get("boxes", {}).items()
        }
        self.box_sources = {
            int(object_id): {
                int(frame_index): str(source)
                for frame_index, source in frame_sources.items()
            }
            for object_id, frame_sources in payload.get("box_sources", {}).items()
        }
        self.bounds = {
            int(object_id): ObjectBounds(start=value.get("start"), end=value.get("end"))
            for object_id, value in payload.get("bounds", {}).items()
        }
        self.object_labels = {
            int(object_id): str(label) for object_id, label in payload.get("object_labels", {}).items()
        }
        for object_id in set(self.prompts) | set(self.boxes):
            self.object_labels.setdefault(object_id, f"object_{object_id}")
        self.grounding_unresolved = {
            int(object_id): str(reason)
            for object_id, reason in payload.get("grounding_unresolved", {}).items()
        }
        self.grounding_not_visible = {
            int(object_id): str(reason)
            for object_id, reason in payload.get("grounding_not_visible", {}).items()
        }
        self.excluded_frames = {int(index) for index in payload.get("excluded_frames", [])}
        return True

    def _import_grounding_boxes(self) -> None:
        if self.grounding_review is None:
            return
        assert self.episode_index is not None and self.camera is not None
        resolution = self.grounding_review.resolve(self.episode_index, self.camera)
        for prompt in resolution.prompts:
            self.boxes.setdefault(prompt.object_id, {})[prompt.frame_index] = prompt.box_xyxy
            self.box_sources.setdefault(prompt.object_id, {})[prompt.frame_index] = prompt.source
            self.object_labels[prompt.object_id] = prompt.object_name
            default_start = 0 if self.camera == "top" else prompt.frame_index
            self.bounds[prompt.object_id] = ObjectBounds(start=default_start)
        self.grounding_unresolved = dict(resolution.unresolved)
        self.grounding_not_visible = dict(resolution.intentionally_not_visible)
        for object_id, reason in {
            **self.grounding_unresolved,
            **self.grounding_not_visible,
        }.items():
            self.object_labels.setdefault(object_id, reason.partition(":")[0])
        self._save_workspace()
        logger.info(
            "Imported %d DINO boxes for episode %d camera %s; %d unresolved",
            len(resolution.prompts),
            self.episode_index,
            self.camera,
            len(resolution.unresolved),
        )

    def _load_saved_masks(self) -> None:
        self.propagated_object_masks = None
        self.propagated_masks = None
        sam_path = self._sam_data_path()
        if sam_path.is_file():
            sam_table = pd.read_parquet(sam_path).sort_values(["object_id", "frame_index"])
            tracks: dict[int, list[np.ndarray]] = {}
            for object_id, rows in sam_table.groupby("object_id", sort=True):
                if len(rows) != len(self.video_frames):
                    logger.warning(
                        "Ignoring object %s in %s because its frame count does not match",
                        object_id,
                        sam_path,
                    )
                    continue
                rows = rows.sort_values("frame_index")
                tracks[int(object_id)] = [
                    decode_mask_runs(row.mask_runs, int(row.height), int(row.width))
                    for row in rows.itertuples(index=False)
                ]
                first_row = rows.iloc[0]
                self.object_labels[int(object_id)] = str(first_row["object_label"])
            if tracks:
                self.propagated_object_masks = tracks

        path = self._data_path()
        if not path.is_file() and self.propagated_object_masks is None:
            return
        if path.is_file():
            table = pd.read_parquet(path).sort_values("frame_index")
            if len(table) != len(self.video_frames):
                logger.warning("Ignoring %s because its frame count no longer matches the episode", path)
            else:
                self.propagated_masks = [
                    decode_mask_runs(row.mask_runs, int(row.height), int(row.width))
                    for row in table.itertuples(index=False)
                ]
                self.excluded_frames = {
                    int(row.frame_index)
                    for row in table.itertuples(index=False)
                    if float(row.loss_weight) == 0.0
                }

        if self.propagated_masks is None and self.propagated_object_masks:
            height, width = self.video_frames[0].shape[:2]
            self.propagated_masks = []
            for frame_index in range(len(self.video_frames)):
                union = np.zeros((height, width), dtype=bool)
                for masks in self.propagated_object_masks.values():
                    union |= masks[frame_index]
                self.propagated_masks.append(union)

    def load_episode(self, episode_index: int, camera: str) -> tuple[np.ndarray, str, int, str]:
        with self.lock:
            episode_index = int(episode_index)
            if episode_index not in self.source.episodes.index:
                raise gr.Error(f"Episode {episode_index} does not exist")
            camera = str(camera)
            self._release_session()
            logger.info("Decoding episode %d camera %s", episode_index, camera)
            self.video_frames = self.source.frames(episode_index, camera)
            self.episode_index = episode_index
            self.camera = camera
            self.task_prompt = self.source.prompt(episode_index)
            self.preview_mask = None
            workspace_loaded = self._load_workspace()
            if not workspace_loaded:
                self._import_grounding_boxes()
            self._load_saved_masks()
            status = self.status(0)
            return self.render(0), self.task_prompt, len(self.video_frames) - 1, status

    def _release_session(self) -> None:
        self.session = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _require_loaded_video(self) -> None:
        if self.episode_index is None or self.camera is None or not self.video_frames:
            raise gr.Error("Load an episode and camera first")

    def _new_session(self) -> Any:
        self._require_loaded_video()
        self._load_model()
        assert self.processor is not None
        logger.info("Preparing SAM2 session for %d frames", len(self.video_frames))
        return self.processor.init_video_session(
            video=self.video_frames,
            inference_device=self.device,
            inference_state_device="cpu",
            processing_device="cpu",
            video_storage_device="cpu",
            max_vision_features_cache_size=1,
            dtype=self.dtype,
        )

    def _set_session_prompt(self, object_id: int, frame_index: int) -> None:
        assert self.processor is not None and self.session is not None
        points = self.prompts.get(object_id, {}).get(frame_index, [])
        box = self.boxes.get(object_id, {}).get(frame_index)
        if not points and box is None:
            raise RuntimeError(f"Object {object_id} has no prompt on frame {frame_index}")
        coordinates = [[[[x, y] for x, y, _ in points]]] if points else None
        labels = [[[label for _, _, label in points]]] if points else None
        boxes = [[list(box)]] if box is not None else None
        self.processor.add_inputs_to_inference_session(
            inference_session=self.session,
            frame_idx=frame_index,
            obj_ids=object_id,
            input_points=coordinates,
            input_labels=labels,
            input_boxes=boxes,
            clear_old_inputs=True,
        )

    def _conditioning_objects(self, frame_index: int) -> list[int]:
        """Return object IDs having point or box prompts on one frame."""
        object_ids = set()
        object_ids.update(
            object_id for object_id, frame_prompts in self.prompts.items() if frame_index in frame_prompts
        )
        object_ids.update(
            object_id for object_id, frame_boxes in self.boxes.items() if frame_index in frame_boxes
        )
        return sorted(object_ids)

    def _conditioning_frames(self) -> list[int]:
        """Return sorted frames containing any point or box prompt."""
        frames = {
            frame_index for frame_prompts in self.prompts.values() for frame_index in frame_prompts
        }
        frames.update(frame_index for frame_boxes in self.boxes.values() for frame_index in frame_boxes)
        return sorted(frames)

    def _ensure_session(self) -> None:
        if self.session is not None:
            return
        if not self.prompts and not self.boxes:
            raise gr.Error("Add at least one positive point or box before running SAM2")
        self.session = self._new_session()
        assert self.model is not None
        for frame_index in self._conditioning_frames():
            for object_id in self._conditioning_objects(frame_index):
                self._set_session_prompt(object_id, frame_index)
                self.model(inference_session=self.session, frame_idx=frame_index)

    def _object_range(self, object_id: int) -> tuple[int, int]:
        prompt_frames = sorted(set(self.prompts.get(object_id, {})) | set(self.boxes.get(object_id, {})))
        if not prompt_frames:
            return 0, -1
        bounds = self.bounds.get(object_id, ObjectBounds())
        start = prompt_frames[0] if bounds.start is None else int(bounds.start)
        end = len(self.video_frames) - 1 if bounds.end is None else int(bounds.end)
        return start, end

    def _output_object_masks(self, output: Any, frame_index: int) -> dict[int, np.ndarray]:
        assert self.processor is not None and self.session is not None
        processed = self.processor.post_process_masks(
            [output.pred_masks],
            original_sizes=[[self.session.video_height, self.session.video_width]],
            binarize=True,
        )[0]
        masks = processed.detach().to("cpu").numpy().astype(bool)
        object_masks: dict[int, np.ndarray] = {}
        for object_position, object_id in enumerate(self.session.obj_ids):
            object_id = int(object_id)
            start, end = self._object_range(object_id)
            if start <= frame_index <= end:
                object_masks[object_id] = np.any(masks[object_position], axis=0)
            else:
                object_masks[object_id] = np.zeros(
                    (self.session.video_height, self.session.video_width), dtype=bool
                )
        return object_masks

    @staticmethod
    def _merge_object_masks(
        object_masks: dict[int, np.ndarray],
        height: int,
        width: int,
    ) -> np.ndarray:
        union = np.zeros((height, width), dtype=bool)
        for mask in object_masks.values():
            union |= mask
        return union

    def add_point(
        self,
        frame_index: int,
        object_id: int,
        object_label: str,
        point_mode: str,
        event: gr.SelectData,
    ) -> tuple[np.ndarray, str]:
        with self.lock:
            self._require_loaded_video()
            frame_index = int(frame_index)
            object_id = int(object_id)
            if object_id < 1:
                raise gr.Error("Object ID must be at least 1")
            x, y = (int(value) for value in event.index[:2])
            label = 1 if point_mode == "Positive" else 0
            self.prompts.setdefault(object_id, {}).setdefault(frame_index, []).append((x, y, label))
            self.object_labels[object_id] = str(object_label).strip() or self.object_labels.get(
                object_id, f"object_{object_id}"
            )
            self.bounds.setdefault(object_id, ObjectBounds())
            if label == 1:
                self.grounding_unresolved.pop(object_id, None)
            self.propagated_object_masks = None
            self.propagated_masks = None
            self._save_workspace()

            try:
                if self.session is None:
                    self._ensure_session()
                else:
                    self._set_session_prompt(object_id, frame_index)
                assert self.model is not None
                output = self.model(inference_session=self.session, frame_idx=frame_index)
            except Exception:
                logger.exception("SAM2 session update failed; rebuilding the session")
                self._release_session()
                self._ensure_session()
                assert self.model is not None
                output = self.model(inference_session=self.session, frame_idx=frame_index)
            object_masks = self._output_object_masks(output, frame_index)
            height, width = self.video_frames[frame_index].shape[:2]
            self.preview_mask = self._merge_object_masks(object_masks, height, width)
            return self.render(frame_index), self.status(frame_index)

    def undo_point(self, frame_index: int, object_id: int) -> tuple[np.ndarray, str]:
        with self.lock:
            self._require_loaded_video()
            frame_index = int(frame_index)
            object_id = int(object_id)
            points = self.prompts.get(object_id, {}).get(frame_index, [])
            if not points:
                return self.render(frame_index), self.status(frame_index)
            points.pop()
            if not points:
                self.prompts[object_id].pop(frame_index)
            if not self.prompts.get(object_id):
                self.prompts.pop(object_id, None)
            if object_id not in self.prompts and object_id not in self.boxes:
                label = self.object_labels.get(object_id, f"object_{object_id}")
                self.grounding_unresolved[object_id] = (
                    f"{label}: add a positive manual click"
                )
                self.bounds.pop(object_id, None)
            elif object_id not in self.boxes and not any(
                label == 1
                for frame_points in self.prompts.get(object_id, {}).values()
                for _, _, label in frame_points
            ):
                label = self.object_labels.get(object_id, f"object_{object_id}")
                self.grounding_unresolved[object_id] = (
                    f"{label}: add a positive manual click"
                )
            self.preview_mask = None
            self.propagated_object_masks = None
            self.propagated_masks = None
            self._release_session()
            self._save_workspace()
            return self.render(frame_index), self.status(frame_index)

    def clear_prompts(self, frame_index: int) -> tuple[np.ndarray, str]:
        with self.lock:
            self._require_loaded_video()
            labels = dict(self.object_labels)
            self.prompts = {}
            self.boxes = {}
            self.box_sources = {}
            self.object_labels = labels
            self.bounds = {}
            self.grounding_unresolved = {
                object_id: f"{label}: prompts cleared; add a positive manual click"
                for object_id, label in labels.items()
            }
            self.grounding_not_visible = {}
            self.preview_mask = None
            self.propagated_object_masks = None
            self.propagated_masks = None
            self._release_session()
            self._save_workspace()
            return self.render(int(frame_index)), self.status(int(frame_index))

    def clear_object_prompts(self, frame_index: int, object_id: int) -> tuple[np.ndarray, str]:
        """Remove every point and box prompt for one object."""
        with self.lock:
            object_id = int(object_id)
            label = self.object_labels.get(object_id, f"object_{object_id}")
            self.prompts.pop(object_id, None)
            self.boxes.pop(object_id, None)
            self.box_sources.pop(object_id, None)
            self.bounds.pop(object_id, None)
            self.object_labels[object_id] = label
            self.grounding_unresolved[object_id] = (
                f"{label}: prompts cleared; add a positive manual click"
            )
            self.grounding_not_visible.pop(object_id, None)
            self.preview_mask = None
            self.propagated_object_masks = None
            self.propagated_masks = None
            self._release_session()
            self._save_workspace()
            return self.render(int(frame_index)), self.status(int(frame_index))

    def set_object_end(self, frame_index: int, object_id: int) -> tuple[np.ndarray, str]:
        with self.lock:
            self._require_loaded_video()
            object_id = int(object_id)
            if object_id not in self.prompts and object_id not in self.boxes:
                raise gr.Error(f"Object {object_id} has no prompt")
            self.bounds.setdefault(object_id, ObjectBounds()).end = int(frame_index)
            self.propagated_object_masks = None
            self.propagated_masks = None
            self._release_session()
            self._save_workspace()
            return self.render(int(frame_index)), self.status(int(frame_index))

    def clear_object_end(self, frame_index: int, object_id: int) -> tuple[np.ndarray, str]:
        with self.lock:
            object_id = int(object_id)
            if object_id in self.bounds:
                self.bounds[object_id].end = None
            self.propagated_object_masks = None
            self.propagated_masks = None
            self._release_session()
            self._save_workspace()
            return self.render(int(frame_index)), self.status(int(frame_index))

    def set_frame_loss(self, frame_index: int, include: bool) -> tuple[np.ndarray, str]:
        with self.lock:
            frame_index = int(frame_index)
            if include:
                self.excluded_frames.discard(frame_index)
            else:
                self.excluded_frames.add(frame_index)
            self._save_workspace()
            if self.propagated_masks is not None:
                self._save_masks()
            return self.render(frame_index), self.status(frame_index)

    def propagate_and_save(
        self,
        frame_index: int,
        progress: gr.Progress | None = gr.Progress(),
    ) -> tuple[np.ndarray, str]:
        with self.lock:
            self._require_loaded_video()
            if self.grounding_unresolved:
                reasons = "; ".join(self.grounding_unresolved.values())
                raise gr.Error(f"Resolve all manual Grounding DINO items before saving: {reasons}")

            conditioning_frames = self._conditioning_frames()
            active_object_ids = sorted(set(self.prompts) | set(self.boxes))
            tracked_object_ids = sorted(set(active_object_ids) | set(self.grounding_not_visible))
            if not tracked_object_ids:
                raise gr.Error("Add at least one positive point or reviewed box before running SAM2")

            height, width = self.video_frames[0].shape[:2]
            masks = [np.zeros((height, width), dtype=bool) for _ in self.video_frames]
            object_tracks = {
                object_id: [np.zeros((height, width), dtype=bool) for _ in self.video_frames]
                for object_id in tracked_object_ids
            }

            if active_object_ids:
                self._ensure_session()
                assert self.model is not None and self.session is not None
                first_prompt_by_object = {
                    object_id: min(
                        set(self.prompts.get(object_id, {})) | set(self.boxes.get(object_id, {}))
                    )
                    for object_id in active_object_ids
                }
                first_prompt_frame = conditioning_frames[0]
                last_prompt_frame = conditioning_frames[-1]
                forward_steps = len(self.video_frames) - first_prompt_frame
                reverse_steps = last_prompt_frame + 1 if last_prompt_frame > 0 else 0
                total_steps = forward_steps + reverse_steps
                completed_steps = 0

                forward_iterator = self.model.propagate_in_video_iterator(
                    self.session,
                    start_frame_idx=first_prompt_frame,
                    max_frame_num_to_track=len(self.video_frames),
                    reverse=False,
                    show_progress_bar=False,
                )
                for output in forward_iterator:
                    output_frame_index = int(output.frame_idx)
                    output_masks = self._output_object_masks(output, output_frame_index)
                    for object_id, mask in output_masks.items():
                        if output_frame_index >= first_prompt_by_object[object_id]:
                            object_tracks[object_id][output_frame_index] = mask
                    completed_steps += 1
                    if progress is not None:
                        progress(
                            completed_steps / max(total_steps, 1),
                            desc=f"Forward tracking frame {output_frame_index + 1}",
                        )

                if last_prompt_frame > 0:
                    reverse_iterator = self.model.propagate_in_video_iterator(
                        self.session,
                        start_frame_idx=last_prompt_frame,
                        max_frame_num_to_track=last_prompt_frame,
                        reverse=True,
                        show_progress_bar=False,
                    )
                    for output in reverse_iterator:
                        output_frame_index = int(output.frame_idx)
                        output_masks = self._output_object_masks(output, output_frame_index)
                        for object_id, mask in output_masks.items():
                            if output_frame_index < first_prompt_by_object[object_id]:
                                object_tracks[object_id][output_frame_index] = mask
                        completed_steps += 1
                        if progress is not None:
                            progress(
                                completed_steps / max(total_steps, 1),
                                desc=f"Reverse tracking frame {output_frame_index + 1}",
                            )

            for output_frame_index in range(len(self.video_frames)):
                masks[output_frame_index] = self._merge_object_masks(
                    {
                        object_id: object_tracks[object_id][output_frame_index]
                        for object_id in tracked_object_ids
                    },
                    height,
                    width,
                )
            self.propagated_object_masks = object_tracks
            self.propagated_masks = masks
            self.preview_mask = None
            self._save_object_masks()
            self._save_masks()
            self._write_preview_video()
            return self.render(int(frame_index)), self.status(int(frame_index), saved=True)

    def _save_object_masks(self) -> None:
        if not self.propagated_object_masks:
            return
        path = self._sam_data_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for object_id, masks in sorted(self.propagated_object_masks.items()):
            object_label = self.object_labels.get(object_id, f"object_{object_id}")
            for frame_index, mask in enumerate(masks):
                height, width = mask.shape
                rows.append(
                    {
                        "episode_index": np.int64(self.episode_index),
                        "frame_index": np.int64(frame_index),
                        "camera": self.camera,
                        "object_id": np.int64(object_id),
                        "object_label": object_label,
                        "height": np.int32(height),
                        "width": np.int32(width),
                        "mask_runs": encode_mask_runs(mask),
                    }
                )
        table = pd.DataFrame(rows)
        temporary = path.with_suffix(".parquet.tmp")
        table.to_parquet(temporary, index=False)
        temporary.replace(path)
        logger.info(
            "Saved %d per-object masks for %d objects to %s",
            len(table),
            len(self.propagated_object_masks),
            path,
        )

    def _save_masks(self) -> None:
        if self.propagated_masks is None:
            return
        path = self._data_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = self.propagated_masks[0].shape
        table = pd.DataFrame(
            {
                "episode_index": np.full(len(self.propagated_masks), self.episode_index, dtype=np.int64),
                "frame_index": np.arange(len(self.propagated_masks), dtype=np.int64),
                "camera": np.full(len(self.propagated_masks), self.camera, dtype=object),
                "height": np.full(len(self.propagated_masks), height, dtype=np.int32),
                "width": np.full(len(self.propagated_masks), width, dtype=np.int32),
                "mask_runs": [encode_mask_runs(mask) for mask in self.propagated_masks],
                "loss_weight": np.array(
                    [0.0 if index in self.excluded_frames else 1.0 for index in range(len(self.propagated_masks))],
                    dtype=np.float32,
                ),
            }
        )
        temporary = path.with_suffix(".parquet.tmp")
        table.to_parquet(temporary, index=False)
        temporary.replace(path)
        logger.info("Saved %d masks to %s", len(table), path)

    def _write_preview_video(self) -> None:
        if self.propagated_masks is None:
            return
        assert self.episode_index is not None and self.camera is not None
        path = self.sam_root / "previews" / f"episode-{self.episode_index:06d}-camera-{self.camera}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = self.video_frames[0].shape[:2]
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=self.source.fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "23", "preset": "veryfast"}
            for frame, mask in zip(self.video_frames, self.propagated_masks, strict=True):
                video_frame = av.VideoFrame.from_ndarray(overlay_mask(frame, mask), format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def render(self, frame_index: int) -> np.ndarray:
        self._require_loaded_video()
        frame_index = int(np.clip(frame_index, 0, len(self.video_frames) - 1))
        mask = None
        if self.propagated_masks is not None:
            mask = self.propagated_masks[frame_index]
        elif self.preview_mask is not None:
            mask = self.preview_mask
        image = Image.fromarray(overlay_mask(self.video_frames[frame_index], mask))
        draw = ImageDraw.Draw(image)
        for object_id, frame_boxes in sorted(self.boxes.items()):
            box = frame_boxes.get(frame_index)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            color = (255, 190, 35)
            draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
            draw.text(
                (x0 + 4, max(0, y0 - 15)),
                f"{object_id}:{self.object_labels.get(object_id, f'object_{object_id}')}",
                fill=color,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
        for object_id, frame_prompts in sorted(self.prompts.items()):
            for x, y, label in frame_prompts.get(frame_index, []):
                color = (0, 230, 90) if label == 1 else (240, 60, 60)
                radius = 5
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))
                draw.text((x + 7, y - 8), str(object_id), fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
        return np.asarray(image)

    def status(self, frame_index: int, saved: bool = False) -> str:
        if self.episode_index is None:
            return "No episode loaded."
        point_count = sum(len(points) for frames in self.prompts.values() for points in frames.values())
        box_count = sum(len(boxes) for boxes in self.boxes.values())
        object_descriptions = []
        for object_id in sorted(set(self.prompts) | set(self.boxes)):
            start, end = self._object_range(object_id)
            label = self.object_labels.get(object_id, f"object_{object_id}")
            initialization_frames = sorted(
                set(self.prompts.get(object_id, {})) | set(self.boxes.get(object_id, {}))
            )
            object_descriptions.append(
                f"{object_id}:{label} init={initialization_frames} track={start}-{end}"
            )
        included = int(frame_index) not in self.excluded_frames
        saved_text = " Saved." if saved else ""
        unresolved_text = ""
        if self.grounding_unresolved:
            unresolved_text = " | unresolved [" + "; ".join(self.grounding_unresolved.values()) + "]"
        if self.grounding_not_visible:
            unresolved_text += " | not visible [" + "; ".join(self.grounding_not_visible.values()) + "]"
        return (
            f"Episode {self.episode_index} | {self.camera} | frame {int(frame_index)}/{len(self.video_frames) - 1} | "
            f"objects [{', '.join(object_descriptions)}] | boxes {box_count} | points {point_count} | "
            f"loss {'on' if included else 'off'}.{saved_text}{unresolved_text}"
        )


def build_app(annotator: TaskRelevanceAnnotator, default_episode: int, default_camera: str) -> gr.Blocks:
    with gr.Blocks(title="Task-Relevance Mask Annotation") as app:
        with gr.Row():
            episode = gr.Number(
                value=default_episode,
                precision=0,
                minimum=min(annotator.source.episode_indices),
                maximum=max(annotator.source.episode_indices),
                label="Episode",
            )
            camera = gr.Dropdown(choices=list(CAMERAS), value=default_camera, label="Camera")
            load_button = gr.Button("Load", variant="primary")

        prompt = gr.Textbox(label="Task prompt", interactive=False)
        frame_slider = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="Frame")
        image = gr.Image(type="numpy", label="Frame and relevance mask", interactive=False, height=600)

        with gr.Row():
            object_id = gr.Number(value=1, precision=0, minimum=1, step=1, label="Object ID")
            object_label = gr.Textbox(value="", placeholder="tomato", label="Object label")
            point_mode = gr.Radio(choices=["Positive", "Negative"], value="Positive", label="Point")
            undo_button = gr.Button("Undo point")
            clear_object_button = gr.Button("Clear object prompts")
            clear_button = gr.Button("Clear prompts")

        with gr.Row():
            set_end_button = gr.Button("Set object end")
            clear_end_button = gr.Button("Clear object end")
            include_button = gr.Button("Include frame")
            exclude_button = gr.Button("Exclude frame")
            propagate_button = gr.Button("Propagate and save", variant="primary")

        status = gr.Textbox(label="Status", interactive=False)

        def load(episode_value: float, camera_value: str):
            frame, task, maximum, message = annotator.load_episode(int(episode_value), camera_value)
            return frame, task, gr.update(value=0, maximum=maximum), message

        def show_frame(frame_index: float):
            index = int(frame_index)
            return annotator.render(index), annotator.status(index)

        load_button.click(load, inputs=[episode, camera], outputs=[image, prompt, frame_slider, status])
        frame_slider.change(show_frame, inputs=frame_slider, outputs=[image, status], show_progress="hidden")
        object_id.change(lambda _: "", inputs=object_id, outputs=object_label, show_progress="hidden")
        image.select(
            annotator.add_point,
            inputs=[frame_slider, object_id, object_label, point_mode],
            outputs=[image, status],
        )
        undo_button.click(
            annotator.undo_point,
            inputs=[frame_slider, object_id],
            outputs=[image, status],
        )
        clear_object_button.click(
            annotator.clear_object_prompts,
            inputs=[frame_slider, object_id],
            outputs=[image, status],
        )
        clear_button.click(annotator.clear_prompts, inputs=frame_slider, outputs=[image, status])
        set_end_button.click(
            annotator.set_object_end,
            inputs=[frame_slider, object_id],
            outputs=[image, status],
        )
        clear_end_button.click(
            annotator.clear_object_end,
            inputs=[frame_slider, object_id],
            outputs=[image, status],
        )
        include_button.click(
            lambda frame_index: annotator.set_frame_loss(frame_index, True),
            inputs=frame_slider,
            outputs=[image, status],
        )
        exclude_button.click(
            lambda frame_index: annotator.set_frame_loss(frame_index, False),
            inputs=frame_slider,
            outputs=[image, status],
        )
        propagate_button.click(
            annotator.propagate_and_save,
            inputs=frame_slider,
            outputs=[image, status],
            concurrency_limit=1,
        )

    return app


def parse_episode_indices(specification: str, available: list[int]) -> list[int]:
    """Parse ``all`` or comma-separated inclusive ranges such as ``0,3,8-12``."""
    if specification.strip().lower() == "all":
        return list(available)
    selected: set[int] = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending episode range {item!r}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))
    invalid = sorted(selected - set(available))
    if invalid:
        raise ValueError(f"Dataset does not contain episode indices: {invalid}")
    if not selected:
        raise ValueError("No episode indices were selected")
    return sorted(selected)


def _batch_outputs_complete(
    annotator: TaskRelevanceAnnotator,
    episode_index: int,
    camera: str,
) -> bool:
    """Return whether both saved mask tables contain a complete episode."""
    expected_frames = int(annotator.source.episodes.loc[episode_index, "length"])
    merged_path = (
        annotator.output_root
        / "data"
        / f"episode-{episode_index:06d}"
        / f"camera-{camera}.parquet"
    )
    tracks_path = (
        annotator.sam_root
        / "data"
        / f"episode-{episode_index:06d}"
        / f"camera-{camera}.parquet"
    )
    if not merged_path.is_file() or not tracks_path.is_file():
        return False
    try:
        merged = pd.read_parquet(merged_path, columns=["frame_index"])
        tracks = pd.read_parquet(tracks_path, columns=["object_id", "frame_index"])
    except Exception:
        logger.exception("Could not validate existing outputs for episode %d camera %s", episode_index, camera)
        return False
    track_lengths = tracks.groupby("object_id").size()
    return (
        len(merged) == expected_frames
        and not tracks.empty
        and bool((track_lengths == expected_frames).all())
    )


def run_batch(
    annotator: TaskRelevanceAnnotator,
    episode_indices: list[int],
    cameras: list[str],
    overwrite: bool,
) -> None:
    """Run reviewed DINO-to-SAM2 conversion while preserving resumable progress."""
    report_path = annotator.sam_root / "meta" / "sam2_batch_report.json"
    report: dict[str, Any] = {
        "dataset_root": str(annotator.source.root),
        "episode_indices": episode_indices,
        "cameras": cameras,
        "processed": [],
        "skipped_existing": [],
        "skipped_unresolved": [],
        "failed": [],
    }

    def save_report() -> None:
        annotator._write_json_atomic(report_path, report)

    pair_count = len(episode_indices) * len(cameras)
    pair_number = 0
    try:
        for episode_index in episode_indices:
            for camera in cameras:
                pair_number += 1
                pair = {"episode_index": episode_index, "camera": camera}
                if not overwrite and _batch_outputs_complete(annotator, episode_index, camera):
                    logger.info(
                        "[%d/%d] Skipping complete episode %d camera %s",
                        pair_number,
                        pair_count,
                        episode_index,
                        camera,
                    )
                    report["skipped_existing"].append(pair)
                    save_report()
                    continue
                logger.info(
                    "[%d/%d] Processing episode %d camera %s",
                    pair_number,
                    pair_count,
                    episode_index,
                    camera,
                )
                try:
                    annotator.load_episode(episode_index, camera)
                    if annotator.grounding_unresolved:
                        report["skipped_unresolved"].append(
                            {
                                **pair,
                                "objects": {
                                    str(object_id): reason
                                    for object_id, reason in annotator.grounding_unresolved.items()
                                },
                            }
                        )
                        logger.warning(
                            "Skipping episode %d camera %s: %s",
                            episode_index,
                            camera,
                            "; ".join(annotator.grounding_unresolved.values()),
                        )
                    else:
                        annotator.propagate_and_save(0, progress=None)
                        report["processed"].append(pair)
                except Exception as error:
                    logger.exception("SAM2 failed for episode %d camera %s", episode_index, camera)
                    report["failed"].append({**pair, "error": f"{type(error).__name__}: {error}"})
                    annotator._release_session()
                save_report()
    except KeyboardInterrupt:
        report["interrupted"] = True
        save_report()
        raise
    finally:
        annotator._release_session()
    logger.info(
        "Batch complete: %d processed, %d existing, %d unresolved, %d failed. Report: %s",
        len(report["processed"]),
        len(report["skipped_existing"]),
        len(report["skipped_unresolved"]),
        len(report["failed"]),
        report_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ui", "batch"), default="ui")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sam-root", type=Path, default=DEFAULT_SAM_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--grounding-root", type=Path, default=DEFAULT_GROUNDING_ROOT)
    parser.add_argument("--grounding-review", type=Path, default=DEFAULT_GROUNDING_REVIEW)
    parser.add_argument(
        "--auto-grounding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Import reviewed Grounding DINO boxes when no SAM2 workspace exists.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, or cpu")
    parser.add_argument("--episode", type=int, default=0, help="Initial episode in UI mode.")
    parser.add_argument("--camera", choices=CAMERAS, default="top")
    parser.add_argument(
        "--episodes",
        default="all",
        help="Batch episodes: all or comma-separated indices/inclusive ranges, e.g. 0,3,8-12.",
    )
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=list(CAMERAS))
    parser.add_argument("--overwrite", action="store_true", help="Reprocess complete batch outputs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7866)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    annotator = TaskRelevanceAnnotator(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        sam_root=args.sam_root,
        model_path=args.model_path,
        device=args.device,
        grounding_root=args.grounding_root,
        grounding_review=args.grounding_review,
        auto_grounding=args.auto_grounding,
    )
    if args.mode == "batch":
        episode_indices = parse_episode_indices(args.episodes, annotator.source.episode_indices)
        run_batch(annotator, episode_indices, args.cameras, args.overwrite)
        return
    app = build_app(annotator, args.episode, args.camera)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
