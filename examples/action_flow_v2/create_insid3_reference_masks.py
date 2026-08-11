#!/usr/bin/env python3
"""Author precise SAM2 reference masks for INSID3 from LeRobot videos.

Brief user manual:

1. Select an episode and camera, then press ``Load episode``.
2. Move the frame slider to a frame where the target is fully visible.
3. Select or enter the entity name.
4. Add positive clicks inside the target and negative clicks on nearby objects
   or background. SAM2 is rerun after every click.
5. Inspect all three SAM2 candidates when the highest-scoring mask is not the
   cleanest one.
6. Press ``Save reference`` only when the mask contains one intended object.

Each save writes the original frame, binary mask, preview, and provenance into
an INSID3-compatible ``references.json`` manifest. Multiple references may be
saved for each entity and camera. Existing records are preserved.

Example:

    HF_HUB_OFFLINE=1 python examples/action_flow_v2/create_insid3_reference_masks.py \
        --dataset-root runs/dataset/rm65_single_table_preparation_and_bussing \
        --output-root artifacts/insid3/references/rm65_tabletop_sam2 \
        --model-name facebook/sam2.1-hiera-small \
        --device cuda \
        --port 7868
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

from sam2_image import DEFAULT_SAM2_MODEL, Sam2PointSegmenter


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = WORKSPACE / "runs/dataset/rm65_single_table_preparation_and_bussing"
DEFAULT_OUTPUT_ROOT = WORKSPACE / "artifacts/insid3/references/rm65_tabletop_sam2"
DEFAULT_MODEL_NAME = DEFAULT_SAM2_MODEL
DEFAULT_ENTITIES = (
    "tomato",
    "corn",
    "cucumber",
    "garlic",
    "blue plate",
    "pink plate",
    "yellow plate",
    "collection box",
)
OVERLAY_COLOR = np.array([20, 210, 120], dtype=np.float32)
logger = logging.getLogger(__name__)


class EpisodeVideoSource:
    """Load camera frames and task metadata from a LeRobot v3 dataset."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        info_path = self.root / "meta/info.json"
        if not info_path.is_file():
            raise FileNotFoundError(info_path)
        self.info = json.loads(info_path.read_text())
        episode_files = sorted((self.root / "meta/episodes").glob("chunk-*/*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"No episode metadata under {self.root}")
        episodes = pd.concat((pd.read_parquet(path) for path in episode_files), ignore_index=True)
        self.episodes = episodes.set_index("episode_index", drop=False).sort_index()
        self.fps = float(self.info["fps"])
        self.camera_keys = {
            key.rsplit(".", maxsplit=1)[-1]: key
            for key, feature in self.info["features"].items()
            if feature.get("dtype") == "video"
        }
        if not self.camera_keys:
            raise ValueError(f"Dataset {self.root} has no video features")

    @property
    def episode_indices(self) -> list[int]:
        return [int(value) for value in self.episodes.index]

    @property
    def cameras(self) -> list[str]:
        return sorted(self.camera_keys)

    def prompt(self, episode_index: int) -> str:
        tasks = self.episodes.loc[episode_index, "tasks"]
        if isinstance(tasks, str):
            return tasks
        if len(tasks) != 1:
            return "; ".join(str(task) for task in tasks)
        return str(tasks[0])

    def frames(self, episode_index: int, camera: str) -> list[np.ndarray]:
        if episode_index not in self.episodes.index:
            raise ValueError(f"Unknown episode {episode_index}")
        if camera not in self.camera_keys:
            raise ValueError(f"Unknown camera {camera!r}; choices: {self.cameras}")

        row = self.episodes.loc[episode_index]
        video_key = self.camera_keys[camera]
        prefix = f"videos/{video_key}"
        relative_path = self.info["video_path"].format(
            video_key=video_key,
            chunk_index=int(row[f"{prefix}/chunk_index"]),
            file_index=int(row[f"{prefix}/file_index"]),
        )
        video_path = self.root / relative_path
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        start_s = float(row[f"{prefix}/from_timestamp"])
        expected_frames = int(row["length"])
        tolerance_s = 0.5 / self.fps
        decoded: list[np.ndarray] = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if start_s > 0:
                seek_pts = max(0, int((start_s - 1.0 / self.fps) / float(stream.time_base)))
                container.seek(seek_pts, stream=stream, backward=True)
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
                f"Decoded {len(decoded)} frames from {video_path}; expected {expected_frames}"
            )
        return decoded


def mask_to_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def component_count(mask: np.ndarray) -> int:
    _, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return int(count)


def entity_slug(entity: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", entity.strip().lower()).strip("_")
    if not slug:
        raise ValueError("Entity name must contain at least one letter or number")
    return slug


class ReferenceMaskAuthor:
    """Own the current frame, SAM2 point prompts, and saved reference bank."""

    def __init__(
        self,
        dataset_root: Path,
        output_root: Path,
        model_name: str,
        device: str,
    ):
        self.source = EpisodeVideoSource(dataset_root)
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_root / "references.json"
        self.model_name = model_name
        self.segmenter = Sam2PointSegmenter(model_name, device)
        self.frames: list[np.ndarray] = []
        self.episode_index: int | None = None
        self.camera: str | None = None
        self.task_prompt = ""
        self.frame_index = 0
        self.points: list[tuple[int, int, int]] = []
        self.candidate_masks: list[np.ndarray] = []
        self.candidate_scores: list[float] = []
        self.selected_candidate = 0
        self.lock = threading.RLock()

    def _require_frame(self) -> np.ndarray:
        if not self.frames:
            raise RuntimeError("Load an episode before selecting a target")
        return self.frames[self.frame_index]

    def _clear_annotation(self) -> None:
        self.points = []
        self.candidate_masks = []
        self.candidate_scores = []
        self.selected_candidate = 0

    def _clear_embedding(self) -> None:
        self.segmenter.clear_image()

    def load_episode(self, episode_index: float, camera: str) -> tuple[Any, ...]:
        with self.lock:
            index = int(episode_index)
            logger.info("Loading episode %d camera %s", index, camera)
            self.frames = self.source.frames(index, camera)
            self.episode_index = index
            self.camera = camera
            self.task_prompt = self.source.prompt(index)
            self.frame_index = 0
            self._clear_annotation()
            self._clear_embedding()
            return (
                self.render(),
                self.task_prompt,
                self.frame_index,
                len(self.frames) - 1,
                self.status(),
                self.saved_rows(),
            )

    def select_frame(self, frame_index: float) -> tuple[np.ndarray, str]:
        with self.lock:
            self._require_frame()
            index = int(np.clip(int(frame_index), 0, len(self.frames) - 1))
            if index != self.frame_index:
                self.frame_index = index
                self._clear_annotation()
                self._clear_embedding()
            return self.render(), self.status()

    def _run_sam2(self) -> None:
        if not any(label == 1 for _, _, label in self.points):
            self.candidate_masks = []
            self.candidate_scores = []
            return
        assert self.episode_index is not None and self.camera is not None
        self.candidate_masks, self.candidate_scores = self.segmenter.segment(
            self._require_frame(),
            self.points,
            image_key=(self.episode_index, self.camera, self.frame_index),
        )
        self.selected_candidate = int(np.argmax(self.candidate_scores))

    def add_point(self, point_mode: str, event: Any) -> tuple[np.ndarray, str, str]:
        with self.lock:
            frame = self._require_frame()
            x, y = (int(round(value)) for value in event.index[:2])
            height, width = frame.shape[:2]
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Click {(x, y)} is outside image size {(width, height)}")
            label = 1 if point_mode == "Positive" else 0
            self.points.append((x, y, label))
            self._run_sam2()
            return self.render(), self.status(), str(self.selected_candidate + 1)

    def undo_point(self) -> tuple[np.ndarray, str, str]:
        with self.lock:
            if self.points:
                self.points.pop()
            self._run_sam2()
            return self.render(), self.status(), str(self.selected_candidate + 1)

    def clear_points(self) -> tuple[np.ndarray, str, str]:
        with self.lock:
            self._clear_annotation()
            return self.render(), self.status(), "1"

    def choose_candidate(self, candidate: str) -> tuple[np.ndarray, str]:
        with self.lock:
            index = int(candidate) - 1
            if self.candidate_masks:
                self.selected_candidate = int(np.clip(index, 0, len(self.candidate_masks) - 1))
            return self.render(), self.status()

    def selected_mask(self) -> np.ndarray | None:
        if not self.candidate_masks:
            return None
        return self.candidate_masks[self.selected_candidate]

    def render(self) -> np.ndarray:
        frame = self._require_frame().copy()
        mask = self.selected_mask()
        if mask is not None and np.any(mask):
            pixels = frame[mask].astype(np.float32)
            frame[mask] = np.clip(0.55 * pixels + 0.45 * OVERLAY_COLOR, 0, 255).astype(np.uint8)
            box = mask_to_xyxy(mask)
            if box is not None:
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline="#ff304f", width=2)
                frame = np.asarray(image).copy()
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        for x, y, label in self.points:
            color = "#00ff80" if label == 1 else "#ff304f"
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=1)
        return np.asarray(image)

    def status(self) -> str:
        if not self.frames:
            return "Load an episode to begin."
        positives = sum(label == 1 for _, _, label in self.points)
        negatives = len(self.points) - positives
        mask = self.selected_mask()
        if mask is None:
            mask_text = "no mask"
        else:
            area = float(mask.mean())
            score = self.candidate_scores[self.selected_candidate]
            mask_text = (
                f"candidate {self.selected_candidate + 1}, score={score:.3f}, "
                f"area={area:.3%}, components={component_count(mask)}"
            )
        return (
            f"Episode {self.episode_index} | {self.camera} | frame "
            f"{self.frame_index}/{len(self.frames) - 1} | positive={positives}, "
            f"negative={negatives} | {mask_text}"
        )

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.is_file():
            return json.loads(self.manifest_path.read_text())
        return {
            "format": "insid3_rm65_references",
            "format_version": 2,
            "bbox_format": "xyxy_inclusive_exclusive",
            "mask_format": "binary_png_0_255",
            "references": {},
        }

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(self.manifest_path)

    def save_reference(self, entity: str) -> tuple[str, list[list[Any]]]:
        with self.lock:
            frame = self._require_frame()
            mask = self.selected_mask()
            entity = entity.strip().lower()
            if not entity:
                raise ValueError("Select or enter an entity name")
            if mask is None or not np.any(mask):
                raise RuntimeError("Create and inspect a non-empty SAM2 mask before saving")
            if not any(label == 1 for _, _, label in self.points):
                raise RuntimeError("At least one positive click is required")
            assert self.episode_index is not None and self.camera is not None

            manifest = self._load_manifest()
            entry = manifest["references"].setdefault(
                entity,
                {
                    "reference_images": [],
                    "reference_masks": [],
                    "reference_cameras": [],
                    "mask_method": "manually_verified_sam2_points",
                    "records": [],
                },
            )
            sequence = len(entry.get("records", []))
            slug = entity_slug(entity)
            stem = f"reference-{sequence:03d}"
            relative_root = Path(slug) / self.camera
            image_path = self.output_root / "images" / relative_root / f"{stem}.png"
            mask_path = self.output_root / "masks" / relative_root / f"{stem}.png"
            preview_path = self.output_root / "previews" / relative_root / f"{stem}.jpg"
            for path in (image_path, mask_path, preview_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(image_path)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            Image.fromarray(self.render()).save(preview_path, quality=95)

            score = float(self.candidate_scores[self.selected_candidate])
            record = {
                "reference_id": f"{slug}-{self.camera}-{sequence:03d}",
                "episode_index": self.episode_index,
                "camera": self.camera,
                "frame_index": self.frame_index,
                "task_prompt": self.task_prompt,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "preview_path": str(preview_path.resolve()),
                "points": [
                    {"x": x, "y": y, "label": "positive" if label == 1 else "negative"}
                    for x, y, label in self.points
                ],
                "sam_candidate": self.selected_candidate + 1,
                "predicted_iou": score,
                "bbox_xyxy": mask_to_xyxy(mask),
                "foreground_fraction": float(mask.mean()),
                "component_count": component_count(mask),
            }
            entry.setdefault("reference_images", []).append(str(image_path.resolve()))
            entry.setdefault("reference_masks", []).append(str(mask_path.resolve()))
            entry.setdefault("reference_cameras", []).append(self.camera)
            entry.setdefault("records", []).append(record)
            self._write_manifest(manifest)
            message = f"Saved {record['reference_id']} to {self.manifest_path}"
            logger.info(message)
            return message, self.saved_rows(manifest)

    def saved_rows(self, manifest: dict[str, Any] | None = None) -> list[list[Any]]:
        manifest = manifest or self._load_manifest()
        rows: list[list[Any]] = []
        for entity, entry in sorted(manifest.get("references", {}).items()):
            for record in entry.get("records", []):
                rows.append(
                    [
                        entity,
                        record["reference_id"],
                        record["episode_index"],
                        record["camera"],
                        record["frame_index"],
                        f"{record['predicted_iou']:.3f}",
                        f"{record['foreground_fraction']:.3%}",
                    ]
                )
        return rows


def sanitize_proxy_environment() -> None:
    """Remove the unsupported ``socks://`` spelling before importing Gradio."""

    for name in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(name, "")
        if value.lower().startswith("socks://"):
            logger.warning("Ignoring unsupported %s=%s for the local Gradio server", name, value)
            os.environ.pop(name, None)


def build_app(author: ReferenceMaskAuthor, entities: list[str]) -> Any:
    sanitize_proxy_environment()
    import gradio as gr

    first_episode = author.source.episode_indices[0]
    first_camera = author.source.cameras[0]
    with gr.Blocks(title="INSID3 SAM2 Reference Author") as app:
        with gr.Row():
            episode = gr.Number(
                value=first_episode,
                precision=0,
                minimum=min(author.source.episode_indices),
                maximum=max(author.source.episode_indices),
                label="Episode",
            )
            camera = gr.Dropdown(author.source.cameras, value=first_camera, label="Camera")
            load_button = gr.Button("Load episode", variant="primary")

        task_prompt = gr.Textbox(label="Task prompt", interactive=False)
        frame_slider = gr.Slider(0, 1, value=0, step=1, label="Frame")
        image = gr.Image(type="numpy", label="Reference frame and SAM2 mask", height=600)

        with gr.Row():
            entity = gr.Dropdown(
                entities,
                value=entities[0],
                allow_custom_value=True,
                label="Entity",
            )
            point_mode = gr.Radio(["Positive", "Negative"], value="Positive", label="Click type")
            candidate = gr.Radio(["1", "2", "3"], value="1", label="SAM2 candidate")
            undo_button = gr.Button("Undo")
            clear_button = gr.Button("Clear")
            save_button = gr.Button("Save reference", variant="primary")

        status = gr.Textbox(value="Load an episode to begin.", label="Status", interactive=False)
        saved = gr.Dataframe(
            headers=["Entity", "Reference ID", "Episode", "Camera", "Frame", "Score", "Area"],
            value=author.saved_rows(),
            interactive=False,
            label="Saved references",
        )

        def load(episode_value: float, camera_value: str):
            frame, prompt, frame_value, frame_maximum, message, rows = author.load_episode(
                episode_value, camera_value
            )
            return (
                frame,
                prompt,
                gr.update(value=frame_value, maximum=frame_maximum),
                message,
                "1",
                rows,
            )

        def add_point(point_value: str, event: Any):
            return author.add_point(point_value, event)

        add_point.__annotations__["event"] = gr.SelectData

        load_button.click(
            load,
            inputs=[episode, camera],
            outputs=[image, task_prompt, frame_slider, status, candidate, saved],
            concurrency_limit=1,
        )
        frame_slider.change(
            author.select_frame,
            inputs=frame_slider,
            outputs=[image, status],
            show_progress="hidden",
        )
        image.select(
            add_point,
            inputs=point_mode,
            outputs=[image, status, candidate],
            concurrency_limit=1,
        )
        candidate.change(
            author.choose_candidate,
            inputs=candidate,
            outputs=[image, status],
            show_progress="hidden",
        )
        undo_button.click(
            author.undo_point,
            outputs=[image, status, candidate],
            concurrency_limit=1,
        )
        clear_button.click(
            author.clear_points,
            outputs=[image, status, candidate],
        )
        save_button.click(
            author.save_reference,
            inputs=entity,
            outputs=[status, saved],
            concurrency_limit=1,
        )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, or cpu")
    parser.add_argument("--entities", nargs="+", default=list(DEFAULT_ENTITIES))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7868)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    author = ReferenceMaskAuthor(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        model_name=args.model_name,
        device=args.device,
    )
    app = build_app(author, args.entities)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
