#!/usr/bin/env python
"""Evaluate one loaded policy with multiple language prompts.

The policy, robot, and cameras are initialized once and reused across trials.
Saved trials are standard LeRobot dataset episodes, with every configured
camera written under the session dataset's ``videos`` directory. Additional
prompts can be preloaded with ``--prompts-file=<path>`` using one prompt per
line.

This script deliberately lives outside the LeRobot package. It composes the
public rollout context and inference factory without modifying LeRobot source.

Brief User Manual
-----------------
1. Run ``./scripts/evaluate_pi0.sh`` from the project root.
   The launcher supports PI0, SmolVLA, Action Flow V2, and other LeRobot policy
   checkpoints; select the checkpoint with ``POLICY_ID`` in ``scripts/env.sh``.
2. At ``Prompt>``, enter a listed prompt number or type a new prompt. Enter
   ``q`` to end the evaluation session.
3. Prepare the scene and press Enter to start the selected trial.
4. During execution, press Enter to stop the trial early; otherwise it stops
   after ``dataset.episode_time_s``.
5. Enter ``s`` to save the trial, ``d`` to discard it, or ``q`` to discard it
   and end the session.
6. Every saved episode has separate camera MP4s under
   ``videos/observation.images.<camera>``. AFV2 evaluations additionally save
   matching relevance overlays under ``videos/heatmap.images.<camera>``.
"""

import argparse
import contextlib
import json
import logging
import select
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

CURRENT_FILE = Path(__file__).resolve()
WORKSPACE = CURRENT_FILE.parent.parent
LEROBOT_SRC = WORKSPACE / "repos" / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

# Import config implementations so draccus can resolve the YAML choices.
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: E402, F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: E402, F401
from lerobot.configs import parser  # noqa: E402
from lerobot.datasets import VideoEncodingManager  # noqa: E402
from lerobot.rollout import RolloutConfig, build_rollout_context, create_inference_engine  # noqa: E402
from lerobot.rollout.strategies.core import RolloutStrategy, send_next_action  # noqa: E402
from lerobot.utils.action_interpolator import ActionInterpolator  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STR  # noqa: E402
from lerobot.utils.feature_utils import build_dataset_frame  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402
from lerobot.utils.process import ProcessSignalHandler  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data  # noqa: E402

logger = logging.getLogger(__name__)

LEGACY_POLICY_FIELDS = {
    "action_flow_v2": {
        "rebuild_preprocessor",
        "relevance_annotation_root",
        "relevance_camera_map",
        "require_relevance_targets",
    }
}
LEGACY_PROCESSOR_STEPS = {
    "action_flow_v2": {
        "action_flow_v2_relevance_target_processor",
    }
}
_TEMP_POLICY_DIRS: list[tempfile.TemporaryDirectory] = []


def prepare_legacy_policy_checkpoint(argv: list[str]) -> None:
    """Create a temporary checkpoint view when a saved config has retired fields."""

    path_arg_index = None
    policy_path = None
    for index, arg in enumerate(argv):
        if arg.startswith("--policy.path="):
            path_arg_index = index
            policy_path = arg.split("=", maxsplit=1)[1]
            break
        if arg == "--policy.path" and index + 1 < len(argv):
            path_arg_index = index + 1
            policy_path = argv[index + 1]
            break

    if path_arg_index is None or policy_path is None:
        return

    source_dir = Path(policy_path).expanduser()
    config_path = source_dir / "config.json"
    if not config_path.is_file():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    policy_type = config.get("type")
    retired_fields = LEGACY_POLICY_FIELDS.get(policy_type, set()).intersection(config)

    preprocessor_path = source_dir / "policy_preprocessor.json"
    preprocessor = (
        json.loads(preprocessor_path.read_text(encoding="utf-8"))
        if preprocessor_path.is_file()
        else None
    )
    retired_step_names = LEGACY_PROCESSOR_STEPS.get(policy_type, set())
    retired_steps = set()
    if preprocessor is not None:
        retired_steps = {
            step["registry_name"]
            for step in preprocessor["steps"]
            if step.get("registry_name") in retired_step_names
        }

    if not retired_fields and not retired_steps:
        return

    temp_dir = tempfile.TemporaryDirectory(prefix=f"lerobot_{policy_type}_checkpoint_")
    _TEMP_POLICY_DIRS.append(temp_dir)
    runtime_dir = Path(temp_dir.name)
    for entry in source_dir.iterdir():
        if entry.name not in {"config.json", "policy_preprocessor.json"}:
            (runtime_dir / entry.name).symlink_to(entry.resolve(), target_is_directory=entry.is_dir())

    for field_name in retired_fields:
        config.pop(field_name)
    (runtime_dir / "config.json").write_text(
        json.dumps(config, indent=4) + "\n",
        encoding="utf-8",
    )
    if preprocessor is not None:
        preprocessor["steps"] = [
            step
            for step in preprocessor["steps"]
            if step.get("registry_name") not in retired_step_names
        ]
        (runtime_dir / "policy_preprocessor.json").write_text(
            json.dumps(preprocessor, indent=2) + "\n",
            encoding="utf-8",
        )

    if argv[path_arg_index].startswith("--policy.path="):
        argv[path_arg_index] = f"--policy.path={runtime_dir}"
    else:
        argv[path_arg_index] = str(runtime_dir)
    logger.warning(
        "Using a temporary %s checkpoint view without retired config fields/processor steps: %s",
        policy_type,
        ", ".join(sorted(retired_fields | retired_steps)),
    )


def parse_evaluator_args() -> tuple[argparse.Namespace, list[str]]:
    """Extract evaluator-only arguments and leave rollout arguments untouched."""
    arg_parser = argparse.ArgumentParser(
        description="Load a policy once and evaluate multiple prompts.",
        add_help=False,
    )
    arg_parser.add_argument(
        "--eval-prompt",
        action="append",
        default=[],
        help="Prompt to add to the interactive menu. May be repeated.",
    )
    arg_parser.add_argument(
        "--prompts-file",
        type=Path,
        help="Optional text file containing one prompt per line; # starts a comment.",
    )
    arg_parser.add_argument(
        "--save-heatmap-videos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save AFV2 relevance overlays under videos/heatmap.images.<camera>.",
    )
    arg_parser.add_argument(
        "--heatmap-overlay-alpha",
        type=float,
        default=0.45,
        help="Heatmap opacity in saved camera overlays.",
    )
    return arg_parser.parse_known_args()


class HeatmapVideoRecorder:
    """Write temporary camera/heatmap overlays and commit them with a saved episode."""

    def __init__(
        self,
        dataset_root: Path,
        fps: float,
        camera_features: list[str],
        overlay_alpha: float,
        model_image_size: tuple[int, int] | None = None,
    ) -> None:
        if not 0 <= overlay_alpha <= 1:
            raise ValueError(f"`heatmap_overlay_alpha` must be in [0, 1], got {overlay_alpha}.")
        self.dataset_root = Path(dataset_root)
        self.fps = fps
        self.camera_features = camera_features
        self.overlay_alpha = overlay_alpha
        self.model_image_size = model_image_size
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix=".heatmap-trial-",
            dir=self.dataset_root,
        )
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._paths: dict[str, Path] = {}
        self._closed = False

    @staticmethod
    def _camera_name(feature_key: str) -> str:
        return feature_key.rsplit(".", maxsplit=1)[-1]

    @staticmethod
    def _as_rgb_image(frame) -> np.ndarray:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        if frame.ndim == 3 and frame.shape[0] in {1, 3, 4} and frame.shape[-1] not in {1, 3, 4}:
            frame = np.moveaxis(frame, 0, -1)
        if frame.ndim != 3 or frame.shape[-1] not in {3, 4}:
            raise ValueError(f"Expected an HWC RGB image, got shape {frame.shape}.")
        frame = frame[..., :3]
        if np.issubdtype(frame.dtype, np.floating):
            if frame.min(initial=0) < 0:
                frame = (frame + 1) * 127.5
            elif frame.max(initial=0) <= 1:
                frame = frame * 255
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _get_writer(self, camera_name: str, frame_size: tuple[int, int]) -> cv2.VideoWriter:
        writer = self._writers.get(camera_name)
        if writer is not None:
            return writer

        path = Path(self._temp_dir.name) / f"{camera_name}.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            frame_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open heatmap video writer: {path}")
        self._writers[camera_name] = writer
        self._paths[camera_name] = path
        return writer

    def _project_heatmap(self, heatmap: np.ndarray, width: int, height: int) -> np.ndarray:
        """Undo the policy's top/left letterbox padding, then resize to the camera frame."""
        if self.model_image_size is None:
            return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)

        model_width, model_height = self.model_image_size
        ratio = max(width / model_width, height / model_height)
        resized_width = int(width / ratio)
        resized_height = int(height / ratio)
        pad_width = max(0, model_width - resized_width)
        pad_height = max(0, model_height - resized_height)

        model_heatmap = cv2.resize(
            heatmap,
            (model_width, model_height),
            interpolation=cv2.INTER_CUBIC,
        )
        content_heatmap = model_heatmap[
            pad_height : pad_height + resized_height,
            pad_width : pad_width + resized_width,
        ]
        return cv2.resize(content_heatmap, (width, height), interpolation=cv2.INTER_CUBIC)

    def add_frame(self, observation: dict, heatmaps: torch.Tensor | None) -> None:
        if heatmaps is None:
            return
        if heatmaps.ndim == 4:
            heatmaps = heatmaps[0]
        if heatmaps.ndim != 3 or heatmaps.shape[0] != len(self.camera_features):
            raise ValueError(
                "AFV2 heatmaps must have shape (num_cameras, height, width), got "
                f"{tuple(heatmaps.shape)} for {len(self.camera_features)} cameras."
            )

        for camera_index, feature_key in enumerate(self.camera_features):
            camera_name = self._camera_name(feature_key)
            frame = observation.get(camera_name)
            if frame is None:
                frame = observation.get(feature_key)
            if frame is None:
                continue

            rgb = self._as_rgb_image(frame)
            height, width = rgb.shape[:2]
            probability = heatmaps[camera_index].detach().cpu().numpy()
            probability = self._project_heatmap(probability, width, height)
            probability = np.clip(probability, 0, 1)
            colored = cv2.applyColorMap(
                np.round(probability * 255).astype(np.uint8),
                cv2.COLORMAP_JET,
            )
            base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(
                base,
                1 - self.overlay_alpha,
                colored,
                self.overlay_alpha,
                0,
            )
            self._get_writer(camera_name, (width, height)).write(overlay)

    def close(self) -> None:
        if self._closed:
            return
        for writer in self._writers.values():
            writer.release()
        self._closed = True

    def commit(self, episode_index: int, dataset) -> list[Path]:
        """Save one heatmap MP4 per episode using LeRobot-style chunk/file numbering."""
        self.close()
        latest_episode = dataset.meta.latest_episode
        if latest_episode is None:
            raise RuntimeError(f"Episode {episode_index} metadata is unavailable after saving")

        output_paths = []
        for feature_key in self.camera_features:
            camera_name = self._camera_name(feature_key)
            source = self._paths.get(camera_name)
            if source is None:
                continue

            chunk_index = latest_episode[f"videos/{feature_key}/chunk_index"][0]
            file_index = latest_episode[f"videos/{feature_key}/file_index"][0]
            destination = self.dataset_root / Path(
                dataset.meta.video_path.format(
                    video_key=f"heatmap.images.{camera_name}",
                    chunk_index=chunk_index,
                    file_index=file_index,
                )
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"Heatmap episode video already exists: {destination}")
            shutil.move(source, destination)
            output_paths.append(destination)
        self._temp_dir.cleanup()
        return output_paths

    def discard(self) -> None:
        self.close()
        self._temp_dir.cleanup()


def load_prompts(args: argparse.Namespace, default_prompt: str) -> list[str]:
    prompts = list(args.eval_prompt)
    if args.prompts_file is not None:
        for line in args.prompts_file.read_text(encoding="utf-8").splitlines():
            prompt = line.strip()
            if prompt and not prompt.startswith("#"):
                prompts.append(prompt)
    if default_prompt:
        prompts.insert(0, default_prompt)
    return list(dict.fromkeys(prompts))


def choose_prompt(prompts: list[str]) -> str | None:
    print("\nAvailable prompts:")
    for index, prompt in enumerate(prompts, start=1):
        print(f"  {index}. {prompt}")
    print("Enter a number, type a new prompt, or enter q to finish.")

    while True:
        choice = input("Prompt> ").strip()
        if choice.lower() == "q":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(prompts):
            return prompts[int(choice) - 1]
        if choice:
            if choice not in prompts:
                prompts.append(choice)
            return choice
        print("Choose a prompt number or type a prompt.")


def stdin_line_ready() -> bool:
    """Return True after the operator presses Enter during a trial."""
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return False
    sys.stdin.readline()
    return True


def make_prompt_engine(ctx, prompt: str):
    """Create a fresh inference state for a prompt while reusing the loaded policy."""
    cfg = ctx.runtime.cfg
    return create_inference_engine(
        cfg.inference,
        policy=ctx.policy.policy,
        preprocessor=ctx.policy.preprocessor,
        postprocessor=ctx.policy.postprocessor,
        robot_wrapper=ctx.hardware.robot_wrapper,
        hw_features=ctx.data.hw_features,
        dataset_features=ctx.data.dataset_features,
        ordered_action_keys=ctx.data.ordered_action_keys,
        task=prompt,
        fps=cfg.fps,
        device=cfg.device,
        use_torch_compile=cfg.use_torch_compile,
        compile_warmup_inferences=cfg.compile_warmup_inferences,
        shutdown_event=ctx.runtime.shutdown_event,
    )


def make_heatmap_recorder(ctx) -> HeatmapVideoRecorder | None:
    """Create an AFV2 heatmap recorder when requested and supported."""
    if not EVALUATOR_ARGS.save_heatmap_videos:
        return None

    policy = ctx.policy.policy
    if not callable(getattr(policy, "get_latest_relevance_heatmaps", None)):
        logger.warning(
            "Heatmap video output was requested, but policy type %s does not expose relevance heatmaps",
            policy.config.type,
        )
        return None

    return HeatmapVideoRecorder(
        dataset_root=ctx.data.dataset.root,
        fps=ctx.runtime.cfg.fps,
        camera_features=list(policy.config.image_features),
        overlay_alpha=EVALUATOR_ARGS.heatmap_overlay_alpha,
        model_image_size=(
            tuple(policy.config.resize_imgs_with_padding)
            if policy.config.resize_imgs_with_padding is not None
            else None
        ),
    )


def run_trial(ctx, prompt: str) -> tuple[int, HeatmapVideoRecorder | None]:
    """Run and buffer one policy-driven trial, returning its frame count and heatmap videos."""
    cfg = ctx.runtime.cfg
    dataset = ctx.data.dataset
    robot = ctx.hardware.robot_wrapper
    features = ctx.data.dataset_features
    duration_s = cfg.dataset.episode_time_s
    heatmap_recorder = make_heatmap_recorder(ctx)
    heatmap_getter = getattr(ctx.policy.policy, "get_latest_relevance_heatmaps", None)

    engine = make_prompt_engine(ctx, prompt)
    ctx.policy.inference = engine
    interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
    control_interval = interpolator.get_control_interval(cfg.fps)
    cached_obs_processed = None
    action_start = None
    frames_added = 0

    engine.reset()
    engine.start()
    engine.resume()
    logger.info("Trial started: %s", prompt)
    print(f"Running for up to {duration_s:g}s. Press Enter to stop this trial early.")

    try:
        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()
            if stdin_line_ready():
                logger.info("Trial stopped early by operator")
                break
            if action_start is not None and loop_start - action_start >= duration_s:
                logger.info("Trial duration reached (%.1fs)", duration_s)
                break

            obs_raw = robot.get_observation()
            if cached_obs_processed is None or interpolator.needs_new_action():
                cached_obs_processed = ctx.processors.robot_observation_processor(obs_raw)
                engine.notify_observation(cached_obs_processed)

            if not engine.ready:
                precise_sleep(control_interval)
                continue

            action_dict = send_next_action(cached_obs_processed, obs_raw, ctx, interpolator)
            if action_dict is not None:
                if action_start is None:
                    action_start = time.perf_counter()
                    logger.info("First action ready; trial timer started")
                obs_frame = build_dataset_frame(features, cached_obs_processed, prefix=OBS_STR)
                action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                dataset.add_frame({**obs_frame, **action_frame, "task": prompt})
                frames_added += 1

                if heatmap_recorder is not None:
                    try:
                        heatmap_recorder.add_frame(cached_obs_processed, heatmap_getter())
                    except Exception as exc:
                        logger.exception(
                            "Disabling AFV2 heatmap recording after a video error: %s",
                            exc,
                        )
                        heatmap_recorder.discard()
                        heatmap_recorder = None

                if cfg.display_data:
                    log_rerun_data(
                        observation=cached_obs_processed,
                        action=action_dict,
                        compress_images=cfg.display_compressed_images,
                    )

            elapsed = time.perf_counter() - loop_start
            if elapsed < control_interval:
                precise_sleep(control_interval - elapsed)
            else:
                logger.warning(
                    "Control loop is slower (%.1f Hz) than target FPS %.1f",
                    1 / elapsed,
                    cfg.fps,
                )
    finally:
        engine.pause()
        engine.stop()
        if heatmap_recorder is not None:
            heatmap_recorder.close()

    return frames_added, heatmap_recorder


def ask_save_decision() -> str:
    while True:
        decision = input("Save trial? [s] save, [d] discard, [q] discard and quit: ").strip().lower()
        if decision in {"s", "d", "q"}:
            return decision


def disconnect(ctx) -> None:
    with contextlib.suppress(Exception):
        ctx.policy.inference.stop()
    robot = ctx.hardware.robot_wrapper.inner
    if robot.is_connected:
        if ctx.runtime.cfg.return_to_initial_position and ctx.hardware.initial_position:
            logger.info("Returning robot to its initial position before disconnect")
            RolloutStrategy._return_to_initial_position(ctx.hardware)
        robot.disconnect()
    teleop = ctx.hardware.teleop
    if teleop is not None and teleop.is_connected:
        teleop.disconnect()


@parser.wrap()
def evaluate(cfg: RolloutConfig) -> None:
    """Build one rollout context and evaluate prompts until the operator quits."""
    init_logging()
    if cfg.dataset is None:
        raise ValueError("A --dataset configuration is required to save camera videos.")
    if cfg.strategy.type != "episodic":
        raise ValueError("Use --strategy.type=episodic with this evaluator.")
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive evaluation requires a terminal stdin.")

    if cfg.display_data:
        init_rerun(session_name="multi_prompt_policy_evaluation", ip=cfg.display_ip, port=cfg.display_port)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    prompts = load_prompts(EVALUATOR_ARGS, cfg.dataset.single_task or cfg.task)
    ctx = None

    try:
        logger.info("Building rollout context; the policy will be loaded only once")
        ctx = build_rollout_context(cfg, signal_handler.shutdown_event)
        dataset = ctx.data.dataset
        logger.info("Evaluation output: %s", dataset.root)

        with VideoEncodingManager(dataset):
            while not ctx.runtime.shutdown_event.is_set():
                prompt = choose_prompt(prompts)
                if prompt is None:
                    break

                answer = input(f"Prepare the scene for: {prompt}\nPress Enter to start, or b to go back: ")
                if answer.strip().lower() == "b":
                    continue

                frames_added, heatmap_recorder = run_trial(ctx, prompt)
                if ctx.runtime.shutdown_event.is_set():
                    dataset.clear_episode_buffer()
                    if heatmap_recorder is not None:
                        heatmap_recorder.discard()
                    break
                if frames_added == 0:
                    logger.warning("No actions were produced; discarding the empty trial")
                    dataset.clear_episode_buffer()
                    if heatmap_recorder is not None:
                        heatmap_recorder.discard()
                    continue

                decision = ask_save_decision()
                if decision == "s":
                    episode_index = dataset.num_episodes
                    dataset.save_episode()
                    logger.info(
                        "Saved episode %d (%d frames): %s",
                        episode_index,
                        frames_added,
                        prompt,
                    )
                    if heatmap_recorder is not None:
                        try:
                            output_paths = heatmap_recorder.commit(episode_index, dataset)
                            for output_path in output_paths:
                                logger.info("Saved AFV2 heatmap video: %s", output_path)
                        except Exception as exc:
                            logger.exception(
                                "Episode was saved, but its AFV2 heatmap videos could not be committed: %s",
                                exc,
                            )
                else:
                    dataset.clear_episode_buffer()
                    if heatmap_recorder is not None:
                        heatmap_recorder.discard()
                    logger.info("Discarded trial: %s", prompt)
                    if decision == "q":
                        break

                if cfg.strategy.reset_to_initial_position:
                    logger.info("Returning robot to its initial position before the next trial")
                    RolloutStrategy._return_to_initial_position(ctx.hardware)

        logger.info("Camera videos saved under: %s", dataset.root / "videos")
    finally:
        if ctx is not None:
            disconnect(ctx)


def main() -> None:
    register_third_party_plugins()
    evaluate()


prepare_legacy_policy_checkpoint(sys.argv)
EVALUATOR_ARGS, ROLLOUT_ARGS = parse_evaluator_args()
sys.argv = [sys.argv[0], *ROLLOUT_ARGS]


if __name__ == "__main__":
    main()
