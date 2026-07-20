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
2. At ``Prompt>``, enter a listed prompt number or type a new prompt. Enter
   ``q`` to end the evaluation session.
3. Prepare the scene and press Enter to start the selected trial.
4. During execution, press Enter to stop the trial early; otherwise it stops
   after ``dataset.episode_time_s``.
5. Enter ``s`` to save the trial, ``d`` to discard it, or ``q`` to discard it
   and end the session.
"""

import argparse
import contextlib
import logging
import select
import sys
import time
from pathlib import Path

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
from lerobot.rollout.strategies.core import send_next_action  # noqa: E402
from lerobot.utils.action_interpolator import ActionInterpolator  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STR  # noqa: E402
from lerobot.utils.feature_utils import build_dataset_frame  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402
from lerobot.utils.process import ProcessSignalHandler  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data  # noqa: E402

logger = logging.getLogger(__name__)


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
    return arg_parser.parse_known_args()


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


def run_trial(ctx, prompt: str) -> int:
    """Run and buffer one policy-driven trial, returning its frame count."""
    cfg = ctx.runtime.cfg
    dataset = ctx.data.dataset
    robot = ctx.hardware.robot_wrapper
    features = ctx.data.dataset_features
    duration_s = cfg.dataset.episode_time_s

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

    return frames_added


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
        init_rerun(session_name="pi0_multi_prompt_evaluation", ip=cfg.display_ip, port=cfg.display_port)

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

                frames_added = run_trial(ctx, prompt)
                if ctx.runtime.shutdown_event.is_set():
                    dataset.clear_episode_buffer()
                    break
                if frames_added == 0:
                    logger.warning("No actions were produced; discarding the empty trial")
                    dataset.clear_episode_buffer()
                    continue

                decision = ask_save_decision()
                if decision == "s":
                    dataset.save_episode()
                    logger.info(
                        "Saved episode %d (%d frames): %s",
                        dataset.num_episodes - 1,
                        frames_added,
                        prompt,
                    )
                else:
                    dataset.clear_episode_buffer()
                    logger.info("Discarded trial: %s", prompt)
                    if decision == "q":
                        break

        logger.info("Camera videos saved under: %s", dataset.root / "videos")
    finally:
        if ctx is not None:
            disconnect(ctx)


def main() -> None:
    register_third_party_plugins()
    evaluate()


EVALUATOR_ARGS, ROLLOUT_ARGS = parse_evaluator_args()
sys.argv = [sys.argv[0], *ROLLOUT_ARGS]


if __name__ == "__main__":
    main()
