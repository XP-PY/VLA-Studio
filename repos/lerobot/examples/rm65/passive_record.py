#!/usr/bin/env python
"""Passively record RM65 demonstrations in LeRobot dataset format.

This script is for setups where an external teleoperator/master arm already
controls the RM65 directly. It only reads robot state and camera observations,
then stores LeRobot-compatible observation/action frames; it does not send arm
motion commands during recording.

For the RealMan Aloha-style teleoperator, use ``--io-read-source aloha`` with
``--teleop-active-io 1`` and ``--record-button-io 2``. IO1 indicates that the
teleoperator is active, and IO2 is the momentary enable button used as a
recording toggle. The robot config's ``control_arm`` field selects which arm
provides these signals. It defaults to ``right`` in bimanual mode.

If ``--root`` already contains a LeRobot dataset, new episodes are appended to
that dataset so collection can be split across multiple recording sessions.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus

from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.robots.config import RobotConfig
from lerobot.robots.rm65 import RM65Robot, RM65RobotConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

IO_READ_SOURCES = ("state", "input", "output", "tool", "aloha", "auto")


@dataclass
class RobotProfileConfig:
    robot: RobotConfig


def load_robot_config(config_path: str | Path) -> RM65RobotConfig:
    cfg = draccus.parse(RobotProfileConfig, config_path=str(config_path), args=[])
    if not isinstance(cfg.robot, RM65RobotConfig):
        raise TypeError(f"Expected an rm65 robot config, got {type(cfg.robot).__name__}")
    return cfg.robot


def make_dataset_features(robot: RM65Robot, use_videos: bool) -> dict[str, dict]:
    return combine_feature_dicts(
        hw_to_dataset_features(robot.action_features, ACTION, use_video=use_videos),
        hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=use_videos),
    )


def action_from_future_observation(action_names: list[str], future_obs: dict[str, Any]) -> dict[str, float]:
    missing = [name for name in action_names if name not in future_obs]
    if missing:
        raise KeyError(f"Future observation is missing action source keys: {missing}")
    return {name: float(future_obs[name]) for name in action_names}


def add_passive_frame(
    dataset: LeRobotDataset,
    obs: dict[str, Any],
    future_obs: dict[str, Any],
    action_names: list[str],
    task: str,
) -> None:
    action = action_from_future_observation(action_names, future_obs)
    obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
    action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
    dataset.add_frame({**obs_frame, **action_frame, "task": task})


def is_lerobot_dataset_root(root: str | Path) -> bool:
    root = Path(root).expanduser()
    return (root / "meta" / "info.json").exists() and (root / "meta" / "tasks.parquet").exists()


def move_incomplete_dataset_stub(root: Path) -> None:
    if not root.exists():
        return

    files = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())
    if files != [Path("meta/info.json")]:
        missing = [
            str(path)
            for path in [root / "meta" / "tasks.parquet", root / "meta" / "episodes.parquet"]
            if not path.exists()
        ]
        raise RuntimeError(
            f"Dataset root exists but is incomplete: {root}. Missing: {missing}. "
            "Use a different --root or move this directory aside manually."
        )

    backup = root.with_name(f"{root.name}.incomplete-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.move(str(root), str(backup))
    logger.warning("Moved incomplete dataset stub from %s to %s.", root, backup)


def create_or_resume_dataset(
    args: argparse.Namespace,
    robot: RM65Robot,
    features: dict[str, dict],
    use_videos: bool,
) -> LeRobotDataset:
    root = Path(args.root).expanduser()
    image_writer_threads = args.image_writer_threads_per_camera * len(robot.cameras)
    if is_lerobot_dataset_root(root):
        dataset = LeRobotDataset.resume(
            repo_id=args.repo_id,
            root=root,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=image_writer_threads,
            streaming_encoding=args.streaming_encoding,
        )
        if dataset.fps != args.fps:
            logger.warning("Existing dataset fps=%s; ignoring requested --fps=%s.", dataset.fps, args.fps)
        if dataset.features != features:
            logger.warning("Existing dataset features differ from current robot config; appending may fail.")
        if bool(dataset.meta.video_keys) != use_videos:
            logger.warning("Existing dataset video setting differs from --no-video; using existing metadata.")
        logger.info(
            "Appending to existing LeRobot dataset at %s with %s episode(s), %s frame(s).",
            dataset.root,
            dataset.num_episodes,
            dataset.num_frames,
        )
        return dataset

    move_incomplete_dataset_stub(root)

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=root,
        robot_type=robot.name,
        use_videos=use_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=image_writer_threads,
        streaming_encoding=args.streaming_encoding,
    )


def _read_indexed_io_values(values: Any, io_num: int, label: str) -> bool:
    if not isinstance(values, list):
        raise RuntimeError(f"{label} did not return a list: {values}")
    if io_num < 1 or io_num > len(values):
        raise ValueError(f"IO number {io_num} is outside {label} range 1..{len(values)}")
    value = values[io_num - 1]
    if value == -1:
        raise RuntimeError(f"{label}[{io_num}] is unavailable because this IO is not in that mode")
    return bool(value)


def resolve_control_side(robot: RM65Robot) -> str:
    control_arm = robot.config.control_arm
    if control_arm is None:
        raise RuntimeError("RM65 control_arm was not resolved during configuration")
    if control_arm == "single":
        control_side = ""
    else:
        control_side = control_arm

    if control_side not in robot.arm_configs:
        available = ["single" if side == "" else side for side in robot.arm_configs]
        raise ValueError(f"control_arm={control_arm!r} is invalid for this config. Available: {available}")
    return control_side


def control_side_label(control_side: str) -> str:
    return control_side or "single"


def _read_controller_io_source(
    robot: RM65Robot,
    io_num: int,
    io_read_source: str,
    control_side: str,
) -> bool:
    client = robot._client_for(control_side)
    if io_read_source == "state":
        state = client.get_io_state(io_num)
        return bool(state["IO_state"])
    if io_read_source == "input":
        values = client.get_io_input()
        return _read_indexed_io_values(values, io_num, "rm_get_io_input")
    if io_read_source == "output":
        values = client.get_io_output()
        return _read_indexed_io_values(values, io_num, "rm_get_io_output")
    if io_read_source == "tool":
        state = client.get_tool_io_state()
        return _read_indexed_io_values(state.get("IO_State"), io_num, "rm_get_tool_io_state")
    if io_read_source == "auto":
        errors = []
        for source in ("input", "state", "output", "tool"):
            try:
                return _read_controller_io_source(robot, io_num, source, control_side)
            except Exception as exc:
                errors.append(f"{source}: {exc}")
        raise RuntimeError(f"Could not read IO{io_num} from any source: {'; '.join(errors)}")
    raise ValueError(f"Unsupported IO read source: {io_read_source}")


def read_controller_io(
    robot: RM65Robot,
    io_num: int,
    io_read_source: str,
    io_active_low: bool,
    control_side: str,
    aloha_reader: AlohaStateReader | None = None,
) -> bool:
    if io_read_source == "aloha":
        if aloha_reader is None:
            raise RuntimeError("Aloha IO source requires an active AlohaStateReader")
        value = aloha_reader.read_io(io_num)
    else:
        value = _read_controller_io_source(robot, io_num, io_read_source, control_side)
    return not value if io_active_low else value


def configure_io_inputs(robot: RM65Robot, io_nums: list[int], control_side: str) -> None:
    client = robot._client_for(control_side)
    for io_num in sorted(set(io_nums)):
        client.set_io_mode(io_num, 0)
        logger.info("Set %s-arm IO%s to general input mode.", control_side_label(control_side), io_num)


class AlohaStateReader:
    def __init__(
        self,
        robot: RM65Robot,
        control_side: str,
        target_ip: str | None,
        push_port: int,
        cycle_ms: int,
    ) -> None:
        self.robot = robot
        self.control_side = control_side
        arm_config = robot.arm_configs[control_side]
        self.target_ip = target_ip or self._local_ip_for_robot(arm_config.ip)
        self.push_port = push_port
        self.cycle_ms = cycle_ms
        self.json_cycle = max(1, int(round(cycle_ms / 5)))
        self._latest: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._started = False
        self._accepted_cycle = cycle_ms
        self._accepted_force_coordinate: int | None = 2
        self._last_error_log_s = 0.0
        self._last_joint_clear_s = 0.0

    @staticmethod
    def _local_ip_for_robot(robot_ip: str) -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((robot_ip, 1))
            return sock.getsockname()[0]

    def start(self) -> None:
        client = self.robot._client_for(self.control_side)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.2)
        self._sock.bind(("", self.push_port))
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

        full_custom = {
            "joint_speed": False,
            "lift_state": False,
            "expand_state": False,
            "arm_current_status": True,
            "hand": False,
            "aloha_state": True,
            "rm_plus_base": False,
            "rm_plus_state": False,
        }
        minimal_custom = {"arm_current_status": True, "aloha_state": True}
        attempts = [
            ("ms-cycle force=2 full-custom", self.cycle_ms, 2, full_custom),
            ("json-cycle force=2 full-custom", self.json_cycle, 2, full_custom),
            ("ms-cycle force=0 full-custom", self.cycle_ms, 0, full_custom),
            ("ms-cycle no-force full-custom", self.cycle_ms, None, full_custom),
            ("ms-cycle force=2 minimal-custom", self.cycle_ms, 2, minimal_custom),
            ("ms-cycle no-force minimal-custom", self.cycle_ms, None, minimal_custom),
            ("json-cycle no-force minimal-custom", self.json_cycle, None, minimal_custom),
        ]
        errors: list[str] = []
        try:
            for label, cycle, force_coordinate, custom in attempts:
                try:
                    client.set_realtime_push(
                        enable=True,
                        target_ip=self.target_ip,
                        port=self.push_port,
                        cycle=cycle,
                        force_coordinate=force_coordinate,
                        custom=custom,
                    )
                except Exception as exc:
                    errors.append(f"{label}: {exc}")
                    continue

                self._accepted_cycle = cycle
                self._accepted_force_coordinate = force_coordinate
                logger.info("Accepted RM65 realtime push command variant: %s.", label)
                break
            else:
                raise RuntimeError(
                    "RM65 set_realtime_push failed for all JSON variants: " + "; ".join(errors)
                )
        except Exception:
            self._close_udp_listener()
            raise
        self._started = True
        logger.info(
            "Started %s-arm RM65 realtime aloha_state trigger on %s:%s cycle_ms=%s accepted_cycle=%s.",
            control_side_label(self.control_side),
            self.target_ip,
            self.push_port,
            self.cycle_ms,
            self._accepted_cycle,
        )

    def stop(self) -> None:
        if self.robot.is_connected and self._started:
            try:
                self.robot._client_for(self.control_side).set_realtime_push(
                    enable=False,
                    target_ip=self.target_ip,
                    port=self.push_port,
                    cycle=self._accepted_cycle,
                    force_coordinate=self._accepted_force_coordinate,
                    custom={
                        "joint_speed": False,
                        "lift_state": False,
                        "expand_state": False,
                        "arm_current_status": False,
                        "hand": False,
                        "aloha_state": False,
                        "rm_plus_base": False,
                        "rm_plus_state": False,
                    },
                )
            except Exception as exc:
                logger.warning("set_realtime_push disable failed: %s.", exc)
        self._started = False
        self._close_udp_listener()

    def _close_udp_listener(self) -> None:
        self._stop_event.set()
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _read_loop(self) -> None:
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                packet, _addr = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                data = json.loads(packet.decode("utf-8"))
            except Exception as exc:
                logger.debug("Ignoring malformed RM65 realtime UDP packet: %s", exc)
                continue

            aloha = data.get("aloha_state")
            if not isinstance(aloha, dict):
                continue

            try:
                latest = {
                    "err_code": self._as_int(data.get("err", 0), default=0),
                    "arm_current_status": data.get("arm_current_status", ""),
                    "io1": self._as_int(aloha.get("io1_state", 0), default=0),
                    "io2": self._as_int(aloha.get("io2_state", 0), default=0),
                }
            except Exception as exc:
                logger.debug("Ignoring RM65 realtime UDP packet with unexpected shape: %s", exc)
                continue

            with self._lock:
                self._latest = latest

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        if isinstance(value, (list, tuple)):
            for item in value:
                converted = AlohaStateReader._as_int(item, default=default)
                if converted != default:
                    return converted
            return default
        return default

    def _handle_realtime_error(self, latest: dict[str, Any]) -> None:
        now_s = time.monotonic()
        if now_s - self._last_error_log_s >= 2.0:
            self._last_error_log_s = now_s
            logger.warning(
                "RM65 realtime aloha_state reported err_code=%s, status=%s. "
                "Using IO values and clearing joint errors.",
                latest.get("err_code"),
                latest.get("arm_current_status"),
            )

        if now_s - self._last_joint_clear_s < 1.0:
            return

        self._last_joint_clear_s = now_s
        clear_rm65_joint_errors(self.robot, sides=[self.control_side])

    def read_io(self, io_num: int, timeout_s: float = 2.0) -> bool:
        if io_num not in (1, 2):
            raise ValueError("Aloha realtime trigger only exposes IO1 and IO2")

        deadline = time.monotonic() + timeout_s
        key = f"io{io_num}"
        while time.monotonic() < deadline:
            with self._lock:
                latest = dict(self._latest)
            if key in latest:
                if latest.get("err_code") != 0:
                    self._handle_realtime_error(latest)
                return bool(latest[key])
            precise_sleep(0.01)
        raise RuntimeError("Timed out waiting for realtime aloha_state callback")


def wait_for_button_press(
    robot: RM65Robot,
    record_button_io: int,
    poll_interval_s: float,
    io_read_source: str,
    io_active_low: bool,
    control_side: str,
    aloha_reader: AlohaStateReader | None,
) -> None:
    while read_controller_io(
        robot, record_button_io, io_read_source, io_active_low, control_side, aloha_reader
    ):
        precise_sleep(poll_interval_s)

    while not read_controller_io(
        robot, record_button_io, io_read_source, io_active_low, control_side, aloha_reader
    ):
        precise_sleep(poll_interval_s)

    while read_controller_io(
        robot, record_button_io, io_read_source, io_active_low, control_side, aloha_reader
    ):
        precise_sleep(poll_interval_s)


def wait_for_io_start(
    robot: RM65Robot,
    record_button_io: int | None,
    teleop_active_io: int | None,
    record_button_mode: str,
    poll_interval_s: float,
    io_read_source: str,
    io_active_low: bool,
    control_side: str,
    aloha_reader: AlohaStateReader | None,
) -> None:
    if record_button_io is None and teleop_active_io is None:
        logger.info("Start recording.")
        return

    if record_button_mode == "toggle" and record_button_io is None:
        raise ValueError("--record-button-mode=toggle requires --record-button-io")

    if teleop_active_io is not None:
        logger.info(
            "Waiting for IO%s active before recording. source=%s active_low=%s",
            teleop_active_io,
            io_read_source,
            io_active_low,
        )
        while not read_controller_io(
            robot, teleop_active_io, io_read_source, io_active_low, control_side, aloha_reader
        ):
            precise_sleep(poll_interval_s)

    if record_button_io is None:
        logger.info("Start recording.")
        return

    if record_button_mode == "level":
        logger.info(
            "Waiting for IO%s active to start recording. source=%s active_low=%s",
            record_button_io,
            io_read_source,
            io_active_low,
        )
        while not read_controller_io(
            robot, record_button_io, io_read_source, io_active_low, control_side, aloha_reader
        ):
            precise_sleep(poll_interval_s)
        logger.info("Start recording.")
        logger.info("Recording will stop when IO%s goes inactive.", record_button_io)
    else:
        logger.info(
            "Waiting for IO%s press to start recording. source=%s active_low=%s",
            record_button_io,
            io_read_source,
            io_active_low,
        )
        wait_for_button_press(
            robot, record_button_io, poll_interval_s, io_read_source, io_active_low, control_side, aloha_reader
        )
        logger.info("Start recording.")
        logger.info("Waiting for IO%s press to stop.", record_button_io)


def should_stop_recording(
    robot: RM65Robot,
    record_button_io: int | None,
    teleop_active_io: int | None,
    record_button_mode: str,
    previous_button_state: bool,
    io_read_source: str,
    io_active_low: bool,
    control_side: str,
    aloha_reader: AlohaStateReader | None,
) -> tuple[bool, bool, str | None]:
    if teleop_active_io is not None and not read_controller_io(
        robot, teleop_active_io, io_read_source, io_active_low, control_side, aloha_reader
    ):
        return True, previous_button_state, f"IO{teleop_active_io} went low"

    if record_button_io is None:
        return False, previous_button_state, None

    button_state = read_controller_io(
        robot, record_button_io, io_read_source, io_active_low, control_side, aloha_reader
    )
    if record_button_mode == "level" and not button_state:
        return True, button_state, f"IO{record_button_io} went low"
    if record_button_mode == "toggle" and button_state and not previous_button_state:
        return True, button_state, f"IO{record_button_io} pressed"

    return False, button_state, None


def wait_for_record_button_release(
    robot: RM65Robot,
    record_button_io: int | None,
    poll_interval_s: float,
    io_read_source: str,
    io_active_low: bool,
    control_side: str,
    aloha_reader: AlohaStateReader | None,
) -> None:
    if record_button_io is None:
        return

    while read_controller_io(
        robot, record_button_io, io_read_source, io_active_low, control_side, aloha_reader
    ):
        precise_sleep(poll_interval_s)


def clear_rm65_joint_errors(robot: RM65Robot, sides: list[str] | None = None) -> None:
    if sides is None:
        sides = list(robot.arm_configs)

    failures = []
    for side in sides:
        client = robot._client_for(side)
        side_label = control_side_label(side)
        for joint_num in range(1, len(robot.arm_configs[side].joint_names) + 1):
            try:
                client.set_joint_clear_err(joint_num)
            except Exception as exc:
                failures.append(f"{side_label} joint {joint_num}: {exc}")

    if failures:
        logger.warning("RM65 joint error clear had failures: %s", "; ".join(failures))
    else:
        logger.info("Cleared RM65 joint errors for %s.", ", ".join(control_side_label(side) for side in sides))


def read_observation_with_joint_error_recovery(
    robot: RM65Robot,
    max_attempts: int,
    retry_sleep_s: float,
) -> dict[str, Any] | None:
    for attempt_idx in range(max_attempts + 1):
        try:
            return robot.get_observation()
        except Exception as exc:
            if attempt_idx >= max_attempts:
                logger.warning(
                    "Skipping frame after %s failed observation read(s): %s",
                    attempt_idx + 1,
                    exc,
                )
                return None

            logger.warning(
                "RM65 observation read failed: %s. Clearing joint errors and retrying (%s/%s).",
                exc,
                attempt_idx + 1,
                max_attempts,
            )
            clear_rm65_joint_errors(robot)
            precise_sleep(retry_sleep_s)

    return None


def record_passive_episode(
    robot: RM65Robot,
    dataset: LeRobotDataset,
    task: str,
    future_offset: int,
    record_button_io: int | None,
    teleop_active_io: int | None,
    record_button_mode: str,
    io_poll_interval_s: float,
    io_read_source: str,
    io_active_low: bool,
    control_side: str,
    aloha_reader: AlohaStateReader | None,
    joint_error_recovery_attempts: int,
    joint_error_recovery_sleep_s: float,
) -> int:
    wait_for_io_start(
        robot=robot,
        record_button_io=record_button_io,
        teleop_active_io=teleop_active_io,
        record_button_mode=record_button_mode,
        poll_interval_s=io_poll_interval_s,
        io_read_source=io_read_source,
        io_active_low=io_active_low,
        control_side=control_side,
        aloha_reader=aloha_reader,
    )

    action_names = list(dataset.features[ACTION]["names"])
    pending: deque[dict[str, Any]] = deque()
    control_interval_s = 1.0 / dataset.fps
    frames_added = 0
    previous_button_state = bool(
        record_button_io is not None
        and read_controller_io(
            robot, record_button_io, io_read_source, io_active_low, control_side, aloha_reader
        )
    )

    while True:
        loop_start = time.perf_counter()
        stop, previous_button_state, reason = should_stop_recording(
            robot=robot,
            record_button_io=record_button_io,
            teleop_active_io=teleop_active_io,
            record_button_mode=record_button_mode,
            previous_button_state=previous_button_state,
            io_read_source=io_read_source,
            io_active_low=io_active_low,
            control_side=control_side,
            aloha_reader=aloha_reader,
        )
        if stop:
            logger.info("Stopping episode: %s.", reason)
            break

        obs = read_observation_with_joint_error_recovery(
            robot,
            max_attempts=joint_error_recovery_attempts,
            retry_sleep_s=joint_error_recovery_sleep_s,
        )
        if obs is None:
            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(control_interval_s - dt_s, 0.0))
            continue

        if future_offset == 0:
            add_passive_frame(dataset, obs, obs, action_names, task)
            frames_added += 1
        else:
            pending.append(obs)
            if len(pending) > future_offset:
                current_obs = pending.popleft()
                future_obs = pending[-1]
                add_passive_frame(dataset, current_obs, future_obs, action_names, task)
                frames_added += 1

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(control_interval_s - dt_s, 0.0))

    return frames_added


def wait_for_reset(reset_time_s: float) -> None:
    if reset_time_s <= 0:
        return
    logger.info("Reset environment for %.1fs. Recording is paused.", reset_time_s)
    precise_sleep(reset_time_s)


def ask_save_or_discard(frames_added: int) -> bool:
    logger.info("Episode recorded with %s frame(s).", frames_added)
    while True:
        choice = input("Save this episode? [s=save, d=discard]: ").strip().lower()
        if choice == "s":
            return True
        if choice == "d":
            return False
        print("Please enter 's' to save or 'd' to discard.")


def setup_logging(log_path: str | Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        path = Path(log_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Passively record an RM65 LeRobot dataset while an external/master "
            "controller moves the robot."
        )
    )
    parser.add_argument("--config-path", default="configs/rm65_wrist_front.yaml")
    parser.add_argument("--repo-id", default="local/rm65_passive_test")
    parser.add_argument(
        "--root",
        default="outputs/rm65_passive_test",
        help="Dataset directory. If it already contains a LeRobot dataset, append new episodes.",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--reset-time-s", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--record-button-io",
        type=int,
        default=None,
        help="IO number used as record trigger.",
    )
    parser.add_argument(
        "--record-button-mode",
        choices=["level", "toggle"],
        default="level",
        help="'level': record while IO is high. 'toggle': press once to start, press again to stop.",
    )
    parser.add_argument(
        "--teleop-active-io",
        type=int,
        default=None,
        help="Optional IO that must be high before/during recording.",
    )
    parser.add_argument(
        "--io-read-source",
        choices=IO_READ_SOURCES,
        default="state",
        help=(
            "SDK field used to read --record-button-io and --teleop-active-io. "
            "Use 'input' after setting controller IOs to input mode; use 'tool' "
            "for end-effector IO1/IO2; use 'aloha' for realtime teleoperator IO1/IO2."
        ),
    )
    parser.add_argument(
        "--aloha-target-ip",
        default=None,
        help=(
            "Local PC IP used as the RealMan UDP realtime push target when "
            "--io-read-source=aloha. Defaults to auto-detect from route to robot."
        ),
    )
    parser.add_argument(
        "--aloha-push-port",
        type=int,
        default=8098,
        help="Local UDP port for RealMan realtime aloha_state push.",
    )
    parser.add_argument(
        "--aloha-cycle-ms",
        type=int,
        default=20,
        help="RealMan realtime push cycle in milliseconds. Must be a 5 ms multiple.",
    )
    parser.add_argument(
        "--io-active-low",
        action="store_true",
        help="Treat IO value 0 as active and IO value 1 as inactive.",
    )
    parser.add_argument(
        "--io-set-input",
        action="store_true",
        help="Set selected controller IOs to general input mode before recording.",
    )
    parser.add_argument("--io-poll-hz", type=float, default=20.0)
    parser.add_argument(
        "--future-offset",
        type=int,
        default=1,
        help="Action target frame offset. At 10 FPS, 1 means about 100 ms in the future.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Store camera frames as images instead of videos.",
    )
    parser.add_argument("--streaming-encoding", action="store_true")
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads-per-camera", type=int, default=4)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--log-path", default=None, help="Optional file path for recording logs.")
    parser.add_argument(
        "--ask-save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After each episode, ask whether to save it. Use --no-ask-save to save automatically.",
    )
    parser.add_argument(
        "--joint-error-recovery-attempts",
        type=int,
        default=3,
        help=(
            "Observation read retries after clearing RM65 joint errors with set_joint_clear_err. "
            "Set to 0 to skip failed frames without retrying."
        ),
    )
    parser.add_argument(
        "--joint-error-recovery-sleep-s",
        type=float,
        default=0.05,
        help="Sleep time between set_joint_clear_err recovery and the next observation read.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.future_offset < 0:
        raise ValueError("--future-offset must be >= 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.io_poll_hz <= 0:
        raise ValueError("--io-poll-hz must be > 0")
    if args.record_button_mode == "toggle" and args.record_button_io is None:
        raise ValueError("--record-button-mode=toggle requires --record-button-io")
    if args.io_read_source not in IO_READ_SOURCES:
        raise ValueError(f"--io-read-source must be one of {IO_READ_SOURCES}")
    if args.joint_error_recovery_attempts < 0:
        raise ValueError("--joint-error-recovery-attempts must be >= 0")
    if args.joint_error_recovery_sleep_s < 0:
        raise ValueError("--joint-error-recovery-sleep-s must be >= 0")
    if args.io_set_input and args.io_read_source in ("tool", "aloha"):
        raise ValueError("--io-set-input only applies to controller IO, not tool/aloha IO.")
    if args.io_read_source == "aloha":
        io_nums = [io for io in [args.teleop_active_io, args.record_button_io] if io is not None]
        invalid_io_nums = [io for io in io_nums if io not in (1, 2)]
        if invalid_io_nums:
            raise ValueError(f"--io-read-source=aloha only supports IO1/IO2, got {invalid_io_nums}")
        if args.aloha_cycle_ms <= 0 or args.aloha_cycle_ms % 5 != 0:
            raise ValueError("--aloha-cycle-ms must be a positive 5 ms multiple")

    setup_logging(args.log_path)

    robot_cfg = load_robot_config(args.config_path)
    robot = RM65Robot(robot_cfg)
    control_side = resolve_control_side(robot)
    use_videos = not args.no_video
    features = make_dataset_features(robot, use_videos=use_videos)

    dataset = create_or_resume_dataset(args, robot, features, use_videos)

    aloha_reader = None
    logger.info("Passive recording mode.")
    logger.info("Move the RM65 with your external/master controller after recording starts.")
    logger.info("LeRobot will only read robot state and camera frames; it will not send arm actions.")
    logger.info("Robot mode: %s", robot.config.mode)
    logger.info("IO/Aloha control arm: %s", control_side_label(control_side))
    logger.info("Dataset root: %s", dataset.root)
    logger.info("Action names: %s", dataset.features[ACTION]["names"])
    if args.log_path:
        logger.info("Logging to: %s", Path(args.log_path).expanduser())

    try:
        robot.connect()
        if args.io_read_source == "aloha" and (
            args.record_button_io is not None or args.teleop_active_io is not None
        ):
            aloha_reader = AlohaStateReader(
                robot=robot,
                control_side=control_side,
                target_ip=args.aloha_target_ip,
                push_port=args.aloha_push_port,
                cycle_ms=args.aloha_cycle_ms,
            )
            aloha_reader.start()

        if args.io_set_input:
            io_nums = [
                io_num
                for io_num in [args.teleop_active_io, args.record_button_io]
                if io_num is not None
            ]
            configure_io_inputs(robot, io_nums, control_side)

        with VideoEncodingManager(dataset):
            saved_episodes = 0
            attempt_idx = 0
            while saved_episodes < args.num_episodes:
                attempt_idx += 1
                logger.info(
                    "Preparing episode %s/%s (attempt %s).",
                    saved_episodes + 1,
                    args.num_episodes,
                    attempt_idx,
                )
                frames_added = record_passive_episode(
                    robot=robot,
                    dataset=dataset,
                    task=args.task,
                    future_offset=args.future_offset,
                    record_button_io=args.record_button_io,
                    teleop_active_io=args.teleop_active_io,
                    record_button_mode=args.record_button_mode,
                    io_poll_interval_s=1.0 / args.io_poll_hz,
                    io_read_source=args.io_read_source,
                    io_active_low=args.io_active_low,
                    control_side=control_side,
                    aloha_reader=aloha_reader,
                    joint_error_recovery_attempts=args.joint_error_recovery_attempts,
                    joint_error_recovery_sleep_s=args.joint_error_recovery_sleep_s,
                )
                if frames_added == 0:
                    dataset.clear_episode_buffer()
                    raise RuntimeError(
                        "No frames were recorded. Check the record trigger or lower --future-offset."
                    )

                wait_for_record_button_release(
                    robot=robot,
                    record_button_io=args.record_button_io,
                    poll_interval_s=1.0 / args.io_poll_hz,
                    io_read_source=args.io_read_source,
                    io_active_low=args.io_active_low,
                    control_side=control_side,
                    aloha_reader=aloha_reader,
                )

                if args.ask_save:
                    should_save = ask_save_or_discard(frames_added)
                else:
                    should_save = True

                if should_save:
                    dataset.save_episode()
                    saved_episodes += 1
                    logger.info(
                        "Saved episode %s/%s with %s frames.",
                        saved_episodes,
                        args.num_episodes,
                        frames_added,
                    )
                else:
                    dataset.clear_episode_buffer()
                    logger.info(
                        "Discarded attempt %s with %s frames. Recording another episode for slot %s/%s.",
                        attempt_idx,
                        frames_added,
                        saved_episodes + 1,
                        args.num_episodes,
                    )

                if saved_episodes < args.num_episodes:
                    wait_for_reset(args.reset_time_s)
    finally:
        if aloha_reader is not None:
            aloha_reader.stop()
        if robot.is_connected:
            robot.disconnect()
        dataset.finalize()

    if args.push_to_hub:
        dataset.push_to_hub(private=args.private)

    logger.info("Done. Recorded %s episode(s), %s frame(s).", dataset.num_episodes, dataset.num_frames)


if __name__ == "__main__":
    main()
