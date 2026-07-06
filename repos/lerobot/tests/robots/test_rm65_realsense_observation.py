#!/usr/bin/env python

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
import numpy as np

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.rm65 import RM65ArmConfig, RM65Robot, RM65RobotConfig

try:
    import pytest
except ModuleNotFoundError:
    pytest = None


DEFAULT_WRIST_SERIAL = "335222071948"
DEFAULT_FRONT_SERIAL = "335522070090"


@dataclass
class RobotProfileConfig:
    robot: RobotConfig


def make_config(
    ip: str | None = None,
    port: int | None = None,
    sdk_path: str | Path | None = None,
    calibration_dir: str | Path | None = None,
    wrist_serial: str | None = None,
    front_serial: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> RM65RobotConfig:
    del sdk_path

    wrist_serial = wrist_serial or os.getenv("RM65_WRIST_CAMERA_SERIAL", DEFAULT_WRIST_SERIAL)
    front_serial = front_serial or os.getenv("RM65_FRONT_CAMERA_SERIAL", DEFAULT_FRONT_SERIAL)

    return RM65RobotConfig(
        mode="single",
        single_arm=RM65ArmConfig(
            ip=ip or os.getenv("RM65_IP", "192.168.1.18"),
            port=port or int(os.getenv("RM65_PORT", "8080")),
        ),
        calibration_dir=Path(
            calibration_dir or os.getenv("RM65_CALIBRATION_DIR", "/tmp/lerobot-rm65-calibration")
        ).expanduser(),
        cameras={
            "wrist": RealSenseCameraConfig(
                serial_number_or_name=wrist_serial,
                width=width,
                height=height,
                fps=fps,
            ),
            "front": RealSenseCameraConfig(
                serial_number_or_name=front_serial,
                width=width,
                height=height,
                fps=fps,
            ),
        },
    )


def load_config(config_path: str | Path) -> RM65RobotConfig:
    cfg = draccus.parse(RobotProfileConfig, config_path=str(config_path), args=[])
    if not isinstance(cfg.robot, RM65RobotConfig):
        raise TypeError(
            f"Expected config_path to contain an rm65 robot config, got {type(cfg.robot).__name__}"
        )
    return cfg.robot


def _validate_camera_frame(obs: dict[str, Any], key: str, width: int, height: int) -> np.ndarray:
    if key not in obs:
        raise AssertionError(f"Observation is missing camera key `{key}`. Available keys: {sorted(obs)}")

    frame = obs[key]
    if not isinstance(frame, np.ndarray):
        raise AssertionError(f"`{key}` frame must be a numpy array, got {type(frame).__name__}")
    if frame.shape != (height, width, 3):
        raise AssertionError(f"`{key}` frame shape is {frame.shape}, expected {(height, width, 3)}")
    if frame.dtype != np.uint8:
        raise AssertionError(f"`{key}` frame dtype is {frame.dtype}, expected uint8")

    return frame


def connect_and_read_cameras_once(cfg: RM65RobotConfig, width: int, height: int) -> dict:
    robot = RM65Robot(cfg)
    try:
        robot.connect()
        obs = robot.get_observation()

        wrist = _validate_camera_frame(obs, "wrist", width, height)
        front = _validate_camera_frame(obs, "front", width, height)

        print(f"Connected to RM65 at {robot.default_arm_config.ip}:{robot.default_arm_config.port}")
        print("Robot observation keys:")
        for key in sorted(obs):
            value = obs[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")

        print("Camera frame summaries:")
        print(f"  wrist: min={wrist.min()}, max={wrist.max()}, mean={wrist.mean():.2f}")
        print(f"  front: min={front.min()}, max={front.max()}, mean={front.mean():.2f}")

        return obs
    finally:
        if robot.is_connected:
            robot.disconnect()


def test_rm65_realsense_observation():
    if pytest is None:
        raise RuntimeError("pytest is required to run this as a pytest test.")
    if os.getenv("RUN_RM65_REALSENSE_TEST") != "1":
        pytest.skip("Set RUN_RM65_REALSENSE_TEST=1 to run this hardware test.")

    width = int(os.getenv("RM65_CAMERA_WIDTH", "640"))
    height = int(os.getenv("RM65_CAMERA_HEIGHT", "480"))
    fps = int(os.getenv("RM65_CAMERA_FPS", "30"))
    obs = connect_and_read_cameras_once(make_config(width=width, height=height, fps=fps), width, height)

    for key in [
        "joint_1.pos",
        "joint_2.pos",
        "joint_3.pos",
        "joint_4.pos",
        "joint_5.pos",
        "joint_6.pos",
        "wrist",
        "front",
    ]:
        assert key in obs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect to RM65 and read one observation from both RealSense cameras."
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="YAML config with a top-level `robot` block, for example configs/rm65_realsense.yaml.",
    )
    parser.add_argument(
        "--ip",
        default=None,
        help="Robot controller IP. Defaults to RM65_IP or 192.168.1.18.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Robot controller port. Defaults to RM65_PORT or 8080.",
    )
    parser.add_argument(
        "--sdk-path",
        default=None,
        help="Ignored. RM65 now uses the controller JSON/TCP protocol directly.",
    )
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help=(
            "LeRobot calibration directory. Defaults to RM65_CALIBRATION_DIR "
            "or /tmp/lerobot-rm65-calibration."
        ),
    )
    parser.add_argument(
        "--wrist-serial",
        default=None,
        help=f"Wrist/end-effector RealSense serial. Defaults to {DEFAULT_WRIST_SERIAL}.",
    )
    parser.add_argument(
        "--front-serial",
        default=None,
        help=f"Front/overview RealSense serial. Defaults to {DEFAULT_FRONT_SERIAL}.",
    )
    parser.add_argument("--width", type=int, default=640, help="Color frame width.")
    parser.add_argument("--height", type=int, default=480, help="Color frame height.")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS.")
    args = parser.parse_args()

    if args.config_path:
        cfg = load_config(args.config_path)
        width = cfg.cameras["wrist"].width
        height = cfg.cameras["wrist"].height
        if width is None or height is None:
            raise ValueError("Camera width and height must be set in the config file.")
        connect_and_read_cameras_once(cfg, width=width, height=height)
        return

    connect_and_read_cameras_once(
        make_config(
            ip=args.ip,
            port=args.port,
            sdk_path=args.sdk_path,
            calibration_dir=args.calibration_dir,
            wrist_serial=args.wrist_serial,
            front_serial=args.front_serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
        ),
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
