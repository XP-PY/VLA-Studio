#!/usr/bin/env python3
"""Capture a reference PNG through the same LeRobot camera configuration used for recording.

Example:

python runs/sam/ref/capture_lerobot_reference.py \
    --config configs/rm65_wrist_top.yaml \
    --camera top \
    --output runs/sam/ref/ref_Color.png \
    --overwrite
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import draccus

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.robots.config import RobotConfig


@dataclass
class RobotProfileConfig:
    robot: RobotConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rm65_wrist_top.yaml"),
        help="Robot YAML containing the camera configuration.",
    )
    parser.add_argument("--camera", default="top", help="Camera name under robot.cameras.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/sam/ref/ref_Color_lerobot.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=3.0,
        help="Additional time to read frames while auto-exposure settles.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output image.")
    return parser.parse_args()


def load_camera_config(config_path: Path, camera_name: str) -> RealSenseCameraConfig:
    profile = draccus.parse(RobotProfileConfig, config_path=str(config_path), args=[])
    try:
        camera_config = profile.robot.cameras[camera_name]
    except KeyError as exc:
        available = ", ".join(profile.robot.cameras)
        raise ValueError(f"Camera {camera_name!r} is not configured. Available cameras: {available}") from exc

    if not isinstance(camera_config, RealSenseCameraConfig):
        raise TypeError(
            f"Camera {camera_name!r} uses {type(camera_config).__name__}, "
            "but this script requires RealSenseCameraConfig."
        )

    return replace(camera_config, warmup_s=max(camera_config.warmup_s, 1))


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
    if args.settle_seconds < 0:
        raise ValueError("--settle-seconds must be non-negative")

    camera_config = load_camera_config(config_path, args.camera)
    camera = RealSenseCamera(camera_config)

    print(
        f"Connecting camera {args.camera!r}: serial={camera_config.serial_number_or_name}, "
        f"{camera_config.width}x{camera_config.height}@{camera_config.fps}"
    )

    frame = None
    try:
        camera.connect()
        deadline = time.monotonic() + args.settle_seconds
        while frame is None or time.monotonic() < deadline:
            frame = camera.read()
    finally:
        if camera.is_connected:
            camera.disconnect()

    if frame is None:
        raise RuntimeError("Camera returned no color frame.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = frame if camera_config.color_mode == ColorMode.BGR else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not save image to {output_path}")

    print(f"Saved {frame.shape[1]}x{frame.shape[0]} reference image to {output_path}")


if __name__ == "__main__":
    main()
