#!/usr/bin/env python
"""Smoke-test an RM65 single-arm or bimanual config.

The script loads an RM65 YAML config, connects the configured arm(s) and cameras,
prints observation/action keys, captures camera frames, and saves all camera
frames into one combined local image. Use `--move` to additionally perform a
small joint move and return to the original position.

Usage for single-arm config, connect + capture cameras only:

python examples/test_rm65_config.py \
--config-path configs/rm65_wrist_front.yaml \
--output examples/test_rm65_config_single_test.jpg

Usage for bimanual config:

python examples/test_rm65_config.py \
--config-path configs/rm65_left_right_wrist_front_top.yaml \
--output examples/test_rm65_config_bimanual_test.jpg

To also do a small move and return:

python examples/test_rm65_config.py \
--config-path configs/rm65_left_right_wrist_front_top.yaml \
--output examples/test_rm65_config_bimanual_test.jpg \
--move \
--joint joint_6 \
--delta-deg 20.0 \
--sides all

For only one arm in bimanual mode:

--sides left

or:

--sides right
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import draccus
import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.robots.config import RobotConfig
from lerobot.robots.rm65 import RM65Robot, RM65RobotConfig


@dataclass
class RobotProfileConfig:
    robot: RobotConfig


def load_robot_config(config_path: str | Path) -> RM65RobotConfig:
    cfg = draccus.parse(RobotProfileConfig, config_path=str(config_path), args=[])
    if not isinstance(cfg.robot, RM65RobotConfig):
        raise TypeError(f"Expected an rm65 robot config, got {type(cfg.robot).__name__}")
    return cfg.robot


def arm_sides(robot: RM65Robot, requested: str) -> list[str]:
    if requested == "all":
        return list(robot.arm_configs)

    sides = [side.strip() for side in requested.split(",") if side.strip()]
    invalid = [side for side in sides if side not in robot.arm_configs]
    if invalid:
        available = ["single" if side == "" else side for side in robot.arm_configs]
        raise ValueError(f"Invalid side(s) {invalid}. Available sides: {available}")
    return sides


def action_for_joint_delta(robot: RM65Robot, obs: dict[str, Any], sides: list[str], joint: str, delta: float) -> dict[str, float]:
    action: dict[str, float] = {}
    for side in sides:
        prefix = robot._prefix(side)
        arm_config = robot.arm_configs[side]
        if joint not in arm_config.joint_names:
            raise ValueError(f"Joint `{joint}` is not in {side or 'single'} arm joint_names={arm_config.joint_names}")

        for arm_joint in arm_config.joint_names:
            key = f"{prefix}{arm_joint}.pos"
            action[key] = float(obs[key])
        action[f"{prefix}{joint}.pos"] = float(obs[f"{prefix}{joint}.pos"]) + delta
    return action


def action_for_original_joints(robot: RM65Robot, obs: dict[str, Any], sides: list[str]) -> dict[str, float]:
    action: dict[str, float] = {}
    for side in sides:
        prefix = robot._prefix(side)
        for joint in robot.arm_configs[side].joint_names:
            key = f"{prefix}{joint}.pos"
            action[key] = float(obs[key])
    return action


def frame_to_bgr(robot: RM65Robot, key: str, frame: np.ndarray) -> np.ndarray:
    camera_config = robot.config.cameras[key]
    color_mode = getattr(camera_config, "color_mode", ColorMode.RGB)
    if color_mode == ColorMode.RGB:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def make_labeled_tile(label: str, frame_bgr: np.ndarray, tile_width: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    scale = tile_width / max(width, 1)
    tile_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame_bgr, (tile_width, tile_height), interpolation=cv2.INTER_AREA)

    label_height = 34
    tile = np.zeros((tile_height + label_height, tile_width, 3), dtype=np.uint8)
    tile[:label_height, :] = (32, 32, 32)
    tile[label_height:, :] = resized
    cv2.putText(tile, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def save_camera_grid(
    robot: RM65Robot,
    obs: dict[str, Any],
    output_path: Path,
    cols: int,
    tile_width: int,
) -> None:
    camera_keys = [key for key in robot.cameras if key in obs and isinstance(obs[key], np.ndarray)]
    if not camera_keys:
        print("No camera frames found in observation; skipping image output.")
        return

    tiles = [make_labeled_tile(key, frame_to_bgr(robot, key, obs[key]), tile_width) for key in camera_keys]
    max_tile_height = max(tile.shape[0] for tile in tiles)
    padded_tiles = []
    for tile in tiles:
        if tile.shape[0] < max_tile_height:
            pad = np.zeros((max_tile_height - tile.shape[0], tile.shape[1], 3), dtype=np.uint8)
            tile = np.vstack([tile, pad])
        padded_tiles.append(tile)

    cols = max(1, cols)
    rows = math.ceil(len(padded_tiles) / cols)
    blank = np.zeros((max_tile_height, tile_width, 3), dtype=np.uint8)
    row_images = []
    for row_idx in range(rows):
        row_tiles = padded_tiles[row_idx * cols : (row_idx + 1) * cols]
        while len(row_tiles) < cols:
            row_tiles.append(blank.copy())
        row_images.append(np.hstack(row_tiles))

    grid = np.vstack(row_images)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
    print(f"Saved combined camera image to {output_path}")


def print_robot_summary(robot: RM65Robot, obs: dict[str, Any]) -> None:
    print(f"RM65 mode: {robot.config.mode}")
    print("Arm endpoints:")
    for side, arm_config in robot.arm_configs.items():
        print(f"  {side or 'single'}: {arm_config.ip}:{arm_config.port}")

    print("Action keys:")
    for key in sorted(robot.action_features):
        print(f"  {key}")

    print("Observation keys:")
    for key in sorted(obs):
        value = obs[key]
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")


def run_test(args: argparse.Namespace) -> None:
    cfg = load_robot_config(args.config_path)
    robot = RM65Robot(cfg)
    moved = False
    original_obs: dict[str, Any] | None = None

    try:
        print(f"Connecting using config: {args.config_path}")
        robot.connect()
        obs = robot.get_observation()
        original_obs = obs
        print_robot_summary(robot, obs)
        save_camera_grid(robot, obs, args.output, args.cols, args.tile_width)

        if args.move:
            sides = arm_sides(robot, args.sides)
            readable_sides = ["single" if side == "" else side for side in sides]
            print(f"Moving joint `{args.joint}` by {args.delta_deg} deg on side(s): {readable_sides}")
            action = action_for_joint_delta(robot, obs, sides, args.joint, args.delta_deg)
            robot.send_action(action, log_angles_info=True)
            moved = True
            time.sleep(args.wait_s)

            moved_obs = robot.get_observation()
            print("Observed moved joint positions:")
            for side in sides:
                key = f"{robot._prefix(side)}{args.joint}.pos"
                print(f"  {key}: {moved_obs[key]:.3f} deg")

            print("Returning to original joint positions.")
            robot.send_action(action_for_original_joints(robot, original_obs, sides), log_angles_info=True)
            moved = False
            time.sleep(args.wait_s)

            returned_obs = robot.get_observation()
            returned_output = args.output.with_name(f"{args.output.stem}_after_return{args.output.suffix}")
            save_camera_grid(robot, returned_obs, returned_output, args.cols, args.tile_width)

    finally:
        if robot.is_connected:
            if moved and original_obs is not None:
                try:
                    sides = arm_sides(robot, args.sides)
                    print("Best-effort return to original joint positions before disconnect.")
                    robot.send_action(action_for_original_joints(robot, original_obs, sides), log_angles_info=True)
                    time.sleep(args.wait_s)
                except Exception as exc:
                    print(f"Best-effort return failed: {exc}")
            robot.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test an RM65 single-arm or bimanual config.")
    parser.add_argument("--config-path", default="configs/rm65_wrist_front.yaml")
    parser.add_argument("--output", type=Path, default=Path("examples/rm65_config_test.jpg"))
    parser.add_argument("--cols", type=int, default=2, help="Number of columns in the combined camera image.")
    parser.add_argument("--tile-width", type=int, default=640, help="Width of each camera tile in pixels.")
    parser.add_argument("--move", action="store_true", help="Perform a small joint move and return.")
    parser.add_argument("--joint", default="joint_6", help="Joint name without side prefix.")
    parser.add_argument("--delta-deg", type=float, default=1.0, help="Small joint delta for --move.")
    parser.add_argument("--wait-s", type=float, default=1.0, help="Wait time after each movement command.")
    parser.add_argument(
        "--sides",
        default="all",
        help="For bimanual mode, use all, left, right, or left,right. Single mode uses all.",
    )
    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
