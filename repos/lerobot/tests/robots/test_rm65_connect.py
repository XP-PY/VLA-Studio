#!/usr/bin/env python

import argparse
import os
from pathlib import Path

from lerobot.robots.rm65 import RM65ArmConfig, RM65Robot, RM65RobotConfig

try:
    import pytest
except ModuleNotFoundError:
    pytest = None


def make_config(
    ip: str | None = None,
    port: int | None = None,
    sdk_path: str | Path | None = None,
    calibration_dir: str | Path | None = None,
) -> RM65RobotConfig:
    del sdk_path

    return RM65RobotConfig(
        mode="single",
        single_arm=RM65ArmConfig(
            ip=ip or os.getenv("RM65_IP", "192.168.1.18"),
            port=port or int(os.getenv("RM65_PORT", "8080")),
        ),
        calibration_dir=Path(
            calibration_dir or os.getenv("RM65_CALIBRATION_DIR", "/tmp/lerobot-rm65-calibration")
        ).expanduser(),
    )


def connect_and_read_once(cfg: RM65RobotConfig) -> dict:
    robot = RM65Robot(cfg)
    try:
        robot.connect()
        obs = robot.get_observation()
        print(f"Connected to RM65 at {robot.default_arm_config.ip}:{robot.default_arm_config.port}")
        print("Joint positions:")
        for name in robot.default_arm_config.joint_names:
            print(f"  {name}: {obs[f'{name}.pos']:.3f} deg")
        if robot.default_arm_config.use_gripper:
            print(f"Gripper position: {obs['gripper.pos']:.3f}")
        return obs
    finally:
        if robot.is_connected:
            robot.disconnect()


def test_rm65_connect_and_read_once():
    if pytest is None:
        raise RuntimeError("pytest is required to run this as a pytest test.")
    if os.getenv("RUN_RM65_HARDWARE_TEST") != "1":
        pytest.skip("Set RUN_RM65_HARDWARE_TEST=1 to run this hardware test.")

    obs = connect_and_read_once(make_config())
    for key in ["joint_1.pos", "joint_2.pos", "joint_3.pos", "joint_4.pos", "joint_5.pos", "joint_6.pos"]:
        assert key in obs
        assert isinstance(obs[key], float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect to a RealMan RM65 and read one observation.")
    parser.add_argument("--ip", default=None, help="Robot controller IP. Defaults to RM65_IP or 192.168.1.18.")
    parser.add_argument("--port", type=int, default=None, help="Robot controller port. Defaults to RM65_PORT or 8080.")
    parser.add_argument(
        "--sdk-path",
        default=None,
        help="Ignored. RM65 now uses the controller JSON/TCP protocol directly.",
    )
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help="LeRobot calibration directory. Defaults to RM65_CALIBRATION_DIR or /tmp/lerobot-rm65-calibration.",
    )
    args = parser.parse_args()
    connect_and_read_once(
        make_config(
            ip=args.ip,
            port=args.port,
            sdk_path=args.sdk_path,
            calibration_dir=args.calibration_dir,
        )
    )


if __name__ == "__main__":
    main()
