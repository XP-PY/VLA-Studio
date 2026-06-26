#!/usr/bin/env python

import argparse
import os
import time
from pathlib import Path

from lerobot.robots.rm65 import RM65Robot, RM65RobotConfig

try:
    import pytest
except ModuleNotFoundError:
    pytest = None


def make_config(
    ip: str | None = None,
    port: int | None = None,
    sdk_path: str | Path | None = None,
    calibration_dir: str | Path | None = None,
    use_gripper: bool = True,
    max_relative_target: float = 5.0,
) -> RM65RobotConfig:
    del sdk_path

    return RM65RobotConfig(
        ip=ip or os.getenv("RM65_IP", "192.168.1.18"),
        port=port or int(os.getenv("RM65_PORT", "8080")),
        calibration_dir=Path(
            calibration_dir or os.getenv("RM65_CALIBRATION_DIR", "/tmp/lerobot-rm65-calibration")
        ).expanduser(),
        use_gripper=use_gripper,
        max_relative_target=max_relative_target,
    )


def _joint_action(obs: dict, joint_name: str, target_position: float) -> dict[str, float]:
    action = {
        key: value
        for key, value in obs.items()
        if key.endswith(".pos") and key.startswith("joint_")
    }
    action[f"{joint_name}.pos"] = target_position
    return action


def run_motion_and_gripper_test(
    cfg: RM65RobotConfig,
    joint_name: str,
    joint_delta_deg: float,
    gripper_open: int,
    gripper_close: int,
    wait_s: float,
    skip_motion: bool = False,
    skip_gripper: bool = False,
    restore_gripper_open: bool = True,
) -> None:
    robot = RM65Robot(cfg)
    original_obs = None
    moved_joint = False

    try:
        robot.connect()
        original_obs = robot.get_observation()
        original_joint = float(original_obs[f"{joint_name}.pos"])
        target_joint = original_joint + joint_delta_deg

        print(f"Connected to RM65 at {cfg.ip}:{cfg.port}")
        print(f"Initial {joint_name}: {original_joint:.3f} deg")

        if not skip_motion:
            print(f"Moving {joint_name} to {target_joint:.3f} deg")
            robot.send_action(_joint_action(original_obs, joint_name, target_joint))
            moved_joint = True
            time.sleep(wait_s)
            moved_obs = robot.get_observation()
            print(f"Observed {joint_name} after move: {moved_obs[f'{joint_name}.pos']:.3f} deg")

            print(f"Returning {joint_name} to {original_joint:.3f} deg")
            robot.send_action(_joint_action(moved_obs, joint_name, original_joint))
            moved_joint = False
            time.sleep(wait_s)
            returned_obs = robot.get_observation()
            print(f"Observed {joint_name} after return: {returned_obs[f'{joint_name}.pos']:.3f} deg")

        if not skip_gripper:
            print(f"Opening gripper to {gripper_open}")
            robot.send_action({"gripper.pos": float(gripper_open)})
            time.sleep(wait_s)
            print(f"Adapter gripper position after open: {robot._read_gripper_position():.1f}")

            print(f"Closing gripper to {gripper_close}")
            robot.send_action({"gripper.pos": float(gripper_close)})
            time.sleep(wait_s)
            print(f"Adapter gripper position after close: {robot._read_gripper_position():.1f}")

            if restore_gripper_open:
                print(f"Restoring gripper to open position {gripper_open}")
                robot.send_action({"gripper.pos": float(gripper_open)})
                time.sleep(wait_s)

    finally:
        if robot.is_connected:
            if moved_joint and original_obs is not None:
                try:
                    original_joint = float(original_obs[f"{joint_name}.pos"])
                    print(f"Best-effort return of {joint_name} to {original_joint:.3f} deg")
                    robot.send_action(_joint_action(original_obs, joint_name, original_joint))
                    time.sleep(wait_s)
                except Exception as e:
                    print(f"Failed to return {joint_name}: {e}")
            robot.disconnect()


def test_rm65_motion_and_gripper():
    if pytest is None:
        raise RuntimeError("pytest is required to run this as a pytest test.")
    if os.getenv("RUN_RM65_MOTION_TEST") != "1":
        pytest.skip("Set RUN_RM65_MOTION_TEST=1 to run this hardware motion test.")

    joint_delta = float(os.getenv("RM65_TEST_JOINT_DELTA_DEG", "2.0"))
    cfg = make_config(max_relative_target=abs(joint_delta) + 0.5)
    run_motion_and_gripper_test(
        cfg=cfg,
        joint_name=os.getenv("RM65_TEST_JOINT", "joint_6"),
        joint_delta_deg=joint_delta,
        gripper_open=int(os.getenv("RM65_TEST_GRIPPER_OPEN", "1000")),
        gripper_close=int(os.getenv("RM65_TEST_GRIPPER_CLOSE", "1")),
        wait_s=float(os.getenv("RM65_TEST_WAIT_S", "1.0")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Move an RM65 joint and command gripper open/close.")
    parser.add_argument("--yes", action="store_true", help="Required. Confirms you want to move hardware.")
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
    parser.add_argument("--joint", default="joint_6", choices=[f"joint_{i}" for i in range(1, 7)])
    parser.add_argument("--delta-deg", type=float, default=2.0, help="Relative joint motion in degrees.")
    parser.add_argument("--wait-s", type=float, default=1.0, help="Wait after each command.")
    parser.add_argument("--gripper-open", type=int, default=1000)
    parser.add_argument("--gripper-close", type=int, default=1)
    parser.add_argument("--skip-motion", action="store_true")
    parser.add_argument("--skip-gripper", action="store_true")
    parser.add_argument("--no-restore-gripper-open", action="store_true")
    args = parser.parse_args()

    if not args.yes:
        raise SystemExit("Refusing to move hardware without --yes.")

    cfg = make_config(
        ip=args.ip,
        port=args.port,
        sdk_path=args.sdk_path,
        calibration_dir=args.calibration_dir,
        use_gripper=not args.skip_gripper,
        max_relative_target=abs(args.delta_deg) + 0.5,
    )
    run_motion_and_gripper_test(
        cfg=cfg,
        joint_name=args.joint,
        joint_delta_deg=args.delta_deg,
        gripper_open=args.gripper_open,
        gripper_close=args.gripper_close,
        wait_s=args.wait_s,
        skip_motion=args.skip_motion,
        skip_gripper=args.skip_gripper,
        restore_gripper_open=not args.no_restore_gripper_open,
    )


if __name__ == "__main__":
    main()
