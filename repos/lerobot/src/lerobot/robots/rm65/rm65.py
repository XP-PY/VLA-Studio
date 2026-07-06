#!/usr/bin/env python

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_rm65 import RM65ArmConfig, RM65RobotConfig
from .raw_client import RM65RawClient

logger = logging.getLogger(__name__)


class RM65Robot(Robot):
    """RealMan RM65 robot adapter backed by the controller JSON/TCP protocol."""

    config_class = RM65RobotConfig
    name = "rm65"

    def __init__(self, config: RM65RobotConfig):
        super().__init__(config)
        self.config = config
        self.arm_configs = self._make_arm_configs(config)
        self.client: RM65RawClient | None = None
        self.clients: dict[str, RM65RawClient] = {}
        self.cameras = make_cameras_from_configs(config.cameras)

    @staticmethod
    def _make_arm_configs(config: RM65RobotConfig) -> dict[str, RM65ArmConfig]:
        if config.mode == "single":
            assert config.single_arm is not None
            return {"": config.single_arm}
        assert config.left_arm is not None and config.right_arm is not None
        return {"left": config.left_arm, "right": config.right_arm}

    @property
    def is_bimanual(self) -> bool:
        return "" not in self.arm_configs

    @property
    def default_arm_config(self) -> RM65ArmConfig:
        return self.arm_configs["right"] if self.is_bimanual else self.arm_configs[""]

    @staticmethod
    def _prefix(side: str) -> str:
        return f"{side}_" if side else ""

    def _client_for(self, side: str) -> RM65RawClient:
        client = self.clients.get(side)
        if client is None:
            raise RuntimeError(f"RM65 {side or 'single'} arm client is not connected")
        return client

    @property
    def _joint_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for side, arm_config in self.arm_configs.items():
            prefix = self._prefix(side)
            features.update({f"{prefix}{joint}.pos": float for joint in arm_config.joint_names})
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {
            **self._joint_features,
            **self._cameras_ft,
        }
        for side, arm_config in self.arm_configs.items():
            if arm_config.use_gripper:
                features[f"{self._prefix(side)}gripper.pos"] = float
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = dict(self._joint_features)
        for side, arm_config in self.arm_configs.items():
            if arm_config.use_gripper:
                features[f"{self._prefix(side)}gripper.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return (
            bool(self.clients)
            and all(client.is_connected for client in self.clients.values())
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        for side, arm_config in self.arm_configs.items():
            client = RM65RawClient(arm_config.ip, arm_config.port, arm_config.socket_timeout_s)
            client.connect()
            self.clients[side] = client

        self.client = self.clients["right"] if self.is_bimanual else self.clients[""]

        for cam in self.cameras.values():
            cam.connect()

        arms = ", ".join(
            f"{side or 'single'}={arm_config.ip}:{arm_config.port}"
            for side, arm_config in self.arm_configs.items()
        )
        logger.info("%s connected to %s.", self, arms)

    def calibrate(self) -> None:
        logger.info("RM65 raw JSON control does not require a LeRobot calibration step.")

    def configure(self) -> None:
        pass

    def _read_state(self, side: str = "") -> dict[str, Any]:
        return self._client_for(side).get_current_arm_state()

    def _current_joint_positions(self, side: str = "") -> dict[str, float]:
        arm_config = self.arm_configs[side]
        state = self._read_state(side)
        joints = state.get("joint")
        if joints is None:
            raise RuntimeError(f"RM65 state did not include `joint`: {state}")
        current = {
            joint: float(joints[i]) / 1000.0 for i, joint in enumerate(arm_config.joint_names)
        }
        return current

    def _read_gripper_position(self, side: str = "") -> float:
        arm_config = self.arm_configs[side]
        state = self._client_for(side).get_rm_plus_state_info()
        positions = state.get("pos")
        if positions is None:
            raise RuntimeError(f"RM Plus state did not include `pos`: {state}")

        index = arm_config.gripper_rm_plus_index
        if index < 0 or index >= len(positions):
            raise RuntimeError(f"RM Plus gripper index {index} is out of range for pos={positions}")

        return float(positions[index])

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        obs: dict[str, Any] = {}
        for side, arm_config in self.arm_configs.items():
            state = self._read_state(side)
            joints = state.get("joint")
            if joints is None:
                raise RuntimeError(f"RM65 state did not include `joint`: {state}")

            prefix = self._prefix(side)
            obs.update(
                {
                    f"{prefix}{joint}.pos": float(joints[i]) / 1000.0
                    for i, joint in enumerate(arm_config.joint_names)
                }
            )

            if arm_config.use_gripper:
                obs[f"{prefix}gripper.pos"] = self._read_gripper_position(side)

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()

        logger.debug("%s get_observation took %.1fms", self, (time.perf_counter() - start) * 1e3)
        return obs

    def _target_joint_positions(
        self, action: RobotAction, side: str = "", arm_config: RM65ArmConfig | None = None
    ) -> dict[str, float]:
        if arm_config is None:
            arm_config = self.arm_configs[side]
        goal_pos = {
            key.removesuffix(".pos"): float(value)
            for key, value in action.items()
            if key.endswith(".pos") and key.removesuffix(".pos") in arm_config.joint_names
        }

        current = self._current_joint_positions(side)
        for joint in arm_config.joint_names:
            goal_pos.setdefault(joint, current[joint])

        for joint, position in goal_pos.items():
            if joint in arm_config.joint_limits:
                min_limit, max_limit = arm_config.joint_limits[joint]
                goal_pos[joint] = max(min_limit, min(max_limit, position))

        if arm_config.max_relative_target is not None:
            goal_present_pos = {joint: (goal, current[joint]) for joint, goal in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, arm_config.max_relative_target)

        return goal_pos

    @check_if_not_connected
    def _send_arm_action(
        self,
        side: str,
        action: RobotAction,
        arm_config: RM65ArmConfig,
        log_angles_info: bool,
    ) -> RobotAction:
        client = self._client_for(side)
        goal_pos = self._target_joint_positions(action, side, arm_config)
        joint_values = [goal_pos[joint] for joint in arm_config.joint_names]
        if log_angles_info:
            print(
                "RM65 %s sent joints=%s gripper=%s"
                % (
                    side or "single",
                    [round(v, 3) for v in joint_values],
                    round(float(action["gripper.pos"]), 3) if "gripper.pos" in action else None,
                )
            )

        if arm_config.use_canfd:
            client.movej_canfd(
                joint_values,
                arm_config.canfd_follow,
                expand=arm_config.canfd_expand,
                trajectory_mode=arm_config.canfd_trajectory_mode,
                radio=arm_config.canfd_smooth,
            )
        else:
            client.movej(
                joint_values,
                arm_config.movej_speed,
                arm_config.movej_blend_radius,
                arm_config.movej_connect,
            )

        sent_action: dict[str, Any] = {f"{joint}.pos": pos for joint, pos in goal_pos.items()}

        if arm_config.use_gripper and "gripper.pos" in action:
            gripper_pos = int(
                max(arm_config.gripper_min, min(arm_config.gripper_max, action["gripper.pos"]))
            )
            client.set_gripper_position(gripper_pos, block=False)
            sent_action["gripper.pos"] = float(gripper_pos)

        return sent_action

    @check_if_not_connected
    def send_action(self, action: RobotAction, log_angles_info: bool = True) -> RobotAction:
        if not self.is_bimanual:
            return self._send_arm_action("", action, self.arm_configs[""], log_angles_info)

        sent_action: dict[str, Any] = {}
        for side, arm_config in self.arm_configs.items():
            prefix = self._prefix(side)
            side_action = {
                key.removeprefix(prefix): value
                for key, value in action.items()
                if key.startswith(prefix)
            }
            if not side_action:
                continue
            side_sent = self._send_arm_action(side, side_action, arm_config, log_angles_info)
            sent_action.update({f"{prefix}{key}": value for key, value in side_sent.items()})
        return sent_action

    @check_if_not_connected
    def disconnect(self) -> None:
        for side, client in self.clients.items():
            try:
                client.set_arm_stop()
            except Exception as e:
                logger.debug("Ignoring RM65 %s stop failure during disconnect: %s", side or "single", e)

        for cam in self.cameras.values():
            cam.disconnect()

        for client in self.clients.values():
            client.close()
        self.clients.clear()
        self.client = None
        logger.info("%s disconnected.", self)
