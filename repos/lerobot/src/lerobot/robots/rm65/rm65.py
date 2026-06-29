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
from .config_rm65 import RM65RobotConfig
from .raw_client import RM65RawClient

logger = logging.getLogger(__name__)


class RM65Robot(Robot):
    """RealMan RM65 robot adapter backed by the controller JSON/TCP protocol."""

    config_class = RM65RobotConfig
    name = "rm65"

    def __init__(self, config: RM65RobotConfig):
        super().__init__(config)
        self.config = config
        self.client: RM65RawClient | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _joint_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.config.joint_names}

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
        if self.config.use_gripper:
            features["gripper.pos"] = float
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = dict(self._joint_features)
        if self.config.use_gripper:
            features["gripper.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self.client is not None and self.client.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.client = RM65RawClient(self.config.ip, self.config.port, self.config.socket_timeout_s)
        self.client.connect()

        for cam in self.cameras.values():
            cam.connect()

        logger.info("%s connected to %s:%s.", self, self.config.ip, self.config.port)

    def calibrate(self) -> None:
        logger.info("RM65 raw JSON control does not require a LeRobot calibration step.")

    def configure(self) -> None:
        pass

    def _read_state(self) -> dict[str, Any]:
        assert self.client is not None
        return self.client.get_current_arm_state()

    def _current_joint_positions(self) -> dict[str, float]:
        state = self._read_state()
        joints = state.get("joint")
        if joints is None:
            raise RuntimeError(f"RM65 state did not include `joint`: {state}")
        current = {
            joint: float(joints[i]) / 1000.0 for i, joint in enumerate(self.config.joint_names)
        }
        return current

    def _read_gripper_position(self) -> float:
        assert self.client is not None
        state = self.client.get_rm_plus_state_info()
        positions = state.get("pos")
        if positions is None:
            raise RuntimeError(f"RM Plus state did not include `pos`: {state}")

        index = self.config.gripper_rm_plus_index
        if index < 0 or index >= len(positions):
            raise RuntimeError(f"RM Plus gripper index {index} is out of range for pos={positions}")

        return float(positions[index])

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        state = self._read_state()
        joints = state.get("joint")
        if joints is None:
            raise RuntimeError(f"RM65 state did not include `joint`: {state}")

        obs: dict[str, Any] = {
            f"{joint}.pos": float(joints[i]) / 1000.0
            for i, joint in enumerate(self.config.joint_names)
        }

        if self.config.use_gripper:
            obs["gripper.pos"] = self._read_gripper_position()

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()

        logger.debug("%s get_observation took %.1fms", self, (time.perf_counter() - start) * 1e3)
        return obs

    def _target_joint_positions(self, action: RobotAction) -> dict[str, float]:
        goal_pos = {
            key.removesuffix(".pos"): float(value)
            for key, value in action.items()
            if key.endswith(".pos") and key.removesuffix(".pos") in self.config.joint_names
        }

        current = self._current_joint_positions()
        for joint in self.config.joint_names:
            goal_pos.setdefault(joint, current[joint])

        for joint, position in goal_pos.items():
            if joint in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[joint]
                goal_pos[joint] = max(min_limit, min(max_limit, position))

        if self.config.max_relative_target is not None:
            goal_present_pos = {joint: (goal, current[joint]) for joint, goal in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        return goal_pos

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        assert self.client is not None
        goal_pos = self._target_joint_positions(action)
        joint_values = [goal_pos[joint] for joint in self.config.joint_names]

        if self.config.use_canfd:
            self.client.movej_canfd(
                joint_values,
                self.config.canfd_follow,
                expand=self.config.canfd_expand,
                trajectory_mode=self.config.canfd_trajectory_mode,
                radio=self.config.canfd_smooth,
            )
        else:
            self.client.movej(
                joint_values,
                self.config.movej_speed,
                self.config.movej_blend_radius,
                self.config.movej_connect,
            )

        sent_action: dict[str, Any] = {f"{joint}.pos": pos for joint, pos in goal_pos.items()}

        if self.config.use_gripper and "gripper.pos" in action:
            gripper_pos = int(
                max(self.config.gripper_min, min(self.config.gripper_max, action["gripper.pos"]))
            )
            self.client.set_gripper_position(gripper_pos, block=False)
            sent_action["gripper.pos"] = float(gripper_pos)

        return sent_action

    @check_if_not_connected
    def disconnect(self) -> None:
        assert self.client is not None
        try:
            self.client.set_arm_stop()
        except Exception as e:
            logger.debug("Ignoring RM65 stop failure during disconnect: %s", e)

        for cam in self.cameras.values():
            cam.disconnect()

        self.client.close()
        self.client = None
        logger.info("%s disconnected.", self)
