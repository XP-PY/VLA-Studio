#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


RM65_JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "joint_1": (-177.96, 177.96),
    "joint_2": (-130.0, 130.0),
    "joint_3": (-135.0, 135.0),
    "joint_4": (-177.96, 177.96),
    "joint_5": (-128.0, 128.0),
    "joint_6": (-359.82, 359.82),
}


@RobotConfig.register_subclass("rm65")
@dataclass
class RM65RobotConfig(RobotConfig):
    """Configuration for a RealMan RM65 arm using the controller JSON/TCP protocol."""

    ip: str = "192.168.1.18"
    port: int = 8080
    socket_timeout_s: float = 3.0

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    joint_names: list[str] = field(
        default_factory=lambda: [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
    )
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: RM65_JOINT_LIMITS_DEG.copy()
    )
    max_relative_target: float | dict[str, float] | None = 10.0

    use_canfd: bool = True
    canfd_follow: bool = False
    canfd_expand: int = 1000
    canfd_trajectory_mode: int = 0
    canfd_smooth: int = 0

    movej_speed: int = 20
    movej_blend_radius: int = 0
    movej_connect: int = 0

    use_gripper: bool = False
    gripper_min: int = 1
    gripper_max: int = 1000
    gripper_rm_plus_index: int = 0
