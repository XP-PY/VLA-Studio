#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig as _OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import (  # noqa: F401
    RealSenseCameraConfig as _RealSenseCameraConfig,
)

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
    """Configuration for one RM65 arm, or a bimanual pair of RM65 arms.

    In `single` mode, `single_arm` defines the arm and action/observation keys
    keep the original names, for example `joint_1.pos`.

    In `bimanual` mode, `left_arm` and `right_arm` define the two arms and
    action/observation keys are prefixed, for example `left_joint_1.pos` and
    `right_joint_1.pos`. Camera keys stay unprefixed.
    """

    mode: str = "single"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    single_arm: "RM65ArmConfig | None" = None
    left_arm: "RM65ArmConfig | None" = None
    right_arm: "RM65ArmConfig | None" = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode == "single":
            if self.single_arm is None:
                raise ValueError("`single_arm` must be provided when `mode` is 'single'.")
            if self.left_arm is not None or self.right_arm is not None:
                raise ValueError("`left_arm` and `right_arm` must be omitted when `mode` is 'single'.")
        elif self.mode == "bimanual":
            if self.left_arm is None or self.right_arm is None:
                raise ValueError("`left_arm` and `right_arm` must be provided when `mode` is 'bimanual'.")
            if self.single_arm is not None:
                raise ValueError("`single_arm` must be omitted when `mode` is 'bimanual'.")
        else:
            raise ValueError(f"Unsupported RM65 mode: {self.mode}")


@dataclass
class RM65ArmConfig:
    """Per-arm RM65 network and control settings."""

    ip: str
    port: int = 8080
    socket_timeout_s: float = 3.0

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
