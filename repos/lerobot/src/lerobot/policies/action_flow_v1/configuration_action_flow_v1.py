#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("action_flow_v1")
@dataclass
class ActionFlowV1Config(PreTrainedConfig):
    """ACT-style image/state policy trained with flow matching over action chunks.

    The policy keeps the standard LeRobot action interface: inference returns
    continuous robot actions. During training, however, it learns a velocity
    field that maps noisy action chunks back to dataset action chunks.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 10

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    # Transformer.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    dropout: float = 0.1

    # Flow matching.
    num_inference_steps: int = 10
    time_sampling_alpha: float = 1.5
    time_sampling_beta: float = 1.0

    # Training preset.
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 10.0

    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps are not handled yet. Got `n_obs_steps={self.n_obs_steps}`."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` must be <= `chunk_size`. Got {self.n_action_steps} and {self.chunk_size}."
            )
        if self.num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0. Got {self.num_inference_steps}.")
        if self.time_sampling_alpha <= 0 or self.time_sampling_beta <= 0:
            raise ValueError(
                "`time_sampling_alpha` and `time_sampling_beta` must be positive. "
                f"Got {self.time_sampling_alpha=} and {self.time_sampling_beta=}."
            )

    def validate_features(self) -> None:
        if not self.image_features and not self.robot_state_feature and not self.env_state_feature:
            raise ValueError("You must provide at least one image, robot-state, or environment-state input.")
        if self.action_feature is None:
            raise ValueError("`action` is required as an output feature.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
