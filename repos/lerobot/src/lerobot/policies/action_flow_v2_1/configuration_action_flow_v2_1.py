#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig

RELEVANCE_MASK = "observation.relevance_mask"
RELEVANCE_LOSS_WEIGHT = "observation.relevance_loss_weight"


@PreTrainedConfig.register_subclass("action_flow_v2_1")
@dataclass
class ActionFlowV21Config(SmolVLAConfig):
    """Configuration for dense prompt-grounded segmentation and action flow."""

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    load_vlm_weights: bool = True

    n_action_steps: int = 10

    relevance_loss_weight: float = 0.1
    relevance_bce_weight: float = 1.0
    relevance_dice_weight: float = 1.0
    relevance_loss_eps: float = 1e-6
    # One learned 2x stage per entry: 8x8 -> 512x512 by default.
    relevance_decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16, 8)
    # (width, height), matching resize_imgs_with_padding ordering.
    relevance_target_size: tuple[int, int] = (512, 512)
    max_cameras: int = 8

    def __post_init__(self):
        super().__post_init__()

        if self.relevance_loss_weight < 0:
            raise ValueError("`relevance_loss_weight` must be non-negative.")
        if self.relevance_bce_weight < 0 or self.relevance_dice_weight < 0:
            raise ValueError("Relevance BCE and Dice weights must be non-negative.")
        if self.relevance_bce_weight == 0 and self.relevance_dice_weight == 0:
            raise ValueError("At least one relevance loss component must be enabled.")
        if self.relevance_loss_eps <= 0:
            raise ValueError("`relevance_loss_eps` must be positive.")
        if not self.relevance_decoder_channels or any(
            channels <= 0 for channels in self.relevance_decoder_channels
        ):
            raise ValueError("`relevance_decoder_channels` must contain positive dimensions.")
        if len(self.relevance_target_size) != 2 or any(
            size <= 0 for size in self.relevance_target_size
        ):
            raise ValueError("`relevance_target_size` must contain two positive dimensions.")
        if self.resize_imgs_with_padding is None:
            raise ValueError("Action Flow V2.1 requires `resize_imgs_with_padding`.")
        if tuple(self.relevance_target_size) != tuple(self.resize_imgs_with_padding):
            raise ValueError(
                "`relevance_target_size` must match `resize_imgs_with_padding` so dense "
                "segmentation and image pixels stay aligned."
            )
        if self.max_cameras <= 0:
            raise ValueError("`max_cameras` must be positive.")

    def validate_features(self) -> None:
        super().validate_features()
        if not self.image_features:
            raise ValueError("Action Flow V2.1 requires at least one image feature.")
        if len(self.image_features) > self.max_cameras:
            raise ValueError(
                f"Configured {len(self.image_features)} cameras, but "
                f"`max_cameras={self.max_cameras}`."
            )

    def set_dataset_feature_metadata(self, features: dict) -> None:
        """Validate dense labels and exclude them from normalized model inputs."""

        missing = {RELEVANCE_MASK, RELEVANCE_LOSS_WEIGHT}.difference(features)
        if missing:
            raise ValueError(
                "Action Flow V2.1 requires merged segmentation features: "
                f"{sorted(missing)}"
            )

        camera_names = [key.rsplit(".", 1)[-1] for key in self.image_features]
        mask_shape = tuple(features[RELEVANCE_MASK]["shape"])
        if len(mask_shape) != 3 or mask_shape[0] != len(camera_names):
            raise ValueError(
                f"`{RELEVANCE_MASK}` must have shape "
                f"({len(camera_names)}, height, width), got {mask_shape}."
            )
        if any(size <= 0 for size in mask_shape[1:]):
            raise ValueError(f"`{RELEVANCE_MASK}` has invalid spatial shape {mask_shape[1:]}.")

        weight_shape = tuple(features[RELEVANCE_LOSS_WEIGHT]["shape"])
        if weight_shape != (len(camera_names),):
            raise ValueError(
                f"`{RELEVANCE_LOSS_WEIGHT}` has shape {weight_shape}, "
                f"expected {(len(camera_names),)}."
            )
        saved_camera_names = features[RELEVANCE_LOSS_WEIGHT].get("names")
        if saved_camera_names != camera_names:
            raise ValueError(
                "Segmentation camera order does not match policy image order: "
                f"{saved_camera_names} != {camera_names}."
            )

        if self.input_features is not None:
            self.input_features.pop(RELEVANCE_MASK, None)
            self.input_features.pop(RELEVANCE_LOSS_WEIGHT, None)
