#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig

RELEVANCE_MASK = "observation.relevance_mask"
RELEVANCE_LOSS_WEIGHT = "observation.relevance_loss_weight"


@PreTrainedConfig.register_subclass("action_flow_v2")
@dataclass
class ActionFlowV2Config(SmolVLAConfig):
    """Configuration for prompt-grounded flow matching with an auxiliary heatmap head."""

    # The V2 shared representation must be trainable for the relevance loss to
    # influence action generation.
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    load_vlm_weights: bool = True

    # Execute a short prefix of each predicted chunk by default.
    n_action_steps: int = 10

    # Auxiliary task-relevance objective.
    relevance_loss_weight: float = 0.1
    relevance_bce_weight: float = 1.0
    relevance_dice_weight: float = 1.0
    relevance_head_dim: int = 256
    relevance_loss_eps: float = 1e-6
    # (width, height), matching resize_imgs_with_padding ordering.
    relevance_target_size: tuple[int, int] = (8, 8)
    max_cameras: int = 8

    def __post_init__(self):
        super().__post_init__()

        if self.relevance_loss_weight < 0:
            raise ValueError("`relevance_loss_weight` must be non-negative.")
        if self.relevance_bce_weight < 0 or self.relevance_dice_weight < 0:
            raise ValueError("Relevance BCE and Dice weights must be non-negative.")
        if self.relevance_bce_weight == 0 and self.relevance_dice_weight == 0:
            raise ValueError("At least one relevance loss component must be enabled.")
        if self.relevance_head_dim <= 0:
            raise ValueError("`relevance_head_dim` must be positive.")
        if len(self.relevance_target_size) != 2 or any(size <= 0 for size in self.relevance_target_size):
            raise ValueError("`relevance_target_size` must contain two positive dimensions.")
        if self.max_cameras <= 0:
            raise ValueError("`max_cameras` must be positive.")

    def validate_features(self) -> None:
        super().validate_features()
        if not self.image_features:
            raise ValueError("Action Flow V2 requires at least one image feature.")
        if len(self.image_features) > self.max_cameras:
            raise ValueError(
                f"Configured {len(self.image_features)} cameras, but `max_cameras={self.max_cameras}`."
            )

    def set_dataset_feature_metadata(self, features: dict) -> None:
        """Validate auxiliary labels and exclude them from normalized model inputs."""

        missing = {RELEVANCE_MASK, RELEVANCE_LOSS_WEIGHT}.difference(features)
        if missing:
            raise ValueError(
                "Action Flow V2 requires merged segmentation features: "
                f"{sorted(missing)}"
            )

        camera_names = [key.rsplit(".", 1)[-1] for key in self.image_features]
        target_width, target_height = self.relevance_target_size
        expected_mask_shape = (len(camera_names), target_height, target_width)
        mask_shape = tuple(features[RELEVANCE_MASK]["shape"])
        if mask_shape != expected_mask_shape:
            raise ValueError(
                f"`{RELEVANCE_MASK}` has shape {mask_shape}, expected {expected_mask_shape}."
            )

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

        # These are training labels, not proprioceptive model inputs. Keeping
        # them out of input_features also prevents state normalization.
        if self.input_features is not None:
            self.input_features.pop(RELEVANCE_MASK, None)
            self.input_features.pop(RELEVANCE_LOSS_WEIGHT, None)
