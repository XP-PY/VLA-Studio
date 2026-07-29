#!/usr/bin/env python

"""Action Flow V2.1: action flow with prompt-conditioned dense segmentation."""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import require_package

from ..act.modeling_act import ACTTemporalEnsembler
from ..action_flow_v2.modeling_action_flow_v2 import ActionFlowV2Model, ActionFlowV2Policy
from ..pretrained import PreTrainedPolicy
from ..smolvla.modeling_smolvla import VLAFlowMatching
from .configuration_action_flow_v2_1 import (
    RELEVANCE_LOSS_WEIGHT,
    RELEVANCE_MASK,
    ActionFlowV21Config,
)


def resize_relevance_masks_with_pad(
    masks: Tensor,
    *,
    target_width: int,
    target_height: int,
) -> Tensor:
    """Resize binary masks with the same geometry used for policy images."""

    if masks.ndim != 4:
        raise ValueError(f"Expected masks shaped (B, C, H, W), got {tuple(masks.shape)}.")

    current_height, current_width = masks.shape[-2:]
    if (current_height, current_width) == (target_height, target_width):
        return (masks > 0).to(dtype=torch.float32)

    ratio = max(current_width / target_width, current_height / target_height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    flattened = masks.flatten(0, 1).unsqueeze(1).to(dtype=torch.float32)
    resized = F.interpolate(
        flattened,
        size=(resized_height, resized_width),
        mode="nearest",
    )
    pad_height = max(0, target_height - resized_height)
    pad_width = max(0, target_width - resized_width)
    resized = F.pad(resized, (pad_width, 0, pad_height, 0), value=0)
    return (resized[:, 0] > 0).reshape(
        masks.shape[0],
        masks.shape[1],
        target_height,
        target_width,
    ).to(dtype=torch.float32)


def prepare_relevance_targets(
    targets: Tensor,
    weights: Tensor,
    *,
    target_width: int,
    target_height: int,
) -> tuple[Tensor, Tensor]:
    """Remove LeRobot's time axis and align dense masks with prepared images."""

    if targets.ndim == 5 and targets.shape[1] == 1:
        targets = targets[:, 0]
    if weights.ndim == 3 and weights.shape[1] == 1:
        weights = weights[:, 0]
    if targets.ndim != 4:
        raise ValueError(f"Expected relevance targets shaped (B, C, H, W), got {targets.shape}.")
    if weights.shape != targets.shape[:2]:
        raise ValueError(
            f"Expected relevance weights shaped {tuple(targets.shape[:2])}, "
            f"got {tuple(weights.shape)}."
        )

    targets = resize_relevance_masks_with_pad(
        targets,
        target_width=target_width,
        target_height=target_height,
    )
    return targets, weights.to(dtype=torch.float32).clamp_min(0)


def compute_dense_relevance_loss(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    eps: float = 1e-6,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute camera-weighted BCE and Dice losses without spatial resampling."""

    if logits.ndim != 4 or targets.ndim != 4:
        raise ValueError(
            f"`logits` and `targets` must be 4D, got {tuple(logits.shape)} "
            f"and {tuple(targets.shape)}."
        )
    if logits.shape != targets.shape:
        raise ValueError(
            "Dense relevance logits and targets must have identical shapes, got "
            f"{tuple(logits.shape)} and {tuple(targets.shape)}."
        )
    if weights.shape != logits.shape[:2]:
        raise ValueError(
            f"`weights` must have shape {tuple(logits.shape[:2])}, got {tuple(weights.shape)}."
        )

    targets = targets.to(device=logits.device, dtype=logits.dtype)
    weights = weights.to(device=logits.device, dtype=logits.dtype)
    bce_per_camera = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(
        dim=(-2, -1)
    )
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum(dim=(-2, -1))
    denominator = probabilities.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    dice_per_camera = 1 - (2 * intersection + eps) / (denominator + eps)

    combined_per_camera = bce_weight * bce_per_camera + dice_weight * dice_per_camera
    weight_sum = weights.sum(dim=1)
    per_sample_loss = (combined_per_camera * weights).sum(dim=1) / weight_sum.clamp_min(eps)
    per_sample_loss = torch.where(weight_sum > 0, per_sample_loss, per_sample_loss * 0)

    valid_weight = weights.sum().clamp_min(eps)
    metrics = {
        "relevance_bce_loss": (bce_per_camera * weights).sum() / valid_weight,
        "relevance_dice_loss": (dice_per_camera * weights).sum() / valid_weight,
        "relevance_valid_camera_fraction": (weights > 0).float().mean(),
    }
    return per_sample_loss, metrics


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DenseRelevanceDecoder(nn.Module):
    """Decode contextualized image tokens into a learned full-resolution mask."""

    def __init__(self, hidden_size: int, channels: tuple[int, ...]):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, channels[0])

        blocks = []
        input_channels = channels[0]
        for output_channels in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(_group_count(output_channels), output_channels),
                    nn.GELU(),
                    nn.Conv2d(
                        output_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(output_channels), output_channels),
                    nn.GELU(),
                )
            )
            input_channels = output_channels
        self.blocks = nn.ModuleList(blocks)
        self.output_projection = nn.Conv2d(input_channels, 1, kernel_size=1)

    def forward(
        self,
        spatial_tokens: Tensor,
        *,
        grid_height: int,
        grid_width: int,
        target_height: int,
        target_width: int,
    ) -> Tensor:
        """Return dense logits with shape `(batch_size, target_height, target_width)`."""

        if spatial_tokens.shape[1] != grid_height * grid_width:
            raise ValueError(
                f"Cannot reshape {spatial_tokens.shape[1]} tokens to "
                f"{grid_height}x{grid_width}."
            )
        scale = 2 ** len(self.blocks)
        if (grid_height * scale, grid_width * scale) != (target_height, target_width):
            raise ValueError(
                f"{len(self.blocks)} decoder stages map {grid_height}x{grid_width} to "
                f"{grid_height * scale}x{grid_width * scale}, not "
                f"{target_height}x{target_width}."
            )

        features = self.input_projection(self.input_norm(spatial_tokens))
        features = features.transpose(1, 2).reshape(
            spatial_tokens.shape[0],
            features.shape[-1],
            grid_height,
            grid_width,
        )
        for block in self.blocks:
            features = F.interpolate(
                features,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
            features = block(features)
        return self.output_projection(features)[:, 0]


class ActionFlowV21Model(ActionFlowV2Model):
    """Action Flow V2 representation with a learned dense relevance decoder."""

    def __init__(self, config: ActionFlowV21Config, rtc_processor=None):
        VLAFlowMatching.__init__(self, config, rtc_processor=rtc_processor)

        hidden_size = self.vlm_with_expert.config.text_config.hidden_size
        self.camera_embeddings = nn.Embedding(config.max_cameras, hidden_size)
        nn.init.zeros_(self.camera_embeddings.weight)
        self.relevance_head = DenseRelevanceDecoder(
            hidden_size,
            config.relevance_decoder_channels,
        )
        self._inference_image_layout: list[tuple[slice, tuple[int, int]]] | None = None
        self.latest_relevance_heatmaps: Tensor | None = None

    def _predict_relevance(
        self,
        prefix_output: Tensor,
        image_layout: list[tuple[slice, tuple[int, int]]],
    ) -> Tensor:
        target_width, target_height = self.config.relevance_target_size
        head_dtype = next(self.relevance_head.parameters()).dtype
        camera_logits = []
        for token_slice, (grid_height, grid_width) in image_layout:
            camera_logits.append(
                self.relevance_head(
                    prefix_output[:, token_slice].to(dtype=head_dtype),
                    grid_height=grid_height,
                    grid_width=grid_width,
                    target_height=target_height,
                    target_width=target_width,
                )
            )
        return torch.stack(camera_logits, dim=1)


class ActionFlowV21Policy(ActionFlowV2Policy):
    """LeRobot policy wrapper for Action Flow V2.1."""

    config_class = ActionFlowV21Config
    name = "action_flow_v2_1"

    def __init__(self, config: ActionFlowV21Config, **kwargs):
        require_package("transformers", extra="smolvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = ActionFlowV21Model(config, rtc_processor=self.rtc_processor)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff,
                config.chunk_size,
            )
        self.reset()

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute flow loss and full-resolution BCE plus Dice supervision."""

        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction {reduction!r}.")
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        action_is_pad = batch.get("action_is_pad")
        flow_losses, relevance_logits = self.model.forward(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            actions,
            noise,
            time,
        )

        original_action_dim = self.config.action_feature.shape[0]
        flow_losses = flow_losses[:, :, :original_action_dim]
        if action_is_pad is not None:
            flow_losses = flow_losses * (~action_is_pad).unsqueeze(-1)
            valid_action_count = (
                (~action_is_pad).sum(dim=1) * flow_losses.shape[-1]
            ).clamp_min(1)
            flow_loss_per_sample = flow_losses.sum(dim=(1, 2)) / valid_action_count
        else:
            flow_loss_per_sample = flow_losses.mean(dim=(1, 2))

        if RELEVANCE_MASK not in batch or RELEVANCE_LOSS_WEIGHT not in batch:
            raise KeyError(
                "Action Flow V2.1 training requires merged dataset features "
                f"`{RELEVANCE_MASK}` and `{RELEVANCE_LOSS_WEIGHT}`."
            )
        target_width, target_height = self.config.relevance_target_size
        relevance_targets, relevance_weights = prepare_relevance_targets(
            batch[RELEVANCE_MASK],
            batch[RELEVANCE_LOSS_WEIGHT],
            target_width=target_width,
            target_height=target_height,
        )
        relevance_loss_per_sample, relevance_metrics = compute_dense_relevance_loss(
            relevance_logits,
            relevance_targets,
            relevance_weights,
            bce_weight=self.config.relevance_bce_weight,
            dice_weight=self.config.relevance_dice_weight,
            eps=self.config.relevance_loss_eps,
        )
        total_loss_per_sample = (
            flow_loss_per_sample + self.config.relevance_loss_weight * relevance_loss_per_sample
        )

        output_dict = {
            "loss": total_loss_per_sample.mean().item(),
            "flow_loss": flow_loss_per_sample.mean().item(),
            "relevance_loss": relevance_loss_per_sample.mean().item(),
            **{name: value.item() for name, value in relevance_metrics.items()},
        }
        if reduction == "none":
            return total_loss_per_sample, output_dict
        return total_loss_per_sample.mean(), output_dict

    def _get_default_peft_targets(self) -> dict[str, Any]:
        targets = super()._get_default_peft_targets()
        targets["modules_to_save"] = ["model.camera_embeddings", "model.relevance_head"]
        return targets
