#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Action Flow V2: prompt-grounded flow matching with auxiliary relevance masks."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import require_package

from ..act.modeling_act import ACTTemporalEnsembler
from ..pretrained import PreTrainedPolicy
from ..smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    make_att_2d_masks,
    pad_tensor,
)
from .configuration_action_flow_v2 import (
    RELEVANCE_LOSS_WEIGHT,
    RELEVANCE_MASK,
    ActionFlowV2Config,
)


def compute_relevance_loss(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    eps: float = 1e-6,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute weighted per-sample BCE and Dice losses for camera heatmaps.

    Args:
        logits: Predicted patch logits with shape
            `(batch_size, num_cameras, patch_height, patch_width)`.
        targets: Binary masks with shape
            `(batch_size, num_cameras, target_height, target_width)`, or with
            LeRobot's singleton observation axis,
            `(batch_size, 1, num_cameras, target_height, target_width)`.
        weights: Annotation-validity weights with shape
            `(batch_size, num_cameras)` or `(batch_size, 1, num_cameras)`.
            Zero excludes a camera frame.

    Returns:
        Per-sample relevance loss with shape `(batch_size,)` and scalar metrics
        for the BCE, Dice, and valid-camera fraction.
    """
    # LeRobot applies observation_delta_indices=[0] to every observation
    # feature, so dataset labels arrive with a singleton observation-time axis.
    if targets.ndim == 5 and targets.shape[1] == 1:
        targets = targets[:, 0]
    if weights.ndim == 3 and weights.shape[1] == 1:
        weights = weights[:, 0]

    if logits.ndim != 4 or targets.ndim != 4:
        raise ValueError(
            f"`logits` and `targets` must be 4D, got {tuple(logits.shape)} and {tuple(targets.shape)}."
        )
    if logits.shape[:2] != targets.shape[:2]:
        raise ValueError(
            "Relevance logits and targets have different batch/camera dimensions: "
            f"{tuple(logits.shape[:2])} != {tuple(targets.shape[:2])}."
        )
    if weights.shape != logits.shape[:2]:
        raise ValueError(
            f"`weights` must have shape {tuple(logits.shape[:2])}, got {tuple(weights.shape)}."
        )

    targets = targets.to(device=logits.device, dtype=logits.dtype)
    weights = weights.to(device=logits.device, dtype=logits.dtype).clamp_min(0)
    if targets.shape[-2:] != logits.shape[-2:]:
        if all(
            target_size >= logit_size
            for target_size, logit_size in zip(targets.shape[-2:], logits.shape[-2:], strict=True)
        ):
            targets = F.adaptive_max_pool2d(targets, output_size=logits.shape[-2:])
        else:
            targets = F.interpolate(targets, size=logits.shape[-2:], mode="nearest")

    bce_per_camera = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(
        dim=(-2, -1)
    )
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum(dim=(-2, -1))
    denominator = probabilities.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    dice_per_camera = 1 - (2 * intersection + eps) / (denominator + eps)

    combined_per_camera = bce_weight * bce_per_camera + dice_weight * dice_per_camera
    weight_sum = weights.sum(dim=1)
    normalizer = weight_sum.clamp_min(eps)
    per_sample_loss = (combined_per_camera * weights).sum(dim=1) / normalizer
    per_sample_loss = torch.where(weight_sum > 0, per_sample_loss, per_sample_loss * 0)

    valid_weight = weights.sum().clamp_min(eps)
    metrics = {
        "relevance_bce_loss": (bce_per_camera * weights).sum() / valid_weight,
        "relevance_dice_loss": (dice_per_camera * weights).sum() / valid_weight,
        "relevance_valid_camera_fraction": (weights > 0).float().mean(),
    }
    return per_sample_loss, metrics


class ActionFlowV2Model(VLAFlowMatching):
    """SmolVLM/action-expert flow model with a prompt-conditioned heatmap head."""

    def __init__(self, config: ActionFlowV2Config, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)

        hidden_size = self.vlm_with_expert.config.text_config.hidden_size
        self.camera_embeddings = nn.Embedding(config.max_cameras, hidden_size)
        nn.init.zeros_(self.camera_embeddings.weight)
        self.relevance_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, config.relevance_head_dim),
            nn.GELU(),
            nn.Linear(config.relevance_head_dim, 1),
        )
        self._inference_image_layout: list[tuple[slice, tuple[int, int]]] | None = None
        self.latest_relevance_heatmaps: Tensor | None = None

    def _embed_prefix_with_layout(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, list[tuple[slice, tuple[int, int]]]]:
        """Embed the multimodal prefix and retain each camera's spatial-token slice."""
        if len(images) > self.config.max_cameras:
            raise ValueError(
                f"Received {len(images)} cameras, but `max_cameras={self.config.max_cameras}`."
            )

        embeddings = []
        pad_masks = []
        attention_groups: list[int] = []
        image_layout: list[tuple[slice, tuple[int, int]]] = []
        token_offset = 0

        for camera_index, (image, image_mask) in enumerate(zip(images, img_masks, strict=True)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(image.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0],
                    dtype=torch.bool,
                    device=image_start_token.device,
                )
                embeddings.append(image_start_token)
                pad_masks.append(image_start_mask)
                attention_groups += [0] * image_start_token.shape[1]
                token_offset += image_start_token.shape[1]

            image_embedding = self.vlm_with_expert.embed_image(image)
            embedding_dim = image_embedding.shape[-1]
            image_embedding = image_embedding * math.sqrt(embedding_dim)
            camera_ids = torch.full(
                (image_embedding.shape[0], image_embedding.shape[1]),
                camera_index,
                dtype=torch.long,
                device=image_embedding.device,
            )
            image_embedding = image_embedding + self.camera_embeddings(camera_ids).to(
                dtype=image_embedding.dtype
            )

            grid_height, grid_width = self._image_token_grid(image)
            if grid_height * grid_width != image_embedding.shape[1]:
                raise ValueError(
                    "The SmolVLM connector output cannot be mapped to a spatial heatmap: "
                    f"expected {grid_height}x{grid_width} tokens from image shape "
                    f"{tuple(image.shape[-2:])}, got {image_embedding.shape[1]}."
                )

            batch_size, num_image_tokens = image_embedding.shape[:2]
            expanded_image_mask = image_mask[:, None].expand(batch_size, num_image_tokens)
            image_layout.append(
                (slice(token_offset, token_offset + num_image_tokens), (grid_height, grid_width))
            )
            token_offset += num_image_tokens
            embeddings.append(image_embedding)
            pad_masks.append(expanded_image_mask)
            attention_groups += [0] * num_image_tokens

            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(image.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0],
                    dtype=torch.bool,
                    device=image_end_token.device,
                )
                embeddings.append(image_end_token)
                pad_masks.append(image_end_mask)
                attention_groups += [0] * image_end_token.shape[1]
                token_offset += image_end_token.shape[1]

        language_embedding = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        language_embedding = language_embedding * math.sqrt(language_embedding.shape[-1])
        embeddings.append(language_embedding)
        pad_masks.append(lang_masks)
        attention_groups += [0] * language_embedding.shape[1]

        state_embedding = self.state_proj(state)
        if state_embedding.ndim == 2:
            state_embedding = state_embedding[:, None, :]
        embeddings.append(state_embedding)
        state_mask = torch.ones(
            state_embedding.shape[:2],
            dtype=torch.bool,
            device=state_embedding.device,
        )
        pad_masks.append(state_mask)
        attention_groups += [1] * state_embedding.shape[1]

        prefix_embeddings = torch.cat(embeddings, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_attention_groups = torch.tensor(
            attention_groups,
            dtype=torch.bool,
            device=prefix_pad_masks.device,
        )[None, :]

        if prefix_pad_masks.shape[1] < self.prefix_length:
            prefix_embeddings = pad_tensor(prefix_embeddings, self.prefix_length, pad_value=0)
            prefix_pad_masks = pad_tensor(prefix_pad_masks, self.prefix_length, pad_value=0)
            prefix_attention_groups = pad_tensor(
                prefix_attention_groups,
                self.prefix_length,
                pad_value=0,
            )

        prefix_attention_groups = prefix_attention_groups.expand(state.shape[0], -1)
        return prefix_embeddings, prefix_pad_masks, prefix_attention_groups, image_layout

    def _image_token_grid(self, image: Tensor) -> tuple[int, int]:
        vision_config = self.vlm_with_expert.config.vision_config
        patch_size = int(vision_config.patch_size)
        scale_factor = int(getattr(self.vlm_with_expert.config, "scale_factor", 1))
        divisor = patch_size * scale_factor
        image_height, image_width = image.shape[-2:]
        if image_height % divisor or image_width % divisor:
            raise ValueError(
                "Prepared image dimensions must be divisible by "
                f"patch_size * scale_factor ({divisor}), got {image_height}x{image_width}."
            )
        return image_height // divisor, image_width // divisor

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed image, language, and state tokens for inherited action sampling."""
        if state is None:
            raise ValueError("Action Flow V2 requires the robot state prefix token.")
        prefix_embeddings, pad_masks, attention_groups, image_layout = self._embed_prefix_with_layout(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        )
        self._inference_image_layout = image_layout
        return prefix_embeddings, pad_masks, attention_groups

    def _on_inference_prefix_output(self, prefix_output: Tensor) -> None:
        """Cache CPU heatmaps from the action sampler's existing prefix pass."""
        if self._inference_image_layout is None:
            return
        self.latest_relevance_heatmaps = (
            self._predict_relevance(prefix_output, self._inference_image_layout)
            .sigmoid()
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )

    def _predict_relevance(
        self,
        prefix_output: Tensor,
        image_layout: list[tuple[slice, tuple[int, int]]],
    ) -> Tensor:
        camera_logits = []
        head_dtype = next(self.relevance_head.parameters()).dtype
        for token_slice, (grid_height, grid_width) in image_layout:
            spatial_tokens = prefix_output[:, token_slice].to(dtype=head_dtype)
            logits = self.relevance_head(spatial_tokens).squeeze(-1)
            camera_logits.append(logits.reshape(logits.shape[0], grid_height, grid_width))
        return torch.stack(camera_logits, dim=1)

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute flow-matching losses and prompt-conditioned camera heatmaps."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        noisy_actions = time_expanded * noise + (1 - time_expanded) * actions
        target_velocity = noise - actions

        prefix_embeddings, prefix_pad_masks, prefix_attention_groups, image_layout = (
            self._embed_prefix_with_layout(images, img_masks, lang_tokens, lang_masks, state)
        )
        suffix_embeddings, suffix_pad_masks, suffix_attention_groups = self.embed_suffix(
            noisy_actions,
            time,
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        attention_groups = torch.cat([prefix_attention_groups, suffix_attention_groups], dim=1)
        attention_mask = make_att_2d_masks(pad_masks, attention_groups)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (prefix_output, suffix_output), _ = self.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embeddings, suffix_embeddings],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_output = suffix_output[:, -self.config.chunk_size :].to(dtype=torch.float32)
        predicted_velocity = self.action_out_proj(suffix_output)
        flow_losses = F.mse_loss(target_velocity, predicted_velocity, reduction="none")
        relevance_logits = self._predict_relevance(prefix_output, image_layout)
        return flow_losses, relevance_logits

    def predict_relevance_logits(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
    ) -> Tensor:
        """Predict relevance logits without constructing or denoising an action suffix."""
        prefix_embeddings, prefix_pad_masks, prefix_attention_groups, image_layout = (
            self._embed_prefix_with_layout(images, img_masks, lang_tokens, lang_masks, state)
        )
        attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_attention_groups)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        (prefix_output, _), _ = self.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embeddings, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        return self._predict_relevance(prefix_output, image_layout)


class ActionFlowV2Policy(SmolVLAPolicy):
    """LeRobot policy wrapper for Action Flow V2 training and inference."""

    config_class = ActionFlowV2Config
    name = "action_flow_v2"

    def __init__(self, config: ActionFlowV2Config, **kwargs):
        require_package("transformers", extra="smolvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = ActionFlowV2Model(config, rtc_processor=self.rtc_processor)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff,
                config.chunk_size,
            )
        self.reset()

    def reset(self):
        """Reset temporal-ensemble or action-queue inference state."""
        self.model.latest_relevance_heatmaps = None
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute flow and auxiliary relevance losses for one training batch."""
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
                "Action Flow V2 training requires merged dataset features "
                f"`{RELEVANCE_MASK}` and `{RELEVANCE_LOSS_WEIGHT}`."
            )
        relevance_loss_per_sample, relevance_metrics = compute_relevance_loss(
            relevance_logits,
            batch[RELEVANCE_MASK],
            batch[RELEVANCE_LOSS_WEIGHT],
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

    @torch.no_grad()
    def predict_relevance_heatmaps(self, batch: dict[str, Tensor]) -> Tensor:
        """Return prompt-conditioned heatmap probabilities with shape `(B, C, H_p, W_p)`."""
        self.eval()
        batch = self._prepare_batch(batch)
        images, img_masks = self.prepare_images(batch)
        logits = self.model.predict_relevance_logits(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            self.prepare_state(batch),
        )
        return logits.sigmoid()

    def get_latest_relevance_heatmaps(self) -> Tensor | None:
        """Return the latest action-inference heatmaps on CPU without another VLM pass."""
        heatmaps = self.model.latest_relevance_heatmaps
        return None if heatmaps is None else heatmaps.clone()

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """Include the grounding modules when Action Flow V2 is trained with PEFT."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = (
            rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        )
        return {
            "target_modules": target_modules,
            "modules_to_save": ["model.camera_embeddings", "model.relevance_head"],
        }
