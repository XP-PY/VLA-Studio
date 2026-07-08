#!/usr/bin/env python

"""ActionFlowV1 policy.

This policy is intentionally close to ACT at the input/output boundary, but it
uses a flow-matching objective over action chunks instead of direct action
regression. It is meant as a small, swappable baseline for experiments where the
image encoder or sequence mixer may change between competition iterations.
"""

import math
from collections import deque
from itertools import chain
from typing import TypedDict, Unpack

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from ..act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
)
from ..pretrained import PreTrainedPolicy
from ..rtc.modeling_rtc import RTCProcessor
from .configuration_action_flow_v1 import ActionFlowV1Config


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_scalar_sinusoidal_embedding(
    time: Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> Tensor:
    """Create sine/cosine embeddings for scalar flow timesteps."""

    if dimension % 2 != 0:
        raise ValueError(f"`dimension` must be even, got {dimension}.")
    if time.ndim != 1:
        raise ValueError(f"`time` must have shape `(batch_size,)`, got {tuple(time.shape)}.")

    dtype = torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=time.device)
    period = min_period * (max_period / min_period) ** fraction
    angular_speed = 2 * math.pi / period
    angles = time[:, None].to(dtype=dtype) * angular_speed[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ActionFlowV1Policy(PreTrainedPolicy):
    """ACT-style image/state policy with flow-matching action chunk generation."""

    config_class = ActionFlowV1Config
    name = "action_flow_v1"

    def __init__(self, config: ActionFlowV1Config, **kwargs):
        super().__init__(config)
        del kwargs

        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = ActionFlowV1(config, rtc_processor=self.rtc_processor)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in the policy config."""
        self.rtc_processor = None

        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def get_optim_params(self) -> list[dict]:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.config.image_features:
            return batch
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        del kwargs
        assert not self._rtc_enabled(), "RTC is not supported for select_action, use it with predict_action_chunk."

        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch, noise=noise)
            action = self.temporal_ensembler.update(actions)
            return action

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()
        return self.model.sample_actions(self._prepare_batch(batch), noise=noise, **kwargs)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self._prepare_batch(batch)
        velocity, target_velocity = self.model(batch)

        losses = F.mse_loss(velocity, target_velocity, reduction="none")
        if "action_is_pad" in batch:
            valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
            num_valid = (valid_mask.sum() * losses.shape[-1]).clamp_min(1)
            loss = (losses * valid_mask).sum() / num_valid
        else:
            loss = losses.mean()

        return loss, {"flow_loss": loss.item()}


class ActionFlowV1(nn.Module):
    """Action Flow V1: The underlying neural network for ActionFlowV1Policy.

    Note: In this code we use the terms `encoder`, `decoder`, and `flow target`. The meanings are as
    follows.
        - The `encoder` is the transformer encoder that encodes the observation condition, including image
          features, robot state, and environment state when available.
        - The `decoder` is the transformer decoder that takes a noisy action chunk and attends to the encoded
          observation condition.
        - The `flow target` is the velocity from clean actions to sampled noise. The model predicts this
          velocity instead of directly regressing the action chunk.

                                  Transformer
                                  Used as a flow model
                                  over action chunks
                                ┌────────────────────────┐
                                │              Outputs   │
                                │                 ▲      │
                                │             ┌───────┐  │
              noisy actions ────┼────────────►│Transf.│  │
              timestep ─────────┼────────────►│decoder│  │
                                │ ┌──────────►│       │  │
                                │ │        │  └───────┘  │
                                │ │Transf. │             │
              inputs ───────────┼►│encoder │             │
                                │ │        │             │
                                │ └▲──▲─▲──┘             │
                                │  │  │ │                │
                                │  │  │ image emb.       │
                                │  │  state emb.         │
                                │  condition emb.        │
                                └────────────────────────┘
    """

    def __init__(self, config: ActionFlowV1Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        # Backbone for image feature extraction.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )

        n_1d_tokens = 1
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_condition_token = nn.Embedding(1, config.dim_model)
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

        action_dim = self.config.action_feature.shape[0]
        self.action_input_proj = nn.Linear(action_dim, config.dim_model)
        self.time_input_proj = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_model),
            nn.SiLU(),
            nn.Linear(config.dim_model, config.dim_model),
        )
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.velocity_head = nn.Linear(config.dim_model, action_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        beta_dist = torch.distributions.Beta(
            concentration1=self.config.time_sampling_alpha,
            concentration0=self.config.time_sampling_beta,
        )
        time = beta_dist.sample((batch_size,)).to(device=device, dtype=torch.float32)
        return time * 0.999 + 0.001

    def encode_condition(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
            device = batch[OBS_IMAGES][0].device
        elif OBS_STATE in batch:
            batch_size = batch[OBS_STATE].shape[0]
            device = batch[OBS_STATE].device
        else:
            batch_size = batch[OBS_ENV_STATE].shape[0]
            device = batch[OBS_ENV_STATE].device

        condition_token = einops.repeat(self.encoder_condition_token.weight, "1 d -> b d", b=batch_size)
        encoder_tokens = [condition_token]
        encoder_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.robot_state_feature:
            encoder_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_tokens.extend(list(cam_features))
                encoder_pos_embed.extend(list(cam_pos_embed))

        encoder_tokens = torch.stack(encoder_tokens, dim=0)
        encoder_pos_embed = torch.stack(encoder_pos_embed, dim=0).to(device=device)
        encoder_out = self.encoder(encoder_tokens, pos_embed=encoder_pos_embed)
        return encoder_out, encoder_pos_embed

    def predict_velocity_from_condition(
        self,
        noisy_actions: Tensor,
        time: Tensor,
        encoder_out: Tensor,
        encoder_pos_embed: Tensor,
    ) -> Tensor:
        action_tokens = self.action_input_proj(noisy_actions)
        time_embed = create_scalar_sinusoidal_embedding(time, self.config.dim_model)
        time_embed = self.time_input_proj(time_embed).to(dtype=action_tokens.dtype)
        decoder_in = action_tokens + time_embed[:, None, :]
        decoder_in = decoder_in.transpose(0, 1)

        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)
        return self.velocity_head(decoder_out)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        actions = batch[ACTION]
        noise = torch.randn_like(actions)
        time = self.sample_time(actions.shape[0], actions.device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        target_velocity = noise - actions

        encoder_out, encoder_pos_embed = self.encode_condition(batch)
        velocity = self.predict_velocity_from_condition(x_t, time, encoder_out, encoder_pos_embed)
        return velocity, target_velocity

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    @torch.no_grad()
    def sample_actions(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        encoder_out, encoder_pos_embed = self.encode_condition(batch)
        batch_size = encoder_out.shape[1]
        device = encoder_out.device
        action_dim = self.config.action_feature.shape[0]

        if noise is None:
            x_t = torch.randn(batch_size, self.config.chunk_size, action_dim, device=device)
        else:
            x_t = noise

        dt = -1.0 / self.config.num_inference_steps
        for step in range(self.config.num_inference_steps):
            time = 1.0 + step * dt
            time_tensor = torch.full((batch_size,), time, dtype=torch.float32, device=device)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.predict_velocity_from_condition(
                    input_x_t, current_timestep, encoder_out, encoder_pos_embed
                )

            if self._rtc_enabled():
                velocity = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                    inference_delay=kwargs.get("inference_delay"),
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=kwargs.get("execution_horizon"),
                )
            else:
                velocity = denoise_step_partial_call(x_t)

            x_t = x_t + dt * velocity

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=velocity)

        return x_t
