#!/usr/bin/env python

"""Jointly trained, knowledge-insulated Action Flow V2.2 policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.import_utils import require_package

from ..act.modeling_act import ACTTemporalEnsembler
from ..pretrained import PreTrainedPolicy
from ..smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching, make_att_2d_masks
from .configuration_action_flow_v2_2 import ActionFlowV22Config
from .data_contracts_action_flow_v2_2 import (
    GROUNDING_TOKEN_LOSS_WEIGHT,
    GROUNDING_TOKEN_MASK,
    GROUNDING_TOKENS,
)
from .processor_action_flow_v2_2 import resolve_tokenizer_artifact_path
from .tokenization_action_flow_v2_2 import (
    ActionFlowV22TargetTokenizer,
    resize_vlm_token_embeddings,
)


@dataclass(frozen=True)
class PackedAutoregressiveTargets:
    """Compact teacher targets shared by the grounding and FAST CE objectives."""

    token_ids: Tensor
    token_mask: Tensor
    box_loss_weight: Tensor
    fast_loss_weight: Tensor


@dataclass(frozen=True)
class ActionFlowV22ForwardOutput:
    """Per-sample and per-element outputs from one joint training pass."""

    flow_losses: Tensor
    box_loss: Tensor
    fast_loss: Tensor


def pack_autoregressive_targets(
    grounding_tokens: Tensor,
    grounding_mask: Tensor,
    grounding_loss_weight: Tensor,
    action_tokens: Tensor,
    action_mask: Tensor,
    *,
    pad_token_id: int,
) -> PackedAutoregressiveTargets:
    """Remove internal padding and concatenate grounding then FAST targets."""

    tensors = (
        grounding_tokens,
        grounding_mask,
        grounding_loss_weight,
        action_tokens,
        action_mask,
    )
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("All autoregressive target tensors must be two-dimensional.")
    batch_size = grounding_tokens.shape[0]
    if any(tensor.shape[0] != batch_size for tensor in tensors):
        raise ValueError("Autoregressive target tensors have different batch sizes.")
    if grounding_tokens.shape != grounding_mask.shape:
        raise ValueError("Grounding token IDs and masks must have identical shapes.")
    if grounding_tokens.shape != grounding_loss_weight.shape:
        raise ValueError("Grounding token IDs and loss weights must have identical shapes.")
    if action_tokens.shape != action_mask.shape:
        raise ValueError("FAST token IDs and masks must have identical shapes.")

    sequences: list[Tensor] = []
    box_weights: list[Tensor] = []
    fast_weights: list[Tensor] = []
    for index in range(batch_size):
        valid_grounding = grounding_mask[index].bool()
        valid_actions = action_mask[index].bool()
        grounding_ids = grounding_tokens[index, valid_grounding]
        action_ids = action_tokens[index, valid_actions]
        if grounding_ids.numel() == 0 or action_ids.numel() == 0:
            raise ValueError("Every training sample requires grounding and FAST target tokens.")

        sequences.append(torch.cat([grounding_ids, action_ids]))
        box_weights.append(
            torch.cat(
                [
                    grounding_loss_weight[index, valid_grounding].float(),
                    torch.zeros(
                        action_ids.shape[0],
                        device=grounding_tokens.device,
                        dtype=torch.float32,
                    ),
                ]
            )
        )
        fast_weights.append(
            torch.cat(
                [
                    torch.zeros(
                        grounding_ids.shape[0],
                        device=grounding_tokens.device,
                        dtype=torch.float32,
                    ),
                    torch.ones(
                        action_ids.shape[0],
                        device=grounding_tokens.device,
                        dtype=torch.float32,
                    ),
                ]
            )
        )

    max_length = max(sequence.shape[0] for sequence in sequences)
    token_ids = torch.full(
        (batch_size, max_length),
        pad_token_id,
        device=grounding_tokens.device,
        dtype=torch.long,
    )
    token_mask = torch.zeros(
        (batch_size, max_length),
        device=grounding_tokens.device,
        dtype=torch.bool,
    )
    packed_box_weights = torch.zeros(
        (batch_size, max_length),
        device=grounding_tokens.device,
        dtype=torch.float32,
    )
    packed_fast_weights = torch.zeros_like(packed_box_weights)
    for index, sequence in enumerate(sequences):
        length = sequence.shape[0]
        token_ids[index, :length] = sequence
        token_mask[index, :length] = True
        packed_box_weights[index, :length] = box_weights[index]
        packed_fast_weights[index, :length] = fast_weights[index]

    return PackedAutoregressiveTargets(
        token_ids=token_ids,
        token_mask=token_mask,
        box_loss_weight=packed_box_weights,
        fast_loss_weight=packed_fast_weights,
    )


def build_joint_attention_mask(
    prefix_pad_mask: Tensor,
    prefix_attention_groups: Tensor,
    target_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build a prefix-LM plus causal-target mask and corresponding position IDs."""

    if prefix_pad_mask.shape != prefix_attention_groups.shape:
        raise ValueError("Prefix padding and attention-group masks must have identical shapes.")
    if target_mask.ndim != 2 or target_mask.shape[0] != prefix_pad_mask.shape[0]:
        raise ValueError("Target mask must be shaped (batch_size, target_length).")

    full_pad_mask = torch.cat([prefix_pad_mask.bool(), target_mask.bool()], dim=1)
    target_attention_groups = target_mask.to(dtype=prefix_attention_groups.dtype)
    full_attention_groups = torch.cat(
        [prefix_attention_groups, target_attention_groups],
        dim=1,
    )
    attention_mask = make_att_2d_masks(full_pad_mask, full_attention_groups)
    position_ids = torch.cumsum(full_pad_mask, dim=1) - 1
    return attention_mask, position_ids


def slice_prefix_key_values(
    past_key_values: dict[int, dict[str, Tensor]],
    *,
    prefix_length: int,
    detach: bool,
) -> dict[int, dict[str, Tensor]]:
    """Discard teacher-target K/V and optionally detach every context cache tensor."""

    if past_key_values is None:
        raise ValueError("Joint Action Flow V2.2 training requires a VLM key/value cache.")

    prefix_cache: dict[int, dict[str, Tensor]] = {}
    for layer_index, layer_cache in past_key_values.items():
        prefix_cache[layer_index] = {}
        for name, value in layer_cache.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Unexpected non-tensor cache entry {layer_index}.{name}.")
            prefix_value = value[:, :prefix_length]
            prefix_cache[layer_index][name] = prefix_value.detach() if detach else prefix_value
    return prefix_cache


def weighted_token_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
) -> Tensor:
    """Return independently normalized token CE for each batch sample."""

    if logits.ndim != 3:
        raise ValueError("Token logits must be shaped (batch_size, sequence_length, vocab_size).")
    if targets.shape != logits.shape[:2] or weights.shape != targets.shape:
        raise ValueError("Token targets and weights must match the first two logit dimensions.")
    token_loss = F.cross_entropy(
        logits.transpose(1, 2).float(),
        targets.long(),
        reduction="none",
    )
    weights = weights.to(device=logits.device, dtype=token_loss.dtype).clamp_min(0)
    weight_sum = weights.sum(dim=1)
    per_sample = (token_loss * weights).sum(dim=1) / weight_sum.clamp_min(1e-8)
    return torch.where(weight_sum > 0, per_sample, per_sample * 0)


class ActionFlowV22Model(VLAFlowMatching):
    """SmolVLM autoregressive objectives plus an insulated continuous flow expert."""

    def __init__(self, config: ActionFlowV22Config, rtc_processor=None):
        compile_model = config.compile_model
        config.compile_model = False
        try:
            super().__init__(config, rtc_processor=rtc_processor)
        finally:
            config.compile_model = compile_model

        artifact_path = resolve_tokenizer_artifact_path(config)
        self.target_tokenizer = ActionFlowV22TargetTokenizer.from_pretrained(
            artifact_path,
            config=config,
        )
        self.vlm_with_expert.processor.tokenizer = self.target_tokenizer.vlm_tokenizer
        resize_vlm_token_embeddings(self.vlm_with_expert, config)
        self._enable_joint_training_parameters()

        if compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

    def _enable_joint_training_parameters(self) -> None:
        """Undo SmolVLA's expert-only DDP freezes for joint VLM/expert training."""

        for parameter in self.vlm_with_expert.vlm.parameters():
            parameter.requires_grad = True
        if self.config.freeze_vision_encoder:
            for parameter in self.vlm_with_expert.get_vlm_model().vision_model.parameters():
                parameter.requires_grad = False
        for parameter in self.vlm_with_expert.lm_expert.parameters():
            parameter.requires_grad = True
        for parameter in self.state_proj.parameters():
            parameter.requires_grad = self.config.train_state_proj

    def _autoregressive_vlm_pass(
        self,
        prefix_embeddings: Tensor,
        prefix_pad_mask: Tensor,
        prefix_attention_groups: Tensor,
        targets: PackedAutoregressiveTargets,
    ) -> tuple[Tensor, dict[int, dict[str, Tensor]]]:
        target_embeddings = self.vlm_with_expert.embed_language_tokens(targets.token_ids)
        target_embeddings = target_embeddings * math.sqrt(target_embeddings.shape[-1])
        full_embeddings = torch.cat([prefix_embeddings, target_embeddings], dim=1)
        attention_mask, position_ids = build_joint_attention_mask(
            prefix_pad_mask,
            prefix_attention_groups,
            targets.token_mask,
        )
        (full_output, _), full_cache = self.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[full_embeddings, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        prefix_length = prefix_embeddings.shape[1]
        valid_prefix_positions = torch.arange(
            prefix_length,
            device=prefix_pad_mask.device,
        )[None, :].expand_as(prefix_pad_mask)
        last_prefix_index = valid_prefix_positions.masked_fill(
            ~prefix_pad_mask.bool(),
            -1,
        ).max(dim=1).values
        if torch.any(last_prefix_index < 0):
            raise ValueError("Every sample requires at least one valid prefix token.")
        first_prediction_hidden = full_output[
            torch.arange(full_output.shape[0], device=full_output.device),
            last_prefix_index,
        ][:, None]
        remaining_prediction_hidden = full_output[
            :,
            prefix_length : prefix_length + targets.token_ids.shape[1] - 1,
        ]
        prediction_hidden = torch.cat(
            [first_prediction_hidden, remaining_prediction_hidden],
            dim=1,
        )
        logits = self.vlm_with_expert.vlm.get_output_embeddings()(prediction_hidden)
        prefix_cache = slice_prefix_key_values(
            full_cache,
            prefix_length=prefix_length,
            detach=self.config.detach_expert_prefix_kv,
        )
        return logits, prefix_cache

    def forward_joint(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        grounding_tokens: Tensor,
        grounding_mask: Tensor,
        grounding_loss_weight: Tensor,
        action_tokens: Tensor,
        action_token_mask: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> ActionFlowV22ForwardOutput:
        """Compute all VLM and expert objectives in one insulated training pass."""

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        target_velocity = noise - actions

        prefix_embeddings, prefix_pad_mask, prefix_attention_groups = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
        )
        packed_targets = pack_autoregressive_targets(
            grounding_tokens,
            grounding_mask,
            grounding_loss_weight,
            action_tokens,
            action_token_mask,
            pad_token_id=self.target_tokenizer.vlm_tokenizer.pad_token_id,
        )
        token_logits, prefix_cache = self._autoregressive_vlm_pass(
            prefix_embeddings,
            prefix_pad_mask,
            prefix_attention_groups,
            packed_targets,
        )
        box_loss = weighted_token_cross_entropy(
            token_logits,
            packed_targets.token_ids,
            packed_targets.box_loss_weight,
        )
        fast_loss = weighted_token_cross_entropy(
            token_logits,
            packed_targets.token_ids,
            packed_targets.fast_loss_weight,
        )

        predicted_velocity = self.denoise_step(
            prefix_pad_masks=prefix_pad_mask,
            past_key_values=prefix_cache,
            x_t=x_t,
            timestep=time,
        )
        flow_losses = F.mse_loss(target_velocity, predicted_velocity, reduction="none")
        return ActionFlowV22ForwardOutput(
            flow_losses=flow_losses,
            box_loss=box_loss,
            fast_loss=fast_loss,
        )


class ActionFlowV22Policy(SmolVLAPolicy):
    """LeRobot wrapper for joint Action Flow V2.2 training and rollout."""

    config_class = ActionFlowV22Config
    name = "action_flow_v2_2"

    def __init__(self, config: ActionFlowV22Config, **kwargs):
        require_package("transformers", extra="smolvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = ActionFlowV22Model(config, rtc_processor=self.rtc_processor)
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff,
                config.chunk_size,
            )
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: ActionFlowV22Config | None = None,
        **kwargs,
    ) -> ActionFlowV22Policy:
        """Load a checkpoint while resolving its bundled tokenizer artifact."""

        if config is None:
            config_load_keys = (
                "force_download",
                "resume_download",
                "proxies",
                "token",
                "cache_dir",
                "local_files_only",
                "revision",
            )
            config = ActionFlowV22Config.from_pretrained(
                pretrained_name_or_path,
                **{key: kwargs[key] for key in config_load_keys if key in kwargs},
            )
        config.pretrained_path = Path(pretrained_name_or_path)
        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            **kwargs,
        )

    def prepare_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Convert LeRobot uint8 video batches to the float image range used by SmolVLM."""

        prepared_batch = dict(batch)
        for key in self.config.image_features:
            image = prepared_batch.get(key)
            if isinstance(image, torch.Tensor) and not image.is_floating_point():
                prepared_batch[key] = image.float().div(255.0)
        return super().prepare_images(prepared_batch)

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float]]:
        """Train VLM grounding/FAST objectives and the insulated flow expert jointly."""

        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction {reduction!r}.")
        required = (
            GROUNDING_TOKENS,
            GROUNDING_TOKEN_MASK,
            GROUNDING_TOKEN_LOSS_WEIGHT,
            ACTION_TOKENS,
            ACTION_TOKEN_MASK,
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Action Flow V2.2 preprocessor did not produce {missing}.")

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        outputs = self.model.forward_joint(
            images=images,
            img_masks=img_masks,
            lang_tokens=batch[OBS_LANGUAGE_TOKENS],
            lang_masks=batch[OBS_LANGUAGE_ATTENTION_MASK],
            state=state,
            actions=actions,
            grounding_tokens=batch[GROUNDING_TOKENS],
            grounding_mask=batch[GROUNDING_TOKEN_MASK],
            grounding_loss_weight=batch[GROUNDING_TOKEN_LOSS_WEIGHT],
            action_tokens=batch[ACTION_TOKENS],
            action_token_mask=batch[ACTION_TOKEN_MASK],
            noise=noise,
            time=time,
        )

        original_action_dim = self.config.action_feature.shape[0]
        flow_losses = outputs.flow_losses[:, :, :original_action_dim]
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is None:
            flow_loss = flow_losses.mean(dim=(1, 2))
        else:
            valid = ~action_is_pad.bool()
            masked_flow_losses = flow_losses * valid.unsqueeze(-1)
            valid_count = (valid.sum(dim=1) * flow_losses.shape[-1]).clamp_min(1)
            flow_loss = masked_flow_losses.sum(dim=(1, 2)) / valid_count

        total_loss = (
            self.config.box_loss_weight * outputs.box_loss
            + self.config.fast_loss_weight * outputs.fast_loss
            + self.config.flow_loss_weight * flow_loss
        )
        metrics = {
            "loss": total_loss.detach().mean().item(),
            "box_ce_loss": outputs.box_loss.detach().mean().item(),
            "fast_ce_loss": outputs.fast_loss.detach().mean().item(),
            "flow_loss": flow_loss.detach().mean().item(),
        }
        if reduction == "none":
            return total_loss, metrics
        return total_loss.mean(), metrics

    def _save_pretrained(
        self,
        save_directory: Path,
        state_dict: dict[str, Tensor] | None = None,
    ) -> None:
        super()._save_pretrained(save_directory, state_dict=state_dict)
        self.model.target_tokenizer.save_pretrained(
            save_directory / self.config.checkpoint_tokenizer_dirname
        )

    def _get_default_peft_targets(self) -> dict[str, Any]:
        raise NotImplementedError(
            "PEFT targets are not defined for joint Action Flow V2.2 training yet."
        )


__all__ = [
    "ActionFlowV22ForwardOutput",
    "ActionFlowV22Model",
    "ActionFlowV22Policy",
    "PackedAutoregressiveTargets",
    "build_joint_attention_mask",
    "pack_autoregressive_targets",
    "slice_prefix_key_values",
    "weighted_token_cross_entropy",
]
