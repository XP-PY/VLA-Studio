#!/usr/bin/env python

"""Jointly trained, knowledge-insulated Action Flow V2.2 policy."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors import safe_open
from torch import Tensor

from lerobot.configs import PreTrainedConfig
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
from ..smolvla.configuration_smolvla import SmolVLAConfig
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
    """Independent compact teacher targets for grounding and FAST decoding."""

    grounding_token_ids: Tensor
    grounding_token_mask: Tensor
    grounding_loss_weight: Tensor
    action_token_ids: Tensor
    action_token_mask: Tensor


@dataclass(frozen=True)
class ActionFlowV22ForwardOutput:
    """Per-sample and per-element outputs from one joint training pass."""

    flow_losses: Tensor
    box_loss: Tensor
    fast_loss: Tensor


@dataclass(frozen=True)
class SmolVLAInitializationReport:
    """Summary of parameters copied from a pretrained SmolVLA policy."""

    source: str
    exact_tensor_count: int
    vocabulary_prefix_tensor_count: int
    retained_v2_2_tensor_count: int
    ignored_source_tensor_count: int


_VOCABULARY_WEIGHT_KEYS = frozenset(
    {
        "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight",
        "model.vlm_with_expert.vlm.lm_head.weight",
    }
)
_REQUIRED_SMOLVLA_WEIGHT_KEYS = _VOCABULARY_WEIGHT_KEYS | {
    "model.action_in_proj.weight",
    "model.action_out_proj.weight",
    "model.state_proj.weight",
}


def _effective_expert_layers(config: SmolVLAConfig) -> int:
    return config.num_expert_layers if config.num_expert_layers > 0 else config.num_vlm_layers


def _validate_smolvla_initialization_config(
    source_config: PreTrainedConfig,
    target_config: ActionFlowV22Config,
) -> None:
    """Reject a source checkpoint whose shared SmolVLA architecture is incompatible."""

    if not isinstance(source_config, SmolVLAConfig) or source_config.type != "smolvla":
        raise TypeError(
            "`smolvla_pretrained_path` must identify a SmolVLA checkpoint, got "
            f"{source_config.type!r}."
        )
    mismatches = []
    for name in (
        "vlm_model_name",
        "num_vlm_layers",
        "expert_width_multiplier",
        "attention_mode",
        "self_attn_every_n_layers",
        "max_state_dim",
        "max_action_dim",
    ):
        source_value = getattr(source_config, name)
        target_value = getattr(target_config, name)
        if source_value != target_value:
            mismatches.append(f"{name}: source={source_value!r}, target={target_value!r}")
    source_expert_layers = _effective_expert_layers(source_config)
    target_expert_layers = _effective_expert_layers(target_config)
    if source_expert_layers != target_expert_layers:
        mismatches.append(
            "effective expert layers: "
            f"source={source_expert_layers}, target={target_expert_layers}"
        )
    if mismatches:
        raise ValueError("Incompatible SmolVLA initialization checkpoint: " + "; ".join(mismatches))


def _resolve_smolvla_checkpoint(source: str) -> tuple[str | Path, Path]:
    """Resolve a local directory/file or Hub model ID to config and safetensors sources."""

    local_source = Path(source).expanduser()
    if local_source.is_file():
        if local_source.name != SAFETENSORS_SINGLE_FILE:
            raise ValueError(
                f"A SmolVLA checkpoint file must be named {SAFETENSORS_SINGLE_FILE!r}."
            )
        return local_source.parent, local_source
    if local_source.is_dir():
        model_file = local_source / SAFETENSORS_SINGLE_FILE
        if not model_file.is_file():
            raise FileNotFoundError(f"SmolVLA weights not found: {model_file}")
        return local_source, model_file
    model_file = Path(hf_hub_download(repo_id=source, filename=SAFETENSORS_SINGLE_FILE))
    return source, model_file


def _smolvla_tensor_copy_mode(
    key: str,
    target_shape: tuple[int, ...],
    source_shape: tuple[int, ...],
    *,
    base_vocab_size: int,
) -> str | None:
    """Classify one compatible exact or vocabulary-prefix tensor copy."""

    if target_shape == source_shape:
        return "exact"
    if (
        key in _VOCABULARY_WEIGHT_KEYS
        and len(target_shape) == len(source_shape) == 2
        and source_shape[0] == base_vocab_size
        and target_shape[0] > source_shape[0]
        and target_shape[1:] == source_shape[1:]
    ):
        return "vocabulary_prefix"
    return None


def _copy_smolvla_tensor(target: Tensor, source: Tensor, mode: str) -> None:
    """Apply one validated SmolVLA tensor copy while preserving added vocabulary rows."""

    source = source.to(device=target.device, dtype=target.dtype)
    if mode == "exact":
        target.copy_(source)
    elif mode == "vocabulary_prefix":
        target[: source.shape[0]].copy_(source)
    else:
        raise ValueError(f"Unsupported SmolVLA tensor copy mode: {mode!r}.")


def initialize_from_smolvla_pretrained(
    policy: ActionFlowV22Policy,
    source: str,
) -> SmolVLAInitializationReport:
    """Warm-start V2.2 from compatible SmolVLA weights without replacing added token rows."""

    config_source, model_file = _resolve_smolvla_checkpoint(source)
    source_config = PreTrainedConfig.from_pretrained(config_source)
    _validate_smolvla_initialization_config(source_config, policy.config)

    target_state = policy.state_dict()
    copy_modes: dict[str, str] = {}
    incompatible = []
    with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
        source_keys = set(checkpoint.keys())
        for key in sorted(source_keys & target_state.keys()):
            source_shape = tuple(checkpoint.get_slice(key).get_shape())
            target_shape = tuple(target_state[key].shape)
            mode = _smolvla_tensor_copy_mode(
                key,
                target_shape,
                source_shape,
                base_vocab_size=policy.config.base_vlm_vocab_size,
            )
            if mode is None:
                incompatible.append(f"{key}: source={source_shape}, target={target_shape}")
            else:
                copy_modes[key] = mode
        missing_required = sorted(_REQUIRED_SMOLVLA_WEIGHT_KEYS - copy_modes.keys())
        if incompatible or missing_required:
            details = []
            if incompatible:
                details.append("incompatible tensors: " + "; ".join(incompatible))
            if missing_required:
                details.append(f"missing required tensors: {missing_required}")
            raise ValueError("Cannot initialize Action Flow V2.2 from SmolVLA: " + "; ".join(details))

        with torch.no_grad():
            for key, mode in copy_modes.items():
                source_tensor = checkpoint.get_tensor(key)
                target_tensor = target_state[key]
                _copy_smolvla_tensor(target_tensor, source_tensor, mode)

    report = SmolVLAInitializationReport(
        source=source,
        exact_tensor_count=sum(mode == "exact" for mode in copy_modes.values()),
        vocabulary_prefix_tensor_count=sum(
            mode == "vocabulary_prefix" for mode in copy_modes.values()
        ),
        retained_v2_2_tensor_count=len(target_state.keys() - copy_modes.keys()),
        ignored_source_tensor_count=len(source_keys - target_state.keys()),
    )
    logging.info(
        "Initialized Action Flow V2.2 from SmolVLA %s: exact=%d, vocabulary_prefix=%d, "
        "retained_v2_2=%d, ignored_source=%d.",
        report.source,
        report.exact_tensor_count,
        report.vocabulary_prefix_tensor_count,
        report.retained_v2_2_tensor_count,
        report.ignored_source_tensor_count,
    )
    return report


def pack_autoregressive_targets(
    grounding_tokens: Tensor,
    grounding_mask: Tensor,
    grounding_loss_weight: Tensor,
    action_tokens: Tensor,
    action_mask: Tensor,
    *,
    pad_token_id: int,
    grounding_start_token_id: int,
    action_start_token_id: int,
) -> PackedAutoregressiveTargets:
    """Compact independent grounding and FAST labels after fixed decoder starts.

    Every serialized grounding sequence begins with ``<grounding>``. That token
    is supplied to the grounding branch rather than predicted. Likewise,
    ``<action>`` starts an independent FAST branch. Neither branch can attend to
    the other, and the continuous action expert receives neither branch.
    """

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

    grounding_sequences: list[Tensor] = []
    grounding_weights: list[Tensor] = []
    action_sequences: list[Tensor] = []
    for index in range(batch_size):
        valid_grounding = grounding_mask[index].bool()
        valid_actions = action_mask[index].bool()
        grounding_ids = grounding_tokens[index, valid_grounding]
        action_ids = action_tokens[index, valid_actions]
        if grounding_ids.numel() == 0 or action_ids.numel() == 0:
            raise ValueError("Every training sample requires grounding and FAST target tokens.")
        if int(grounding_ids[0].item()) != grounding_start_token_id:
            raise ValueError(
                "Every grounding sequence must begin with the configured grounding-start token."
            )
        if int(action_ids[0].item()) != action_start_token_id:
            raise ValueError(
                "Every FAST sequence must begin with the configured action-start token."
            )
        grounding_ids = grounding_ids[1:]
        sample_grounding_weights = grounding_loss_weight[index, valid_grounding].float()[1:]
        action_ids = action_ids[1:]
        if grounding_ids.numel() == 0:
            raise ValueError("Grounding sequences require a target after the decoder-start token.")
        if action_ids.numel() == 0:
            raise ValueError("FAST sequences require a target after the decoder-start token.")

        grounding_sequences.append(grounding_ids)
        grounding_weights.append(sample_grounding_weights)
        action_sequences.append(action_ids)

    max_grounding_length = max(sequence.shape[0] for sequence in grounding_sequences)
    max_action_length = max(sequence.shape[0] for sequence in action_sequences)
    packed_grounding_ids = torch.full(
        (batch_size, max_grounding_length),
        pad_token_id,
        device=grounding_tokens.device,
        dtype=torch.long,
    )
    packed_grounding_mask = torch.zeros(
        (batch_size, max_grounding_length),
        device=grounding_tokens.device,
        dtype=torch.bool,
    )
    packed_grounding_weights = torch.zeros(
        (batch_size, max_grounding_length),
        device=grounding_tokens.device,
        dtype=torch.float32,
    )
    packed_action_ids = torch.full(
        (batch_size, max_action_length),
        pad_token_id,
        device=action_tokens.device,
        dtype=torch.long,
    )
    packed_action_mask = torch.zeros(
        (batch_size, max_action_length),
        device=action_tokens.device,
        dtype=torch.bool,
    )
    for index, sequence in enumerate(grounding_sequences):
        length = sequence.shape[0]
        packed_grounding_ids[index, :length] = sequence
        packed_grounding_mask[index, :length] = True
        packed_grounding_weights[index, :length] = grounding_weights[index]
    for index, sequence in enumerate(action_sequences):
        length = sequence.shape[0]
        packed_action_ids[index, :length] = sequence
        packed_action_mask[index, :length] = True

    return PackedAutoregressiveTargets(
        grounding_token_ids=packed_grounding_ids,
        grounding_token_mask=packed_grounding_mask,
        grounding_loss_weight=packed_grounding_weights,
        action_token_ids=packed_action_ids,
        action_token_mask=packed_action_mask,
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


def build_branched_attention_mask(
    prefix_pad_mask: Tensor,
    prefix_attention_groups: Tensor,
    grounding_mask: Tensor,
    action_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build independent causal grounding and FAST branches over one prefix.

    Both branches attend to the shared image/language/state prefix and their own
    previous tokens. Grounding tokens never attend to FAST tokens, and FAST
    tokens never attend to grounding tokens. Their position IDs independently
    continue from the end of the valid shared prefix.
    """

    if prefix_pad_mask.shape != prefix_attention_groups.shape:
        raise ValueError("Prefix padding and attention-group masks must have identical shapes.")
    batch_size, prefix_length = prefix_pad_mask.shape
    for name, branch_mask in (("grounding", grounding_mask), ("action", action_mask)):
        if branch_mask.ndim != 2 or branch_mask.shape[0] != batch_size:
            raise ValueError(
                f"{name.capitalize()} mask must be shaped (batch_size, branch_length)."
            )

    prefix_pad_mask = prefix_pad_mask.bool()
    grounding_mask = grounding_mask.bool()
    action_mask = action_mask.bool()
    grounding_length = grounding_mask.shape[1]
    action_length = action_mask.shape[1]
    total_length = prefix_length + grounding_length + action_length
    attention_mask = torch.zeros(
        (batch_size, total_length, total_length),
        dtype=torch.bool,
        device=prefix_pad_mask.device,
    )
    attention_mask[:, :prefix_length, :prefix_length] = make_att_2d_masks(
        prefix_pad_mask,
        prefix_attention_groups,
    )

    prefix_keys = prefix_pad_mask[:, None, :]
    grounding_start = prefix_length
    grounding_stop = grounding_start + grounding_length
    action_start = grounding_stop
    action_stop = action_start + action_length
    attention_mask[:, grounding_start:grounding_stop, :prefix_length] = (
        grounding_mask[:, :, None] & prefix_keys
    )
    attention_mask[:, action_start:action_stop, :prefix_length] = (
        action_mask[:, :, None] & prefix_keys
    )

    grounding_causal = torch.ones(
        (grounding_length, grounding_length),
        dtype=torch.bool,
        device=prefix_pad_mask.device,
    ).tril()
    action_causal = torch.ones(
        (action_length, action_length),
        dtype=torch.bool,
        device=prefix_pad_mask.device,
    ).tril()
    attention_mask[:, grounding_start:grounding_stop, grounding_start:grounding_stop] = (
        grounding_mask[:, :, None]
        & grounding_mask[:, None, :]
        & grounding_causal[None]
    )
    attention_mask[:, action_start:action_stop, action_start:action_stop] = (
        action_mask[:, :, None] & action_mask[:, None, :] & action_causal[None]
    )

    prefix_positions = torch.cumsum(prefix_pad_mask, dim=1) - 1
    branch_position_start = prefix_pad_mask.sum(dim=1, keepdim=True)
    grounding_positions = (
        branch_position_start + torch.cumsum(grounding_mask, dim=1) - 1
    )
    action_positions = branch_position_start + torch.cumsum(action_mask, dim=1) - 1
    position_ids = torch.cat(
        [prefix_positions, grounding_positions, action_positions],
        dim=1,
    ).clamp_min(0)
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
    ) -> tuple[Tensor, Tensor, dict[int, dict[str, Tensor]]]:
        """Predict independent grounding and FAST targets in one masked VLM pass.

        The physical token layout is ``prefix, grounding branch, FAST branch``,
        but the block mask makes the two suffixes logically independent. The
        returned cache contains only the original shared prefix for use by the
        continuous action expert.
        """

        batch_size = targets.grounding_token_ids.shape[0]
        grounding_start_ids = torch.full(
            (batch_size, 1),
            self.config.structural_token_id("<grounding>"),
            dtype=torch.long,
            device=targets.grounding_token_ids.device,
        )
        action_start_ids = torch.full(
            (batch_size, 1),
            self.config.structural_token_id("<action>"),
            dtype=torch.long,
            device=targets.action_token_ids.device,
        )
        grounding_start_embeddings = self.vlm_with_expert.embed_language_tokens(
            grounding_start_ids
        )
        action_start_embeddings = self.vlm_with_expert.embed_language_tokens(
            action_start_ids
        )
        grounding_target_embeddings = self.vlm_with_expert.embed_language_tokens(
            targets.grounding_token_ids
        )
        action_target_embeddings = self.vlm_with_expert.embed_language_tokens(
            targets.action_token_ids
        )
        embedding_scale = math.sqrt(prefix_embeddings.shape[-1])
        grounding_start_embeddings = grounding_start_embeddings * embedding_scale
        action_start_embeddings = action_start_embeddings * embedding_scale
        grounding_target_embeddings = grounding_target_embeddings * embedding_scale
        action_target_embeddings = action_target_embeddings * embedding_scale
        full_embeddings = torch.cat(
            [
                prefix_embeddings,
                grounding_start_embeddings,
                grounding_target_embeddings,
                action_start_embeddings,
                action_target_embeddings,
            ],
            dim=1,
        )
        grounding_decoder_mask = torch.cat(
            [
                torch.ones(
                    (batch_size, 1),
                    dtype=torch.bool,
                    device=targets.grounding_token_mask.device,
                ),
                targets.grounding_token_mask,
            ],
            dim=1,
        )
        action_decoder_mask = torch.cat(
            [
                torch.ones(
                    (batch_size, 1),
                    dtype=torch.bool,
                    device=targets.action_token_mask.device,
                ),
                targets.action_token_mask,
            ],
            dim=1,
        )
        attention_mask, position_ids = build_branched_attention_mask(
            prefix_pad_mask,
            prefix_attention_groups,
            grounding_decoder_mask,
            action_decoder_mask,
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
        grounding_target_length = targets.grounding_token_ids.shape[1]
        grounding_start = prefix_length
        grounding_prediction_hidden = torch.cat(
            [
                full_output[:, grounding_start : grounding_start + 1],
                full_output[
                    :,
                    grounding_start + 1 : grounding_start + grounding_target_length,
                ],
            ],
            dim=1,
        )
        action_start = grounding_start + 1 + grounding_target_length
        action_target_length = targets.action_token_ids.shape[1]
        action_prediction_hidden = torch.cat(
            [
                full_output[:, action_start : action_start + 1],
                full_output[
                    :,
                    action_start + 1 : action_start + action_target_length,
                ],
            ],
            dim=1,
        )
        prediction_hidden = torch.cat(
            [grounding_prediction_hidden, action_prediction_hidden],
            dim=1,
        )
        prediction_logits = self.vlm_with_expert.vlm.get_output_embeddings()(
            prediction_hidden
        )
        grounding_logits = prediction_logits[:, :grounding_target_length]
        action_logits = prediction_logits[
            :,
            grounding_target_length : grounding_target_length + action_target_length,
        ]
        prefix_cache = slice_prefix_key_values(
            full_cache,
            prefix_length=prefix_length,
            detach=self.config.detach_expert_prefix_kv,
        )
        return grounding_logits, action_logits, prefix_cache

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
            grounding_start_token_id=self.config.structural_token_id("<grounding>"),
            action_start_token_id=self.config.structural_token_id("<action>"),
        )
        grounding_logits, action_logits, prefix_cache = self._autoregressive_vlm_pass(
            prefix_embeddings,
            prefix_pad_mask,
            prefix_attention_groups,
            packed_targets,
        )
        box_loss = weighted_token_cross_entropy(
            grounding_logits,
            packed_targets.grounding_token_ids,
            packed_targets.grounding_loss_weight,
        )
        fast_loss = weighted_token_cross_entropy(
            action_logits,
            packed_targets.action_token_ids,
            packed_targets.action_token_mask.float(),
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

    def __init__(
        self,
        config: ActionFlowV22Config,
        *,
        initialize_from_smolvla: bool = True,
        **kwargs,
    ):
        require_package("transformers", extra="smolvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = ActionFlowV22Model(config, rtc_processor=self.rtc_processor)
        self.smolvla_initialization_report = None
        if initialize_from_smolvla and config.smolvla_pretrained_path is not None:
            self.smolvla_initialization_report = initialize_from_smolvla_pretrained(
                self,
                config.smolvla_pretrained_path,
            )
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
        # The complete Action Flow V2.2 checkpoint is loaded immediately after
        # construction, so its original SmolVLA warm start is neither needed nor
        # required to remain available on the inference machine.
        kwargs["initialize_from_smolvla"] = False
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
    "SmolVLAInitializationReport",
    "build_branched_attention_mask",
    "build_joint_attention_mask",
    "initialize_from_smolvla_pretrained",
    "pack_autoregressive_targets",
    "slice_prefix_key_values",
    "weighted_token_cross_entropy",
]
