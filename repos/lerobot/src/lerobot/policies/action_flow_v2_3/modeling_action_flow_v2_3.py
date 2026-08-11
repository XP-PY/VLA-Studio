#!/usr/bin/env python

"""Planning-conditioned, knowledge-insulated Action Flow V2.3 policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
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
from ..action_flow_v2_2.data_contracts_action_flow_v2_2 import (
    GROUNDING_TOKEN_LOSS_WEIGHT,
    GROUNDING_TOKEN_MASK,
    GROUNDING_TOKENS,
)
from ..action_flow_v2_2.modeling_action_flow_v2_2 import (
    initialize_from_smolvla_pretrained,
    slice_prefix_key_values,
    weighted_token_cross_entropy,
)
from ..action_flow_v2_2.tokenization_action_flow_v2_2 import resize_vlm_token_embeddings
from ..pretrained import PreTrainedPolicy
from ..smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching, make_att_2d_masks
from .configuration_action_flow_v2_3 import ActionFlowV23Config
from .data_contracts_action_flow_v2_3 import (
    ACTION_TOKEN_LOSS_WEIGHT,
    PLAN_TOKEN_LOSS_WEIGHT,
    PLAN_TOKEN_MASK,
    PLAN_TOKENS,
    SEGMENT_ACTION_MASK,
)
from .processor_action_flow_v2_3 import resolve_tokenizer_artifact_path
from .tokenization_action_flow_v2_3 import ActionFlowV23TargetTokenizer


@dataclass(frozen=True)
class PackedAutoregressiveTargetsV23:
    """Compact plan, grounding, and FAST targets after fixed decoder starts."""

    plan_token_ids: Tensor
    plan_token_mask: Tensor
    plan_loss_weight: Tensor
    grounding_token_ids: Tensor
    grounding_token_mask: Tensor
    grounding_loss_weight: Tensor
    action_token_ids: Tensor
    action_token_mask: Tensor
    action_loss_weight: Tensor


@dataclass(frozen=True)
class ActionFlowV23ForwardOutput:
    """Per-sample autoregressive losses and per-element flow losses."""

    flow_losses: Tensor
    plan_loss: Tensor
    box_loss: Tensor
    fast_loss: Tensor


def _compact_branch(
    token_ids: Tensor,
    token_mask: Tensor,
    loss_weight: Tensor,
    *,
    start_token_id: int,
    pad_token_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if token_ids.ndim != 2 or token_mask.shape != token_ids.shape:
        raise ValueError("Branch token IDs and masks must have shape (batch_size, sequence_length).")
    if loss_weight.shape != token_ids.shape:
        raise ValueError("Branch token loss weights must match token IDs.")
    sequences: list[Tensor] = []
    weights: list[Tensor] = []
    for index in range(token_ids.shape[0]):
        valid = token_mask[index].bool()
        ids = token_ids[index, valid]
        sample_weights = loss_weight[index, valid].float()
        if ids.numel() < 2 or int(ids[0].item()) != start_token_id:
            raise ValueError("Every autoregressive branch requires its configured start token.")
        sequences.append(ids[1:])
        weights.append(sample_weights[1:])

    max_length = max(sequence.shape[0] for sequence in sequences)
    packed_ids = torch.full(
        (token_ids.shape[0], max_length),
        pad_token_id,
        dtype=torch.long,
        device=token_ids.device,
    )
    packed_mask = torch.zeros_like(packed_ids, dtype=torch.bool)
    packed_weight = torch.zeros_like(packed_ids, dtype=torch.float32)
    for index, sequence in enumerate(sequences):
        length = sequence.shape[0]
        packed_ids[index, :length] = sequence
        packed_mask[index, :length] = True
        packed_weight[index, :length] = weights[index]
    return packed_ids, packed_mask, packed_weight


def pack_autoregressive_targets_v2_3(
    *,
    plan_tokens: Tensor,
    plan_mask: Tensor,
    plan_loss_weight: Tensor,
    grounding_tokens: Tensor,
    grounding_mask: Tensor,
    grounding_loss_weight: Tensor,
    action_tokens: Tensor,
    action_mask: Tensor,
    action_loss_weight: Tensor,
    pad_token_id: int,
    plan_start_token_id: int,
    grounding_start_token_id: int,
    action_start_token_id: int,
) -> PackedAutoregressiveTargetsV23:
    """Remove fixed branch seeds and compact three independent target sequences."""

    plan_ids, plan_valid, plan_weights = _compact_branch(
        plan_tokens,
        plan_mask,
        plan_loss_weight,
        start_token_id=plan_start_token_id,
        pad_token_id=pad_token_id,
    )
    grounding_ids, grounding_valid, grounding_weights = _compact_branch(
        grounding_tokens,
        grounding_mask,
        grounding_loss_weight,
        start_token_id=grounding_start_token_id,
        pad_token_id=pad_token_id,
    )
    action_ids, action_valid, action_weights = _compact_branch(
        action_tokens,
        action_mask,
        action_loss_weight,
        start_token_id=action_start_token_id,
        pad_token_id=pad_token_id,
    )
    return PackedAutoregressiveTargetsV23(
        plan_token_ids=plan_ids,
        plan_token_mask=plan_valid,
        plan_loss_weight=plan_weights,
        grounding_token_ids=grounding_ids,
        grounding_token_mask=grounding_valid,
        grounding_loss_weight=grounding_weights,
        action_token_ids=action_ids,
        action_token_mask=action_valid,
        action_loss_weight=action_weights,
    )


def build_hierarchical_attention_mask(
    prefix_pad_mask: Tensor,
    prefix_attention_groups: Tensor,
    plan_mask: Tensor,
    grounding_mask: Tensor,
    action_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build prefix -> plan -> independent grounding/FAST branch attention.

    Plan tokens attend the shared prefix and causal plan history. Grounding and
    FAST tokens attend the shared prefix, the complete valid plan, and their own
    causal history, but never attend one another.
    """

    if prefix_pad_mask.shape != prefix_attention_groups.shape:
        raise ValueError("Prefix padding and attention-group masks must have identical shapes.")
    batch_size, prefix_length = prefix_pad_mask.shape
    for name, mask in (
        ("plan", plan_mask),
        ("grounding", grounding_mask),
        ("action", action_mask),
    ):
        if mask.ndim != 2 or mask.shape[0] != batch_size:
            raise ValueError(f"{name} mask must have shape (batch_size, branch_length).")

    prefix_pad_mask = prefix_pad_mask.bool()
    plan_mask = plan_mask.bool()
    grounding_mask = grounding_mask.bool()
    action_mask = action_mask.bool()
    lengths = (prefix_length, plan_mask.shape[1], grounding_mask.shape[1], action_mask.shape[1])
    total_length = sum(lengths)
    attention = torch.zeros(
        (batch_size, total_length, total_length),
        dtype=torch.bool,
        device=prefix_pad_mask.device,
    )
    attention[:, :prefix_length, :prefix_length] = make_att_2d_masks(
        prefix_pad_mask, prefix_attention_groups
    )

    plan_start = prefix_length
    plan_stop = plan_start + lengths[1]
    grounding_start = plan_stop
    grounding_stop = grounding_start + lengths[2]
    action_start = grounding_stop
    action_stop = action_start + lengths[3]
    prefix_keys = prefix_pad_mask[:, None, :]
    plan_keys = plan_mask[:, None, :]

    attention[:, plan_start:plan_stop, :prefix_length] = plan_mask[:, :, None] & prefix_keys
    attention[:, grounding_start:grounding_stop, :prefix_length] = (
        grounding_mask[:, :, None] & prefix_keys
    )
    attention[:, action_start:action_stop, :prefix_length] = (
        action_mask[:, :, None] & prefix_keys
    )
    attention[:, grounding_start:grounding_stop, plan_start:plan_stop] = (
        grounding_mask[:, :, None] & plan_keys
    )
    attention[:, action_start:action_stop, plan_start:plan_stop] = (
        action_mask[:, :, None] & plan_keys
    )

    for start, stop, mask in (
        (plan_start, plan_stop, plan_mask),
        (grounding_start, grounding_stop, grounding_mask),
        (action_start, action_stop, action_mask),
    ):
        length = stop - start
        causal = torch.ones((length, length), dtype=torch.bool, device=attention.device).tril()
        attention[:, start:stop, start:stop] = (
            mask[:, :, None] & mask[:, None, :] & causal[None]
        )

    prefix_positions = torch.cumsum(prefix_pad_mask, dim=1) - 1
    plan_positions = prefix_pad_mask.sum(dim=1, keepdim=True) + torch.cumsum(plan_mask, dim=1) - 1
    branch_start = prefix_pad_mask.sum(dim=1, keepdim=True) + plan_mask.sum(dim=1, keepdim=True)
    grounding_positions = branch_start + torch.cumsum(grounding_mask, dim=1) - 1
    action_positions = branch_start + torch.cumsum(action_mask, dim=1) - 1
    positions = torch.cat(
        [prefix_positions, plan_positions, grounding_positions, action_positions], dim=1
    ).clamp_min(0)
    return attention, positions


def _build_context_continuation_attention(
    context_attention: Tensor,
    context_pad_mask: Tensor,
    continuation_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Append one causal diagnostic branch to an already structured context."""

    batch_size, context_length = context_pad_mask.shape
    branch_length = continuation_mask.shape[1]
    full = torch.zeros(
        (batch_size, context_length + branch_length, context_length + branch_length),
        dtype=torch.bool,
        device=context_pad_mask.device,
    )
    full[:, :context_length, :context_length] = context_attention
    full[:, context_length:, :context_length] = (
        continuation_mask[:, :, None].bool() & context_pad_mask[:, None, :].bool()
    )
    causal = torch.ones(
        (branch_length, branch_length), dtype=torch.bool, device=full.device
    ).tril()
    full[:, context_length:, context_length:] = (
        continuation_mask[:, :, None].bool()
        & continuation_mask[:, None, :].bool()
        & causal[None]
    )
    context_positions = torch.cumsum(context_pad_mask, dim=1) - 1
    branch_positions = (
        context_pad_mask.sum(dim=1, keepdim=True)
        + torch.cumsum(continuation_mask, dim=1)
        - 1
    )
    return full, torch.cat([context_positions, branch_positions], dim=1).clamp_min(0)


class ActionFlowV23Model(VLAFlowMatching):
    """Three VLM objectives plus a detached plan-conditioned flow expert."""

    def __init__(self, config: ActionFlowV23Config, rtc_processor=None):
        compile_model = config.compile_model
        config.compile_model = False
        try:
            super().__init__(config, rtc_processor=rtc_processor)
        finally:
            config.compile_model = compile_model

        artifact = resolve_tokenizer_artifact_path(config)
        self.target_tokenizer = ActionFlowV23TargetTokenizer.from_pretrained(
            artifact, config=config
        )
        self.vlm_with_expert.processor.tokenizer = self.target_tokenizer.vlm_tokenizer
        resize_vlm_token_embeddings(self.vlm_with_expert, config)
        self._enable_joint_training_parameters()
        self._plan_candidate_sequences = tuple(
            [
                self.target_tokenizer.serialize_plan(stage=stage, target_entity=entity)
                for stage in ("grasp", "put")
                for entity in self.target_tokenizer.entity_names
            ]
            + [self.target_tokenizer.serialize_plan(stage="complete", target_entity="")]
        )
        self.last_plan_token_ids: Tensor | None = None
        self.last_grounding_token_ids: Tensor | None = None

        if compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

    def _enable_joint_training_parameters(self) -> None:
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
        targets: PackedAutoregressiveTargetsV23,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[int, dict[str, Tensor]]]:
        batch_size = prefix_embeddings.shape[0]
        device = prefix_embeddings.device
        branch_specs = (
            ("<plan>", targets.plan_token_ids, targets.plan_token_mask),
            ("<grounding>", targets.grounding_token_ids, targets.grounding_token_mask),
            ("<action>", targets.action_token_ids, targets.action_token_mask),
        )
        decoder_embeddings = []
        decoder_masks = []
        for start_token, token_ids, token_mask in branch_specs:
            start_ids = torch.full(
                (batch_size, 1),
                self.config.structural_token_id(start_token),
                dtype=torch.long,
                device=device,
            )
            decoder_embeddings.append(
                torch.cat(
                    [
                        self.vlm_with_expert.embed_language_tokens(start_ids),
                        self.vlm_with_expert.embed_language_tokens(token_ids),
                    ],
                    dim=1,
                )
            )
            decoder_masks.append(
                torch.cat(
                    [torch.ones((batch_size, 1), dtype=torch.bool, device=device), token_mask.bool()],
                    dim=1,
                )
            )

        scale = math.sqrt(prefix_embeddings.shape[-1])
        full_embeddings = torch.cat(
            [prefix_embeddings, *(embeddings * scale for embeddings in decoder_embeddings)],
            dim=1,
        )
        attention, positions = build_hierarchical_attention_mask(
            prefix_pad_mask,
            prefix_attention_groups,
            decoder_masks[0],
            decoder_masks[1],
            decoder_masks[2],
        )
        (full_output, _), full_cache = self.vlm_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[full_embeddings, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        prefix_length = prefix_embeddings.shape[1]
        target_lengths = (
            targets.plan_token_ids.shape[1],
            targets.grounding_token_ids.shape[1],
            targets.action_token_ids.shape[1],
        )
        prediction_hidden = []
        cursor = prefix_length
        for target_length in target_lengths:
            prediction_hidden.append(
                torch.cat(
                    [
                        full_output[:, cursor : cursor + 1],
                        full_output[:, cursor + 1 : cursor + target_length],
                    ],
                    dim=1,
                )
            )
            cursor += 1 + target_length
        all_logits = self.vlm_with_expert.vlm.get_output_embeddings()(
            torch.cat(prediction_hidden, dim=1)
        )
        plan_stop = target_lengths[0]
        grounding_stop = plan_stop + target_lengths[1]
        plan_logits = all_logits[:, :plan_stop]
        grounding_logits = all_logits[:, plan_stop:grounding_stop]
        action_logits = all_logits[:, grounding_stop:]

        plan_context_length = prefix_length + decoder_masks[0].shape[1]
        plan_context_mask = torch.cat([prefix_pad_mask.bool(), decoder_masks[0]], dim=1)
        plan_context_cache = slice_prefix_key_values(
            full_cache,
            prefix_length=plan_context_length,
            detach=self.config.detach_expert_prefix_kv,
        )
        return (
            plan_logits,
            grounding_logits,
            action_logits,
            plan_context_mask,
            plan_context_cache,
        )

    def forward_joint(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        plan_tokens: Tensor,
        plan_mask: Tensor,
        plan_loss_weight: Tensor,
        grounding_tokens: Tensor,
        grounding_mask: Tensor,
        grounding_loss_weight: Tensor,
        action_tokens: Tensor,
        action_token_mask: Tensor,
        action_token_loss_weight: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> ActionFlowV23ForwardOutput:
        """Compute plan, grounding, FAST, and flow objectives in one pass."""

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        target_velocity = noise - actions

        prefix_embeddings, prefix_pad_mask, prefix_attention_groups = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        packed = pack_autoregressive_targets_v2_3(
            plan_tokens=plan_tokens,
            plan_mask=plan_mask,
            plan_loss_weight=plan_loss_weight,
            grounding_tokens=grounding_tokens,
            grounding_mask=grounding_mask,
            grounding_loss_weight=grounding_loss_weight,
            action_tokens=action_tokens,
            action_mask=action_token_mask,
            action_loss_weight=action_token_loss_weight,
            pad_token_id=self.target_tokenizer.vlm_tokenizer.pad_token_id,
            plan_start_token_id=self.config.structural_token_id("<plan>"),
            grounding_start_token_id=self.config.structural_token_id("<grounding>"),
            action_start_token_id=self.config.structural_token_id("<action>"),
        )
        plan_logits, grounding_logits, action_logits, context_mask, context_cache = (
            self._autoregressive_vlm_pass(
                prefix_embeddings, prefix_pad_mask, prefix_attention_groups, packed
            )
        )
        plan_loss = weighted_token_cross_entropy(
            plan_logits, packed.plan_token_ids, packed.plan_loss_weight
        )
        box_loss = weighted_token_cross_entropy(
            grounding_logits,
            packed.grounding_token_ids,
            packed.grounding_loss_weight,
        )
        fast_loss = weighted_token_cross_entropy(
            action_logits, packed.action_token_ids, packed.action_loss_weight
        )
        predicted_velocity = self.denoise_step(
            prefix_pad_masks=context_mask,
            past_key_values=context_cache,
            x_t=x_t,
            timestep=time,
        )
        return ActionFlowV23ForwardOutput(
            flow_losses=F.mse_loss(target_velocity, predicted_velocity, reduction="none"),
            plan_loss=plan_loss,
            box_loss=box_loss,
            fast_loss=fast_loss,
        )

    def _greedy_decode_plan(
        self,
        prefix_embeddings: Tensor,
        prefix_pad_mask: Tensor,
        prefix_attention_groups: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = prefix_embeddings.shape[0]
        device = prefix_embeddings.device
        pad_id = self.target_tokenizer.vlm_tokenizer.pad_token_id
        end_id = self.config.structural_token_id("</plan>")
        ids = torch.full(
            (batch_size, 1),
            self.config.structural_token_id("<plan>"),
            dtype=torch.long,
            device=device,
        )
        mask = torch.ones_like(ids, dtype=torch.bool)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        scale = math.sqrt(prefix_embeddings.shape[-1])
        for _ in range(self.config.max_plan_tokens - 1):
            decoder_embeddings = self.vlm_with_expert.embed_language_tokens(ids) * scale
            embeddings = torch.cat([prefix_embeddings, decoder_embeddings], dim=1)
            attention, positions = self._prefix_decoder_attention(
                prefix_pad_mask, prefix_attention_groups, mask
            )
            (output, _), _ = self.vlm_with_expert.forward(
                attention_mask=attention,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[embeddings, None],
                use_cache=True,
                fill_kv_cache=True,
            )
            logits = self.vlm_with_expert.vlm.get_output_embeddings()(output[:, -1])
            next_ids = torch.full((batch_size,), pad_id, dtype=torch.long, device=device)
            for sample_index in range(batch_size):
                if finished[sample_index]:
                    continue
                prefix = ids[sample_index, mask[sample_index]].tolist()
                allowed = sorted(
                    {
                        sequence[len(prefix)]
                        for sequence in self._plan_candidate_sequences
                        if len(sequence) > len(prefix) and sequence[: len(prefix)] == prefix
                    }
                )
                if not allowed:
                    raise RuntimeError(f"Generated plan prefix is outside the plan grammar: {prefix}.")
                allowed_ids = torch.tensor(allowed, dtype=torch.long, device=device)
                selected = logits[sample_index, allowed_ids].argmax()
                next_ids[sample_index] = allowed_ids[selected]
            active = ~finished
            ids = torch.cat([ids, next_ids[:, None]], dim=1)
            mask = torch.cat([mask, active[:, None]], dim=1)
            finished = finished | (active & (next_ids == end_id))
            if bool(finished.all()):
                break
        if not bool(finished.all()):
            raise RuntimeError(
                f"Plan grammar did not terminate within max_plan_tokens={self.config.max_plan_tokens}."
            )
        return ids, mask

    @staticmethod
    def _prefix_decoder_attention(
        prefix_pad_mask: Tensor,
        prefix_attention_groups: Tensor,
        decoder_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        empty = decoder_mask.new_zeros((decoder_mask.shape[0], 0))
        return build_hierarchical_attention_mask(
            prefix_pad_mask,
            prefix_attention_groups,
            decoder_mask,
            empty,
            empty,
        )

    def _cache_plan_context(
        self,
        prefix_embeddings: Tensor,
        prefix_pad_mask: Tensor,
        prefix_attention_groups: Tensor,
        plan_ids: Tensor,
        plan_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, dict[int, dict[str, Tensor]]]:
        scale = math.sqrt(prefix_embeddings.shape[-1])
        plan_embeddings = self.vlm_with_expert.embed_language_tokens(plan_ids) * scale
        context_embeddings = torch.cat([prefix_embeddings, plan_embeddings], dim=1)
        attention, positions = self._prefix_decoder_attention(
            prefix_pad_mask, prefix_attention_groups, plan_mask
        )
        (_, _), cache = self.vlm_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[context_embeddings, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        context_mask = torch.cat([prefix_pad_mask.bool(), plan_mask.bool()], dim=1)
        return context_embeddings, attention, context_mask, cache

    def _greedy_decode_grounding(
        self,
        context_embeddings: Tensor,
        context_attention: Tensor,
        context_mask: Tensor,
    ) -> Tensor:
        batch_size = context_embeddings.shape[0]
        device = context_embeddings.device
        pad_id = self.target_tokenizer.vlm_tokenizer.pad_token_id
        end_id = self.config.structural_token_id("</grounding>")
        ids = torch.full(
            (batch_size, 1),
            self.config.structural_token_id("<grounding>"),
            dtype=torch.long,
            device=device,
        )
        mask = torch.ones_like(ids, dtype=torch.bool)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        scale = math.sqrt(context_embeddings.shape[-1])
        for step in range(self.config.max_grounding_tokens - 1):
            branch_embeddings = self.vlm_with_expert.embed_language_tokens(ids) * scale
            embeddings = torch.cat([context_embeddings, branch_embeddings], dim=1)
            attention, positions = _build_context_continuation_attention(
                context_attention, context_mask, mask
            )
            (output, _), _ = self.vlm_with_expert.forward(
                attention_mask=attention,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[embeddings, None],
                use_cache=True,
                fill_kv_cache=True,
            )
            next_ids = self.vlm_with_expert.vlm.get_output_embeddings()(output[:, -1]).argmax(-1)
            if step == self.config.max_grounding_tokens - 2:
                next_ids = torch.where(finished, next_ids, torch.full_like(next_ids, end_id))
            active = ~finished
            next_ids = torch.where(active, next_ids, torch.full_like(next_ids, pad_id))
            ids = torch.cat([ids, next_ids[:, None]], dim=1)
            mask = torch.cat([mask, active[:, None]], dim=1)
            finished = finished | (active & (next_ids == end_id))
            if bool(finished.all()):
                break
        return ids

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs,
    ) -> Tensor:
        """Predict a plan once, then condition flow denoising on its detached K/V."""

        batch_size = state.shape[0]
        device = state.device
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim), device
            )
        prefix_embeddings, prefix_pad_mask, prefix_attention_groups = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        plan_ids, plan_mask = self._greedy_decode_plan(
            prefix_embeddings, prefix_pad_mask, prefix_attention_groups
        )
        context_embeddings, context_attention, context_mask, cache = self._cache_plan_context(
            prefix_embeddings,
            prefix_pad_mask,
            prefix_attention_groups,
            plan_ids,
            plan_mask,
        )
        self.last_plan_token_ids = plan_ids.detach().cpu()
        self.last_grounding_token_ids = None
        if self.config.generate_grounding_during_rollout:
            self.last_grounding_token_ids = self._greedy_decode_grounding(
                context_embeddings, context_attention, context_mask
            ).detach().cpu()

        cache = {
            layer: {name: value.detach() for name, value in values.items()}
            for layer, values in cache.items()
        }
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time_value = 1.0 + step * dt
            timestep = torch.full(
                (batch_size,), time_value, dtype=torch.float32, device=device
            )

            def denoise(input_x_t, current_timestep=timestep):
                return self.denoise_step(
                    prefix_pad_masks=context_mask,
                    past_key_values=cache,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                velocity = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                    inference_delay=kwargs.get("inference_delay"),
                    time=time_value,
                    original_denoise_step_partial=denoise,
                    execution_horizon=kwargs.get("execution_horizon"),
                )
            else:
                velocity = denoise(x_t)
            x_t = x_t + dt * velocity
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time_value, x_t=x_t, v_t=velocity)
        return x_t


class ActionFlowV23Policy(SmolVLAPolicy):
    """LeRobot wrapper for Action Flow V2.3 joint training and rollout."""

    config_class = ActionFlowV23Config
    name = "action_flow_v2_3"

    def __init__(
        self,
        config: ActionFlowV23Config,
        *,
        initialize_from_smolvla: bool = True,
        **kwargs,
    ):
        require_package("transformers", extra="smolvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = ActionFlowV23Model(config, rtc_processor=self.rtc_processor)
        self.smolvla_initialization_report = None
        if initialize_from_smolvla and config.smolvla_pretrained_path is not None:
            self.smolvla_initialization_report = initialize_from_smolvla_pretrained(
                self, config.smolvla_pretrained_path
            )
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: ActionFlowV23Config | None = None,
        **kwargs,
    ) -> ActionFlowV23Policy:
        if config is None:
            keys = (
                "force_download",
                "resume_download",
                "proxies",
                "token",
                "cache_dir",
                "local_files_only",
                "revision",
            )
            loaded_config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path,
                **{key: kwargs[key] for key in keys if key in kwargs},
            )
            if not isinstance(loaded_config, ActionFlowV23Config):
                raise TypeError(
                    "ActionFlowV23Policy requires an action_flow_v2_3 checkpoint, got "
                    f"{loaded_config.type!r}."
                )
            config = loaded_config
        config.pretrained_path = Path(pretrained_name_or_path)
        kwargs["initialize_from_smolvla"] = False
        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)

    def prepare_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        prepared = dict(batch)
        for key in self.config.image_features:
            image = prepared.get(key)
            if isinstance(image, torch.Tensor) and not image.is_floating_point():
                prepared[key] = image.float().div(255.0)
        return super().prepare_images(prepared)

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float]]:
        """Train planning, box, FAST, and segment-aware flow objectives jointly."""

        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction {reduction!r}.")
        required = (
            PLAN_TOKENS,
            PLAN_TOKEN_MASK,
            PLAN_TOKEN_LOSS_WEIGHT,
            GROUNDING_TOKENS,
            GROUNDING_TOKEN_MASK,
            GROUNDING_TOKEN_LOSS_WEIGHT,
            ACTION_TOKENS,
            ACTION_TOKEN_MASK,
            ACTION_TOKEN_LOSS_WEIGHT,
            SEGMENT_ACTION_MASK,
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Action Flow V2.3 preprocessor did not produce {missing}.")

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
            plan_tokens=batch[PLAN_TOKENS],
            plan_mask=batch[PLAN_TOKEN_MASK],
            plan_loss_weight=batch[PLAN_TOKEN_LOSS_WEIGHT],
            grounding_tokens=batch[GROUNDING_TOKENS],
            grounding_mask=batch[GROUNDING_TOKEN_MASK],
            grounding_loss_weight=batch[GROUNDING_TOKEN_LOSS_WEIGHT],
            action_tokens=batch[ACTION_TOKENS],
            action_token_mask=batch[ACTION_TOKEN_MASK],
            action_token_loss_weight=batch[ACTION_TOKEN_LOSS_WEIGHT],
            noise=noise,
            time=time,
        )

        action_dim = self.config.action_feature.shape[0]
        flow_losses = outputs.flow_losses[:, :, :action_dim]
        valid = batch[SEGMENT_ACTION_MASK].bool()
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None:
            valid = valid & ~action_is_pad.bool()
        masked = flow_losses * valid.unsqueeze(-1)
        valid_count = (valid.sum(dim=1) * action_dim).clamp_min(1)
        flow_loss = masked.sum(dim=(1, 2)) / valid_count

        total_loss = (
            self.config.plan_loss_weight * outputs.plan_loss
            + self.config.box_loss_weight * outputs.box_loss
            + self.config.fast_loss_weight * outputs.fast_loss
            + self.config.flow_loss_weight * flow_loss
        )
        metrics = {
            "loss": total_loss.detach().mean().item(),
            "plan_ce_loss": outputs.plan_loss.detach().mean().item(),
            "box_ce_loss": outputs.box_loss.detach().mean().item(),
            "fast_ce_loss": outputs.fast_loss.detach().mean().item(),
            "flow_loss": flow_loss.detach().mean().item(),
            "segment_valid_fraction": valid.float().mean().item(),
        }
        return (total_loss, metrics) if reduction == "none" else (total_loss.mean(), metrics)

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
        raise NotImplementedError("PEFT targets are not defined for Action Flow V2.3.")


__all__ = [
    "ActionFlowV23ForwardOutput",
    "ActionFlowV23Model",
    "ActionFlowV23Policy",
    "PackedAutoregressiveTargetsV23",
    "build_hierarchical_attention_mask",
    "pack_autoregressive_targets_v2_3",
]
