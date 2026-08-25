#!/usr/bin/env python

"""PI0.5-based Action Flow V2.4 with Plan, box, FAST, and flow supervision."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.import_utils import require_package

from ..pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from ..pretrained import PreTrainedPolicy
from .configuration_action_flow_v2_4 import ActionFlowV24Config
from .data_contracts_action_flow_v2_4 import (
    BOX_TOKENS,
    BOX_TOKEN_LOSS_WEIGHT,
    BOX_TOKEN_MASK,
    FAST_TOKENS,
    FAST_TOKEN_LOSS_WEIGHT,
    FAST_TOKEN_MASK,
    PLAN_TOKENS,
    PLAN_TOKEN_LOSS_WEIGHT,
    PLAN_TOKEN_MASK,
    SEGMENT_ACTION_MASK,
)
from .tokenization_action_flow_v2_4 import ActionFlowV24TargetTokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionFlowV24ForwardOutput:
    flow_losses: Tensor
    plan_loss: Tensor
    box_loss: Tensor
    fast_loss: Tensor
    plan_token_accuracy: Tensor
    box_token_accuracy: Tensor
    fast_token_accuracy: Tensor
    fast_token_top5_accuracy: Tensor
    fast_sequence_exact_accuracy: Tensor


def _causal_block(mask: Tensor) -> Tensor:
    length = mask.shape[1]
    causal = torch.ones((length, length), dtype=torch.bool, device=mask.device).tril()
    return causal.unsqueeze(0) & mask[:, :, None] & mask[:, None, :]


def build_action_flow_v2_4_attention_mask(
    shared_mask: Tensor,
    plan_mask: Tensor,
    box_mask: Tensor,
    fast_mask: Tensor,
    action_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build the AFV2.4 Plan -> Box/FAST/Action dependency mask.

    Sequence layout is ``[shared prefix, Plan, Boxes, FAST, action expert]``.
    Plan is causal over itself. Boxes, FAST, and the action expert attend the
    complete Plan but never one another. Box and FAST branches are causal over
    themselves; the action chunk remains bidirectional like the PI0.5 expert.
    """

    masks = [
        shared_mask.bool(),
        plan_mask.bool(),
        box_mask.bool(),
        fast_mask.bool(),
        action_mask.bool(),
    ]
    if any(mask.ndim != 2 for mask in masks):
        raise ValueError("All AFV2.4 sequence masks must have shape (batch_size, sequence_length).")
    if len({mask.shape[0] for mask in masks}) != 1:
        raise ValueError("All AFV2.4 sequence masks must have the same batch size.")

    batch_size = shared_mask.shape[0]
    lengths = [mask.shape[1] for mask in masks]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    attention = torch.zeros(
        (batch_size, offsets[-1], offsets[-1]),
        dtype=torch.bool,
        device=shared_mask.device,
    )
    shared = slice(offsets[0], offsets[1])
    plan = slice(offsets[1], offsets[2])
    boxes = slice(offsets[2], offsets[3])
    fast = slice(offsets[3], offsets[4])
    actions = slice(offsets[4], offsets[5])

    attention[:, shared, shared] = shared_mask[:, :, None] & shared_mask[:, None, :]
    attention[:, plan, shared] = plan_mask[:, :, None] & shared_mask[:, None, :]
    attention[:, plan, plan] = _causal_block(plan_mask)
    attention[:, boxes, shared] = box_mask[:, :, None] & shared_mask[:, None, :]
    attention[:, boxes, plan] = box_mask[:, :, None] & plan_mask[:, None, :]
    attention[:, boxes, boxes] = _causal_block(box_mask)
    attention[:, fast, shared] = fast_mask[:, :, None] & shared_mask[:, None, :]
    attention[:, fast, plan] = fast_mask[:, :, None] & plan_mask[:, None, :]
    attention[:, fast, fast] = _causal_block(fast_mask)
    attention[:, actions, shared] = action_mask[:, :, None] & shared_mask[:, None, :]
    attention[:, actions, plan] = action_mask[:, :, None] & plan_mask[:, None, :]
    attention[:, actions, actions] = action_mask[:, :, None] & action_mask[:, None, :]

    shared_positions = torch.cumsum(shared_mask.long(), dim=1) - 1
    shared_offset = shared_mask.long().sum(dim=1, keepdim=True)
    plan_positions = shared_offset + torch.cumsum(plan_mask.long(), dim=1) - 1
    branch_offset = shared_offset + plan_mask.long().sum(dim=1, keepdim=True)
    box_positions = branch_offset + torch.cumsum(box_mask.long(), dim=1) - 1
    fast_positions = branch_offset + torch.cumsum(fast_mask.long(), dim=1) - 1
    action_positions = branch_offset + torch.cumsum(action_mask.long(), dim=1) - 1
    positions = torch.cat(
        [
            shared_positions,
            plan_positions,
            box_positions,
            fast_positions,
            action_positions,
        ],
        dim=1,
    ).clamp_min(0)
    return attention, positions


def weighted_token_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return per-sample weighted token CE and accuracy."""

    if logits.shape[:2] != targets.shape or targets.shape != weights.shape:
        raise ValueError(
            f"Token shapes do not match: logits={logits.shape}, targets={targets.shape}, "
            f"weights={weights.shape}."
        )
    losses = F.cross_entropy(logits.transpose(1, 2).float(), targets, reduction="none")
    weights = weights.to(losses.dtype)
    denominator = weights.sum(dim=1)
    per_sample_loss = (losses * weights).sum(dim=1) / denominator.clamp_min(1.0)
    correct = (logits.argmax(dim=-1) == targets).to(losses.dtype)
    per_sample_accuracy = (correct * weights).sum(dim=1) / denominator.clamp_min(1.0)
    valid = denominator > 0
    return torch.where(valid, per_sample_loss, torch.zeros_like(per_sample_loss)), torch.where(
        valid, per_sample_accuracy, torch.zeros_like(per_sample_accuracy)
    )


def _trim_token_target(
    token_ids: Tensor,
    token_mask: Tensor,
    loss_weight: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Remove batch-wide right padding while retaining one next-token pair."""

    valid_length = max(2, int(token_mask.long().sum(dim=1).max().item()))
    return (
        token_ids[:, :valid_length],
        token_mask[:, :valid_length],
        loss_weight[:, :valid_length],
    )


def initialize_from_pi05_pretrained(
    policy: "ActionFlowV24Policy",
    pretrained_path: str,
) -> dict[str, Any]:
    """Load all shape-compatible PI0.5 weights without constructing a second model."""

    from safetensors.torch import load_file
    from transformers.utils import cached_file

    source = Path(pretrained_path).expanduser()
    if source.is_dir():
        model_file = source / "model.safetensors"
    else:
        resolved = cached_file(pretrained_path, "model.safetensors")
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve PI0.5 weights from {pretrained_path!r}.")
        model_file = Path(resolved)
    if not model_file.is_file():
        raise FileNotFoundError(f"PI0.5 model weights were not found at {model_file}.")

    source_state = load_file(str(model_file), device="cpu")
    fixed_state = policy._fix_pytorch_state_dict_keys(source_state, policy.config)
    remapped = {
        key if key.startswith("model.") else f"model.{key}": value
        for key, value in fixed_state.items()
    }
    target_state = policy.state_dict()
    compatible = {
        key: value
        for key, value in remapped.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    if not compatible:
        raise RuntimeError(f"No PI0.5 parameters matched Action Flow V2.4 from {pretrained_path!r}.")
    missing, unexpected = policy.load_state_dict(compatible, strict=False)
    loaded_elements = sum(value.numel() for value in compatible.values())
    target_elements = sum(value.numel() for value in target_state.values())
    report = {
        "source": pretrained_path,
        "loaded_tensors": len(compatible),
        "loaded_parameter_fraction": loaded_elements / max(target_elements, 1),
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
    }
    if report["loaded_parameter_fraction"] < 0.9:
        raise RuntimeError(
            "PI0.5 initialization matched too little of Action Flow V2.4: "
            f"{report['loaded_parameter_fraction']:.2%}."
        )
    logger.info("Initialized Action Flow V2.4 from PI0.5: %s", report)
    return report


class ActionFlowV24Model(PI05Pytorch):
    """PI0.5 core with teacher-forced Plan, grounding, and FAST branches."""

    def __init__(
        self,
        config: ActionFlowV24Config,
        target_tokenizer: ActionFlowV24TargetTokenizer,
        rtc_processor=None,
    ) -> None:
        super().__init__(config, rtc_processor=rtc_processor)
        self.config = config
        self.target_tokenizer = target_tokenizer
        self.last_plan_token_ids: Tensor | None = None
        self.last_box_token_ids: Tensor | None = None

    def _target_embeddings(self, token_ids: Tensor) -> Tensor:
        return self.paligemma_with_expert.embed_language_tokens(token_ids)

    def _lm_logits(self, hidden_states: Tensor) -> Tensor:
        return self.paligemma_with_expert.paligemma.lm_head(hidden_states)

    def _training_token_loss(
        self,
        hidden_states: Tensor,
        token_ids: Tensor,
        token_mask: Tensor,
        loss_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits = self._lm_logits(hidden_states[:, :-1])
        targets = token_ids[:, 1:]
        weights = loss_weight[:, 1:] * token_mask[:, 1:].to(loss_weight.dtype)
        return weighted_token_cross_entropy(logits, targets, weights)

    def forward_joint(
        self,
        *,
        images,
        img_masks,
        tokens: Tensor,
        masks: Tensor,
        actions: Tensor,
        plan_tokens: Tensor,
        plan_mask: Tensor,
        plan_loss_weight: Tensor,
        box_tokens: Tensor,
        box_mask: Tensor,
        box_loss_weight: Tensor,
        fast_tokens: Tensor,
        fast_mask: Tensor,
        fast_loss_weight: Tensor,
        noise: Tensor,
        time: Tensor,
    ) -> ActionFlowV24ForwardOutput:
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions
        target_velocity = noise - actions

        shared_embs, shared_mask, _ = self.embed_prefix(images, img_masks, tokens, masks)
        plan_embs = self._target_embeddings(plan_tokens)
        box_embs = self._target_embeddings(box_tokens)
        fast_embs = self._target_embeddings(fast_tokens)
        suffix_embs, suffix_mask, _, adarms_cond = self.embed_suffix(x_t, time)
        model_dtype = (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
        )
        shared_embs = shared_embs.to(model_dtype)
        plan_embs = plan_embs.to(model_dtype)
        box_embs = box_embs.to(model_dtype)
        fast_embs = fast_embs.to(model_dtype)
        suffix_embs = suffix_embs.to(model_dtype)
        prefix_embs = torch.cat([shared_embs, plan_embs, box_embs, fast_embs], dim=1)

        attention, positions = build_action_flow_v2_4_attention_mask(
            shared_mask, plan_mask, box_mask, fast_mask, suffix_mask
        )
        additive_attention = self._prepare_attention_masks_4d(attention)
        (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=additive_attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        shared_length = shared_embs.shape[1]
        plan_length = plan_embs.shape[1]
        box_length = box_embs.shape[1]
        plan_out = prefix_out[:, shared_length : shared_length + plan_length]
        box_start = shared_length + plan_length
        box_out = prefix_out[:, box_start : box_start + box_length]
        fast_out = prefix_out[:, box_start + box_length :]
        plan_loss, plan_accuracy = self._training_token_loss(
            plan_out, plan_tokens, plan_mask, plan_loss_weight
        )
        box_loss, box_accuracy = self._training_token_loss(
            box_out, box_tokens, box_mask, box_loss_weight
        )
        (
            fast_loss,
            fast_accuracy,
            fast_top5_accuracy,
            fast_exact_accuracy,
        ) = self._training_fast_token_loss(
            fast_out,
            fast_tokens,
            fast_mask,
            fast_loss_weight,
        )
        velocity = self.action_out_proj(suffix_out[:, -self.config.chunk_size :].float())
        return ActionFlowV24ForwardOutput(
            flow_losses=F.mse_loss(target_velocity, velocity, reduction="none"),
            plan_loss=plan_loss,
            box_loss=box_loss,
            fast_loss=fast_loss,
            plan_token_accuracy=plan_accuracy,
            box_token_accuracy=box_accuracy,
            fast_token_accuracy=fast_accuracy,
            fast_token_top5_accuracy=fast_top5_accuracy,
            fast_sequence_exact_accuracy=fast_exact_accuracy,
        )

    def _training_fast_token_loss(
        self,
        hidden_states: Tensor,
        token_ids: Tensor,
        token_mask: Tensor,
        loss_weight: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute FAST CE without materializing full-vocabulary logits for action symbols."""

        hidden = hidden_states[:, :-1]
        targets = token_ids[:, 1:]
        weights = loss_weight[:, 1:] * token_mask[:, 1:].to(loss_weight.dtype)
        active = weights > 0
        losses = hidden.new_zeros(targets.shape, dtype=torch.float32)
        predictions = torch.full_like(targets, -1)
        top5_correct = torch.zeros_like(active)
        start = self.config.fast_token_start_id
        stop = self.config.fast_token_stop_id
        fast_positions = active & (targets >= start) & (targets < stop)
        text_positions = active & ~fast_positions
        lm_head = self.paligemma_with_expert.paligemma.lm_head

        def accumulate(position_mask: Tensor, *, restrict: bool) -> None:
            if not bool(position_mask.any()):
                return
            selected_hidden = hidden[position_mask]
            selected_targets = targets[position_mask]
            if restrict:
                logits = F.linear(selected_hidden, lm_head.weight[start:stop])
                local_targets = selected_targets - start
                predicted = logits.argmax(dim=-1) + start
                top5 = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices + start
            else:
                logits = lm_head(selected_hidden)
                local_targets = selected_targets
                predicted = logits.argmax(dim=-1)
                top5 = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
            losses[position_mask] = F.cross_entropy(
                logits.float(), local_targets, reduction="none"
            )
            predictions[position_mask] = predicted
            top5_correct[position_mask] = (top5 == selected_targets[:, None]).any(dim=-1)

        if self.config.restrict_fast_vocabulary:
            accumulate(fast_positions, restrict=True)
            accumulate(text_positions, restrict=False)
        else:
            accumulate(active, restrict=False)

        denominator = weights.sum(dim=1)
        valid = denominator > 0
        per_sample_loss = (losses * weights).sum(dim=1) / denominator.clamp_min(1.0)
        correct = (predictions == targets).to(weights.dtype)
        per_sample_accuracy = (correct * weights).sum(dim=1) / denominator.clamp_min(1.0)
        per_sample_top5 = (
            top5_correct.to(weights.dtype) * weights
        ).sum(dim=1) / denominator.clamp_min(1.0)
        per_sample_exact = ((predictions == targets) | ~active).all(dim=1).to(weights.dtype)
        zero = torch.zeros_like(per_sample_loss)
        return (
            torch.where(valid, per_sample_loss, zero),
            torch.where(valid, per_sample_accuracy, zero),
            torch.where(valid, per_sample_top5, zero),
            torch.where(valid, per_sample_exact, zero),
        )

    def _context_attention(
        self,
        shared_mask: Tensor,
        plan_mask: Tensor,
        box_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if box_mask is None:
            box_mask = shared_mask.new_zeros((shared_mask.shape[0], 0))
        empty_fast = shared_mask.new_zeros((shared_mask.shape[0], 0))
        empty_actions = shared_mask.new_zeros((shared_mask.shape[0], 0))
        return build_action_flow_v2_4_attention_mask(
            shared_mask, plan_mask, box_mask, empty_fast, empty_actions
        )

    def _greedy_decode_plan(
        self,
        shared_embs: Tensor,
        shared_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        candidates = self.target_tokenizer.plan_candidate_sequences
        if not candidates:
            raise RuntimeError("Action Flow V2.4 has no valid Plan candidates.")
        batch_size = shared_embs.shape[0]
        device = shared_embs.device
        prefix = list(self.target_tokenizer.plan_prefix_ids)
        ids = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        plan_mask = torch.ones_like(ids, dtype=torch.bool)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        eos_id = self.target_tokenizer.eos_token_id
        pad_id = self.target_tokenizer.pad_token_id

        for _ in range(self.config.max_plan_tokens - len(prefix)):
            plan_embs = self._target_embeddings(ids).to(shared_embs.dtype)
            context_embs = torch.cat([shared_embs, plan_embs], dim=1)
            attention, positions = self._context_attention(shared_mask, plan_mask)
            (context_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=self._prepare_attention_masks_4d(attention),
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[context_embs, None],
                use_cache=False,
            )
            logits = self._lm_logits(context_out[:, -1])
            next_ids = torch.full((batch_size,), pad_id, dtype=torch.long, device=device)
            for sample_index in range(batch_size):
                if finished[sample_index]:
                    continue
                current = tuple(ids[sample_index, plan_mask[sample_index]].tolist())
                allowed = sorted(
                    {
                        candidate[len(current)]
                        for candidate in candidates
                        if len(candidate) > len(current) and candidate[: len(current)] == current
                    }
                )
                if not allowed:
                    raise RuntimeError(f"Generated Plan prefix is outside the grammar: {current}.")
                allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
                next_ids[sample_index] = allowed_tensor[logits[sample_index, allowed_tensor].argmax()]
            active = ~finished
            ids = torch.cat([ids, next_ids[:, None]], dim=1)
            plan_mask = torch.cat([plan_mask, active[:, None]], dim=1)
            finished |= active & (next_ids == eos_id)
            if bool(finished.all()):
                break
        if not bool(finished.all()):
            raise RuntimeError(f"Plan did not terminate within {self.config.max_plan_tokens} tokens.")
        return ids, plan_mask

    def _cache_plan_context(
        self,
        shared_embs: Tensor,
        shared_mask: Tensor,
        plan_ids: Tensor,
        plan_mask: Tensor,
    ):
        plan_embs = self._target_embeddings(plan_ids).to(shared_embs.dtype)
        context_embs = torch.cat([shared_embs, plan_embs], dim=1)
        attention, positions = self._context_attention(shared_mask, plan_mask)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, cache = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(attention),
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[context_embs, None],
            use_cache=True,
        )
        return context_embs, torch.cat([shared_mask.bool(), plan_mask.bool()], dim=1), cache

    def _greedy_decode_boxes(
        self,
        context_embs: Tensor,
        shared_mask: Tensor,
        plan_mask: Tensor,
    ) -> Tensor:
        batch_size = context_embs.shape[0]
        device = context_embs.device
        prefix = list(self.target_tokenizer.box_prefix_ids)
        ids = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        box_mask = torch.ones_like(ids, dtype=torch.bool)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        eos_id = self.target_tokenizer.eos_token_id
        pad_id = self.target_tokenizer.pad_token_id
        for step in range(self.config.max_box_tokens - len(prefix)):
            box_embs = self._target_embeddings(ids).to(context_embs.dtype)
            embeddings = torch.cat([context_embs, box_embs], dim=1)
            attention, positions = self._context_attention(shared_mask, plan_mask, box_mask)
            (output, _), _ = self.paligemma_with_expert.forward(
                attention_mask=self._prepare_attention_masks_4d(attention),
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[embeddings, None],
                use_cache=False,
            )
            next_ids = self._lm_logits(output[:, -1]).argmax(dim=-1)
            if step == self.config.max_box_tokens - len(prefix) - 1:
                next_ids = torch.where(finished, next_ids, torch.full_like(next_ids, eos_id))
            active = ~finished
            next_ids = torch.where(active, next_ids, torch.full_like(next_ids, pad_id))
            ids = torch.cat([ids, next_ids[:, None]], dim=1)
            box_mask = torch.cat([box_mask, active[:, None]], dim=1)
            finished |= active & (next_ids == eos_id)
            if bool(finished.all()):
                break
        return ids

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs,
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        batch_size = tokens.shape[0]
        device = tokens.device
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim), device
            )

        shared_embs, shared_mask, _ = self.embed_prefix(images, img_masks, tokens, masks)
        model_dtype = (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
        )
        shared_embs = shared_embs.to(model_dtype)
        plan_ids, plan_mask = self._greedy_decode_plan(shared_embs, shared_mask)
        context_embs, context_mask, cache = self._cache_plan_context(
            shared_embs, shared_mask, plan_ids, plan_mask
        )
        self.last_plan_token_ids = plan_ids.detach().cpu()
        self.last_box_token_ids = None
        if self.config.generate_boxes_during_rollout:
            self.last_box_token_ids = self._greedy_decode_boxes(
                context_embs, shared_mask, plan_mask
            ).detach().cpu()

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


class ActionFlowV24Policy(PI05Policy):
    """LeRobot wrapper for joint PI0.5 flow, Plan, box, and FAST training."""

    config_class = ActionFlowV24Config
    name = "action_flow_v2_4"

    def __init__(
        self,
        config: ActionFlowV24Config,
        *,
        initialize_from_pi05: bool = True,
        **kwargs,
    ) -> None:
        require_package("transformers", extra="pi")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        target_tokenizer = ActionFlowV24TargetTokenizer(config)
        self.model = ActionFlowV24Model(config, target_tokenizer, rtc_processor=self.rtc_processor)
        self.pi05_initialization_report = None
        if initialize_from_pi05 and config.pi05_pretrained_path is not None:
            self.pi05_initialization_report = initialize_from_pi05_pretrained(
                self, config.pi05_pretrained_path
            )
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: ActionFlowV24Config | None = None,
        **kwargs,
    ) -> "ActionFlowV24Policy":
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
            if not isinstance(config, ActionFlowV24Config):
                raise TypeError(
                    "ActionFlowV24Policy requires an action_flow_v2_4 checkpoint, got "
                    f"{config.type!r}."
                )
        config.pretrained_path = Path(pretrained_name_or_path)
        kwargs["initialize_from_pi05"] = False
        return PreTrainedPolicy.from_pretrained.__func__(
            cls, pretrained_name_or_path, config=config, **kwargs
        )

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Any]]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction {reduction!r}.")
        required = (
            PLAN_TOKENS,
            PLAN_TOKEN_MASK,
            PLAN_TOKEN_LOSS_WEIGHT,
            BOX_TOKENS,
            BOX_TOKEN_MASK,
            BOX_TOKEN_LOSS_WEIGHT,
            FAST_TOKENS,
            FAST_TOKEN_MASK,
            FAST_TOKEN_LOSS_WEIGHT,
            SEGMENT_ACTION_MASK,
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Action Flow V2.4 preprocessor did not produce {missing}.")

        images, img_masks = self._preprocess_images(batch)
        actions = self.prepare_action(batch)
        plan_tokens, plan_mask, plan_loss_weight = _trim_token_target(
            batch[PLAN_TOKENS],
            batch[PLAN_TOKEN_MASK],
            batch[PLAN_TOKEN_LOSS_WEIGHT],
        )
        box_tokens, box_mask, box_loss_weight = _trim_token_target(
            batch[BOX_TOKENS],
            batch[BOX_TOKEN_MASK],
            batch[BOX_TOKEN_LOSS_WEIGHT],
        )
        fast_tokens, fast_mask, fast_loss_weight = _trim_token_target(
            batch[FAST_TOKENS],
            batch[FAST_TOKEN_MASK],
            batch[FAST_TOKEN_LOSS_WEIGHT],
        )
        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)
        outputs = self.model.forward_joint(
            images=images,
            img_masks=img_masks,
            tokens=batch[OBS_LANGUAGE_TOKENS],
            masks=batch[OBS_LANGUAGE_ATTENTION_MASK],
            actions=actions,
            plan_tokens=plan_tokens,
            plan_mask=plan_mask,
            plan_loss_weight=plan_loss_weight,
            box_tokens=box_tokens,
            box_mask=box_mask,
            box_loss_weight=box_loss_weight,
            fast_tokens=fast_tokens,
            fast_mask=fast_mask,
            fast_loss_weight=fast_loss_weight,
            noise=noise,
            time=time,
        )

        action_dim = self.config.output_features[ACTION].shape[0]
        flow_losses = outputs.flow_losses[:, :, :action_dim]
        if self.config.segment_aware_action_loss:
            action_mask = batch[SEGMENT_ACTION_MASK].bool()
        else:
            action_mask = torch.ones_like(batch[SEGMENT_ACTION_MASK], dtype=torch.bool)
        flow_weights = action_mask.unsqueeze(-1).to(flow_losses.dtype)
        flow_loss = (flow_losses * flow_weights).sum(dim=(1, 2)) / (
            flow_weights.sum(dim=(1, 2)) * action_dim
        ).clamp_min(1.0)
        total = (
            self.config.plan_loss_weight * outputs.plan_loss
            + self.config.box_loss_weight * outputs.box_loss
            + self.config.fast_loss_weight * outputs.fast_loss
            + self.config.flow_loss_weight * flow_loss
        )
        metrics = {
            "loss": total.mean().item(),
            "plan_ce_loss": outputs.plan_loss.mean().item(),
            "box_ce_loss": outputs.box_loss.mean().item(),
            "fast_ce_loss": outputs.fast_loss.mean().item(),
            "flow_loss": flow_loss.mean().item(),
            "plan_token_accuracy": outputs.plan_token_accuracy.mean().item(),
            "box_token_accuracy": outputs.box_token_accuracy.mean().item(),
            "fast_token_accuracy": outputs.fast_token_accuracy.mean().item(),
            "fast_token_top5_accuracy": outputs.fast_token_top5_accuracy.mean().item(),
            "fast_sequence_exact_accuracy": outputs.fast_sequence_exact_accuracy.mean().item(),
            "segment_valid_fraction": action_mask.float().mean().item(),
            "loss_per_dim": flow_losses.mean(dim=(0, 1)).detach().cpu().tolist(),
        }
        return (total if reduction == "none" else total.mean()), metrics


__all__ = [
    "ActionFlowV24ForwardOutput",
    "ActionFlowV24Model",
    "ActionFlowV24Policy",
    "build_action_flow_v2_4_attention_mask",
    "initialize_from_pi05_pretrained",
    "_trim_token_target",
    "weighted_token_cross_entropy",
]
