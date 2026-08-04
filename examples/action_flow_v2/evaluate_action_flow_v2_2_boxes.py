#!/usr/bin/env python
"""Quantitatively evaluate Action Flow V2.2's autoregressive box predictions.

The evaluator reads frames and box labels directly from a LeRobot dataset and
reports two different measurements:

1. Teacher-forced token metrics diagnose whether the VLM fits its serialized
   grounding targets when every previous target token is correct.
2. Free-running metrics greedily generate grounding tokens from images, task
   text, and robot state. These metrics reflect the boxes the VLM can actually
   produce without label leakage.

The output directory contains ``summary.json``, per-sample and per-target CSV
files, structured predictions, every evaluated frame's complete free-running
VLM output in ``vlm_outputs.jsonl``, and optional box-overlay previews. Dataset
episode indices are zero-based.

Example:

    python examples/action_flow_v2/evaluate_action_flow_v2_2_boxes.py \
        --episodes all \
        --frame-stride 30 \
        --max-samples 512 \
        --preview-count 20

Use ``--max-samples 0`` to evaluate every frame selected by ``--frame-stride``.
Free-running decoding is substantially slower than ordinary policy inference
because the current V2.2 VLM has no language-generation KV-cache path.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image, ImageDraw

CURRENT_FILE = Path(__file__).resolve()
WORKSPACE = CURRENT_FILE.parents[2]
LEROBOT_SRC = WORKSPACE / "repos" / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

# Use the workspace model cache prepared for training. Hugging Face Datasets
# creates lock files even for a fully local dataset, so keep only those
# transient files in /tmp instead of the possibly symlinked model cache.
os.environ.setdefault("HF_HOME", str(WORKSPACE / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/vla_studio_hf_datasets")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.action_flow_v2_2.configuration_action_flow_v2_2 import (  # noqa: E402
    ActionFlowV22Config,
)
from lerobot.policies.action_flow_v2_2.data_contracts_action_flow_v2_2 import (  # noqa: E402
    TARGET_BOX_BINS,
    TARGET_BOX_LOSS_WEIGHT,
    TARGET_BOX_VISIBILITY,
    TARGET_ENTITY_IDS,
    TARGET_ENTITY_ROLES,
    TARGET_ENTITY_VALID,
    TargetVisibility,
)
from lerobot.policies.action_flow_v2_2.modeling_action_flow_v2_2 import (  # noqa: E402
    ActionFlowV22Policy,
    PackedAutoregressiveTargets,
    build_joint_attention_mask,
)
from lerobot.policies.action_flow_v2_2.tokenization_action_flow_v2_2 import (  # noqa: E402
    DecodedTargetBox,
)
from lerobot.policies.smolvla.modeling_smolvla import (  # noqa: E402
    make_att_2d_masks,
    resize_with_pad,
)
from lerobot.utils.collate import lerobot_collate_fn  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

DEFAULT_DATASET_ID = "rm65_single_table_preparation_and_bussing_afv2_2_yoloe_boxes"
DEFAULT_DATASET_ROOT = WORKSPACE / "runs" / "dataset" / DEFAULT_DATASET_ID
DEFAULT_POLICY_PATH = (
    WORKSPACE
    / "runs"
    / "train"
    / (
        "rm65_action_flow_v22_single_table_preparation_and_bussing_afv2_2_yoloe_boxes_episodes_203_steps_80000_batch_8_chunk_50__smolvla_uninit_unresize"
    )
    / "checkpoints"
    / "080000"
    / "pretrained_model"
)
DEFAULT_OUTPUT_DIR = WORKSPACE / "runs" / "evaluation" / "action_flow_v2_2_boxes_uninit_smolvla_unresize_steps_80000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=f"local/{DEFAULT_DATASET_ID}")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--episodes",
        default="all",
        help="Zero-based episodes: 'all', '1,4,8', or Python-style ranges such as '0:20,30:40'.",
    )
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=512,
        help="Maximum evenly distributed selected frames; 0 keeps every selected frame.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "none", "bfloat16", "float16"),
        default="auto",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="Free-running grounding limit; 0 uses policy.max_grounding_tokens.",
    )
    parser.add_argument(
        "--stop-on-action-token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop an invalid grounding sequence if it enters the FAST action section early.",
    )
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {args.dataset_root}")
    if not args.policy_path.is_dir():
        raise FileNotFoundError(f"Policy checkpoint does not exist: {args.policy_path}")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")
    if args.max_samples < 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Sample/loader arguments must be non-negative and batch size positive.")
    if args.preview_count < 0 or args.max_new_tokens < 0:
        raise ValueError("Preview and generation limits cannot be negative.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "previews").mkdir(exist_ok=True)


def parse_episode_spec(spec: str, num_episodes: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(num_episodes))
    selected: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            selected.add(int(item))
            continue
        fields = item.split(":")
        if len(fields) not in (2, 3):
            raise ValueError(f"Invalid episode range: {item!r}")
        start = int(fields[0]) if fields[0] else 0
        stop = int(fields[1]) if fields[1] else num_episodes
        step = int(fields[2]) if len(fields) == 3 and fields[2] else 1
        selected.update(range(start, stop, step))
    invalid = sorted(index for index in selected if not 0 <= index < num_episodes)
    if invalid:
        raise ValueError(f"Episode indices outside [0, {num_episodes}): {invalid}")
    if not selected:
        raise ValueError("No episodes were selected.")
    return sorted(selected)


def select_frame_indices(
    dataset: LeRobotDataset,
    episodes: Iterable[int],
    frame_stride: int,
    max_samples: int,
) -> list[int]:
    candidates: list[int] = []
    episode_table = dataset.meta.episodes
    for episode in episodes:
        row = episode_table[int(episode)]
        start = int(row["dataset_from_index"])
        stop = int(row["dataset_to_index"])
        episode_indices = list(range(start, stop, frame_stride))
        if stop > start and (not episode_indices or episode_indices[-1] != stop - 1):
            episode_indices.append(stop - 1)
        candidates.extend(episode_indices)
    if max_samples and len(candidates) > max_samples:
        positions = np.linspace(0, len(candidates) - 1, max_samples, dtype=np.int64)
        candidates = [candidates[int(position)] for position in np.unique(positions)]
    return candidates


def tensor_scalar(value: Any) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def serialize_ground_truth(raw_batch: dict[str, Any], tokenizer: Any) -> tuple[list[list[int]], list[list[float]]]:
    batch_size = raw_batch[TARGET_BOX_BINS].shape[0]
    sequences: list[list[int]] = []
    weights: list[list[float]] = []
    for index in range(batch_size):
        token_ids, token_weights = tokenizer.serialize_grounding(
            target_box_bins=raw_batch[TARGET_BOX_BINS][index],
            target_box_visibility=raw_batch[TARGET_BOX_VISIBILITY][index],
            target_box_loss_weight=raw_batch[TARGET_BOX_LOSS_WEIGHT][index],
            target_entity_ids=raw_batch[TARGET_ENTITY_IDS][index],
            target_entity_roles=raw_batch[TARGET_ENTITY_ROLES][index],
            target_entity_valid=raw_batch[TARGET_ENTITY_VALID][index],
        )
        sequences.append(token_ids)
        weights.append(token_weights)
    return sequences, weights


def pad_grounding_targets(
    sequences: list[list[int]],
    weights: list[list[float]],
    *,
    pad_token_id: int,
    action_end_token_id: int,
    device: torch.device,
) -> PackedAutoregressiveTargets:
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Grounding sequences must be non-empty.")
    decoder_start = sequences[0][0]
    if any(sequence[0] != decoder_start for sequence in sequences):
        raise ValueError("Every grounding sequence must use the same decoder-start token.")
    target_sequences = [sequence[1:] for sequence in sequences]
    target_weights = [sequence_weights[1:] for sequence_weights in weights]
    max_length = max(map(len, target_sequences))
    batch_size = len(sequences)
    token_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long, device=device)
    token_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
    box_weights = torch.zeros((batch_size, max_length), dtype=torch.float32, device=device)
    for index, (sequence, sequence_weights) in enumerate(
        zip(target_sequences, target_weights, strict=True)
    ):
        length = len(sequence)
        token_ids[index, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        token_mask[index, :length] = True
        box_weights[index, :length] = torch.tensor(sequence_weights, dtype=torch.float32, device=device)
    return PackedAutoregressiveTargets(
        grounding_token_ids=token_ids,
        grounding_token_mask=token_mask,
        grounding_loss_weight=box_weights,
        action_token_ids=torch.full(
            (batch_size, 1),
            action_end_token_id,
            dtype=torch.long,
            device=device,
        ),
        action_token_mask=torch.ones((batch_size, 1), dtype=torch.bool, device=device),
    )


def autocast_context(device: torch.device, requested: str):
    if device.type != "cuda" or requested == "none":
        return contextlib.nullcontext()
    dtype_name = "bfloat16" if requested == "auto" else requested
    return torch.autocast(device_type="cuda", dtype=getattr(torch, dtype_name))


def prepare_prefix(policy: ActionFlowV22Policy, batch: dict[str, Any]):
    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    return policy.model.embed_prefix(
        images,
        image_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        state=state,
    )


def teacher_forced_metrics(
    policy: ActionFlowV22Policy,
    prefix: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    targets: PackedAutoregressiveTargets,
) -> list[dict[str, float | int]]:
    logits, action_logits, prefix_cache = policy.model._autoregressive_vlm_pass(
        *prefix,
        targets,
    )
    predictions = logits.argmax(dim=-1)
    token_losses = F.cross_entropy(
        logits.transpose(1, 2).float(),
        targets.grounding_token_ids,
        reduction="none",
    )
    valid = targets.grounding_token_mask
    correct = predictions.eq(targets.grounding_token_ids)
    config = policy.config
    structural_ids = {config.structural_token_id(token) for token in config.structural_tokens}
    coordinate_start = config.location_token_id(0)
    coordinate_stop = coordinate_start + config.coordinate_token_count

    results: list[dict[str, float | int]] = []
    for index in range(targets.grounding_token_ids.shape[0]):
        row: dict[str, float | int] = {}
        sample_valid = valid[index]
        sample_weights = targets.grounding_loss_weight[index] * sample_valid
        row["tf_token_correct"] = int((correct[index] & sample_valid).sum().item())
        row["tf_token_total"] = int(sample_valid.sum().item())
        row["tf_weighted_correct"] = float((correct[index].float() * sample_weights).sum().item())
        row["tf_weight_total"] = float(sample_weights.sum().item())
        row["tf_nll_sum"] = float((token_losses[index] * sample_weights).sum().item())
        row["tf_sequence_exact"] = int(bool(torch.all(correct[index][sample_valid]).item()))
        ids = targets.grounding_token_ids[index]
        structural = torch.tensor(
            [int(token_id) in structural_ids for token_id in ids.tolist()],
            dtype=torch.bool,
            device=ids.device,
        ) & sample_valid
        coordinate = ((ids >= coordinate_start) & (ids < coordinate_stop)) & sample_valid
        text = sample_valid & ~structural & ~coordinate
        for name, category_mask in (
            ("structural", structural),
            ("coordinate", coordinate),
            ("text", text),
        ):
            row[f"tf_{name}_correct"] = int((correct[index] & category_mask).sum().item())
            row[f"tf_{name}_total"] = int(category_mask.sum().item())
        results.append(row)
    del logits, action_logits, prefix_cache, predictions, token_losses
    return results


def next_token_logits(
    policy: ActionFlowV22Policy,
    prefix: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    generated: torch.Tensor,
    generated_lengths: torch.Tensor,
) -> torch.Tensor:
    prefix_embeddings, prefix_pad_mask, prefix_attention_groups = prefix
    batch_size, prefix_length = prefix_pad_mask.shape
    if generated.shape[1] == 0:
        embeddings = prefix_embeddings
        attention_mask = make_att_2d_masks(prefix_pad_mask.bool(), prefix_attention_groups)
        position_ids = torch.cumsum(prefix_pad_mask.bool(), dim=1) - 1
    else:
        target_mask = (
            torch.arange(generated.shape[1], device=generated.device)[None, :]
            < generated_lengths[:, None]
        )
        target_embeddings = policy.model.vlm_with_expert.embed_language_tokens(generated)
        target_embeddings = target_embeddings * math.sqrt(target_embeddings.shape[-1])
        embeddings = torch.cat([prefix_embeddings, target_embeddings], dim=1)
        attention_mask, position_ids = build_joint_attention_mask(
            prefix_pad_mask,
            prefix_attention_groups,
            target_mask,
        )
    (hidden, _), _ = policy.model.vlm_with_expert.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[embeddings, None],
        use_cache=False,
        fill_kv_cache=True,
    )
    if generated.shape[1] == 0:
        positions = torch.arange(prefix_length, device=hidden.device)[None].expand(batch_size, -1)
        last_positions = positions.masked_fill(~prefix_pad_mask.bool(), -1).max(dim=1).values
    else:
        last_positions = prefix_length + generated_lengths - 1
    prediction_hidden = hidden[
        torch.arange(batch_size, device=hidden.device),
        last_positions,
    ]
    logits = policy.model.vlm_with_expert.vlm.get_output_embeddings()(prediction_hidden)
    return logits


def greedy_generate_grounding(
    policy: ActionFlowV22Policy,
    prefix: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    max_new_tokens: int,
    stop_on_action_token: bool,
) -> tuple[list[list[int]], list[str]]:
    batch_size = prefix[0].shape[0]
    device = prefix[0].device
    tokenizer = policy.model.target_tokenizer
    pad_id = int(tokenizer.vlm_tokenizer.pad_token_id)
    grounding_start = policy.config.structural_token_id("<grounding>")
    grounding_end = policy.config.structural_token_id("</grounding>")
    action_start = policy.config.structural_token_id("<action>")
    action_end = policy.config.structural_token_id("<action_end>")
    fast_start = policy.config.fast_token_id(0)
    fast_stop = fast_start + policy.config.fast_token_count

    generated = torch.full(
        (batch_size, 1),
        grounding_start,
        dtype=torch.long,
        device=device,
    )
    lengths = torch.ones(batch_size, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    termination = ["max_tokens"] * batch_size
    for _ in range(max(0, max_new_tokens - 1)):
        logits = next_token_logits(policy, prefix, generated, lengths)
        next_ids = logits.argmax(dim=-1)
        del logits
        active = ~finished
        next_ids = torch.where(active, next_ids, torch.full_like(next_ids, pad_id))
        generated = torch.cat([generated, next_ids[:, None]], dim=1)
        lengths = lengths + active.long()

        for index in torch.nonzero(active, as_tuple=False).flatten().tolist():
            token_id = int(next_ids[index].item())
            if token_id == grounding_end:
                finished[index] = True
                termination[index] = "grounding_end"
            elif stop_on_action_token and (
                token_id in (action_start, action_end) or fast_start <= token_id < fast_stop
            ):
                finished[index] = True
                termination[index] = "entered_action_before_grounding_end"
        if bool(finished.all().item()):
            break
    sequences = [generated[index, : int(lengths[index].item())].tolist() for index in range(batch_size)]
    return sequences, termination


def box_key(box: DecodedTargetBox) -> tuple[str, str, int]:
    return box.camera, box.entity_name, int(box.entity_role)


def box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def decoded_box_dict(box: DecodedTargetBox) -> dict[str, Any]:
    return {
        "camera": box.camera,
        "entity_name": box.entity_name,
        "entity_role": box.entity_role.name.lower(),
        "visibility": box.visibility.name.lower(),
        "coordinate_bins": list(box.coordinate_bins) if box.coordinate_bins is not None else None,
    }


def token_sequence_payload(tokenizer: Any, token_ids: list[int]) -> dict[str, Any]:
    """Return token IDs, token strings, and readable decoded text for one sequence."""

    return {
        "token_ids": token_ids,
        "tokens": tokenizer.convert_ids_to_tokens(token_ids),
        "text": tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }


def match_targets(
    gt_boxes: list[DecodedTargetBox],
    predicted_boxes: list[DecodedTargetBox],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    predicted_by_key: dict[tuple[str, str, int], list[DecodedTargetBox]] = defaultdict(list)
    for box in predicted_boxes:
        predicted_by_key[box_key(box)].append(box)

    target_rows: list[dict[str, Any]] = []
    matched_count = 0
    visibility_correct_count = 0
    gt_visible_count = 0
    iou50_count = 0
    strict = True
    for gt in gt_boxes:
        key = box_key(gt)
        predicted = predicted_by_key[key].pop(0) if predicted_by_key[key] else None
        matched = predicted is not None
        matched_count += int(matched)
        visibility_correct = matched and predicted.visibility is gt.visibility
        visibility_correct_count += int(visibility_correct)
        gt_visible = gt.visibility is TargetVisibility.VISIBLE
        gt_visible_count += int(gt_visible)
        conditional_localization = bool(
            gt_visible
            and matched
            and predicted.visibility is TargetVisibility.VISIBLE
            and gt.coordinate_bins is not None
            and predicted.coordinate_bins is not None
        )
        iou = box_iou(gt.coordinate_bins, predicted.coordinate_bins) if conditional_localization else 0.0
        coordinate_mae = (
            float(np.abs(np.asarray(gt.coordinate_bins) - np.asarray(predicted.coordinate_bins)).mean())
            if conditional_localization
            else None
        )
        iou50_count += int(gt_visible and iou >= 0.5)
        strict &= matched and visibility_correct and (not gt_visible or iou >= 0.5)
        target_rows.append(
            {
                **metadata,
                "camera": gt.camera,
                "entity_name": gt.entity_name,
                "entity_role": gt.entity_role.name.lower(),
                "gt_present": 1,
                "pred_present": int(matched),
                "matched": int(matched),
                "gt_visibility": gt.visibility.name.lower(),
                "pred_visibility": predicted.visibility.name.lower() if predicted else "missing",
                "visibility_correct": int(visibility_correct),
                "gt_visible": int(gt_visible),
                "conditional_localization": int(conditional_localization),
                "iou": iou,
                "iou_at_025": int(gt_visible and iou >= 0.25),
                "iou_at_050": int(gt_visible and iou >= 0.5),
                "coordinate_mae_bins": coordinate_mae,
                "gt_box_bins": json.dumps(gt.coordinate_bins),
                "pred_box_bins": json.dumps(predicted.coordinate_bins) if predicted else "",
            }
        )

    extra_count = 0
    for boxes in predicted_by_key.values():
        for predicted in boxes:
            extra_count += 1
            strict = False
            target_rows.append(
                {
                    **metadata,
                    "camera": predicted.camera,
                    "entity_name": predicted.entity_name,
                    "entity_role": predicted.entity_role.name.lower(),
                    "gt_present": 0,
                    "pred_present": 1,
                    "matched": 0,
                    "gt_visibility": "missing",
                    "pred_visibility": predicted.visibility.name.lower(),
                    "visibility_correct": 0,
                    "gt_visible": 0,
                    "conditional_localization": 0,
                    "iou": 0.0,
                    "iou_at_025": 0,
                    "iou_at_050": 0,
                    "coordinate_mae_bins": None,
                    "gt_box_bins": "",
                    "pred_box_bins": json.dumps(predicted.coordinate_bins),
                }
            )
    return target_rows, {
        "gt_target_count": len(gt_boxes),
        "pred_target_count": len(predicted_boxes),
        "matched_target_count": matched_count,
        "extra_target_count": extra_count,
        "visibility_correct_count": visibility_correct_count,
        "gt_visible_count": gt_visible_count,
        "iou50_count": iou50_count,
        "strict_box_success": int(strict and len(gt_boxes) == len(predicted_boxes)),
    }


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_metrics(sample_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt_count = sum(int(row["gt_present"]) for row in target_rows)
    pred_count = sum(int(row["pred_present"]) for row in target_rows)
    matched = sum(int(row["matched"]) for row in target_rows)
    matched_gt_rows = [row for row in target_rows if row["gt_present"] and row["matched"]]
    visible_gt_rows = [row for row in target_rows if row["gt_visible"]]
    localization_rows = [row for row in target_rows if row["conditional_localization"]]
    tf_total = sum(int(row["tf_token_total"]) for row in sample_rows)
    tf_weight_total = sum(float(row["tf_weight_total"]) for row in sample_rows)
    result = {
        "samples": len(sample_rows),
        "parse_success_rate": safe_div(sum(int(row["parse_success"]) for row in sample_rows), len(sample_rows)),
        "free_sequence_exact_rate": safe_div(
            sum(int(row["free_sequence_exact"]) for row in sample_rows), len(sample_rows)
        ),
        "strict_box_success_rate_iou_0.5": safe_div(
            sum(int(row["strict_box_success"]) for row in sample_rows), len(sample_rows)
        ),
        "target_precision": safe_div(matched, pred_count),
        "target_recall": safe_div(matched, gt_count),
        "visibility_accuracy_matched": safe_div(
            sum(int(row["visibility_correct"]) for row in matched_gt_rows), len(matched_gt_rows)
        ),
        "mean_iou_when_both_visible": safe_div(
            sum(float(row["iou"]) for row in localization_rows), len(localization_rows)
        ),
        "mean_iou_all_visible_targets": safe_div(
            sum(float(row["iou"]) for row in visible_gt_rows), len(visible_gt_rows)
        ),
        "visible_box_recall_iou_0.25": safe_div(
            sum(int(row["iou_at_025"]) for row in visible_gt_rows), len(visible_gt_rows)
        ),
        "visible_box_recall_iou_0.5": safe_div(
            sum(int(row["iou_at_050"]) for row in visible_gt_rows), len(visible_gt_rows)
        ),
        "coordinate_mae_bins_when_both_visible": safe_div(
            sum(float(row["coordinate_mae_bins"]) for row in localization_rows),
            len(localization_rows),
        ),
        "teacher_forced_token_accuracy": safe_div(
            sum(int(row["tf_token_correct"]) for row in sample_rows), tf_total
        ),
        "teacher_forced_weighted_token_accuracy": safe_div(
            sum(float(row["tf_weighted_correct"]) for row in sample_rows), tf_weight_total
        ),
        "teacher_forced_weighted_nll": safe_div(
            sum(float(row["tf_nll_sum"]) for row in sample_rows), tf_weight_total
        ),
        "teacher_forced_sequence_exact_rate": safe_div(
            sum(int(row["tf_sequence_exact"]) for row in sample_rows), len(sample_rows)
        ),
    }
    for category in ("structural", "text", "coordinate"):
        total = sum(int(row[f"tf_{category}_total"]) for row in sample_rows)
        correct = sum(int(row[f"tf_{category}_correct"]) for row in sample_rows)
        result[f"teacher_forced_{category}_token_accuracy"] = safe_div(correct, total)
    return result


def grouped_target_metrics(target_rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        grouped[str(row[field])].append(row)
    output = {}
    for name, rows in sorted(grouped.items()):
        gt_rows = [row for row in rows if row["gt_present"]]
        pred_rows = [row for row in rows if row["pred_present"]]
        matched_rows = [row for row in rows if row["gt_present"] and row["matched"]]
        visible_rows = [row for row in gt_rows if row["gt_visible"]]
        localized_rows = [row for row in gt_rows if row["conditional_localization"]]
        output[name] = {
            "ground_truth_targets": len(gt_rows),
            "predicted_targets": len(pred_rows),
            "target_precision": safe_div(len(matched_rows), len(pred_rows)),
            "target_recall": safe_div(len(matched_rows), len(gt_rows)),
            "visibility_accuracy_matched": safe_div(
                sum(int(row["visibility_correct"]) for row in matched_rows), len(matched_rows)
            ),
            "mean_iou_when_both_visible": safe_div(
                sum(float(row["iou"]) for row in localized_rows), len(localized_rows)
            ),
            "visible_box_recall_iou_0.5": safe_div(
                sum(int(row["iou_at_050"]) for row in visible_rows), len(visible_rows)
            ),
        }
    return output


def grouped_task_metrics(sample_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    tasks = sorted({str(row["task"]) for row in sample_rows})
    for task in tasks:
        task_samples = [row for row in sample_rows if row["task"] == task]
        task_targets = [row for row in target_rows if row["task"] == task]
        output[task] = aggregate_metrics(task_samples, task_targets)
    return output


def padded_image(image: torch.Tensor, size: int = 512) -> Image.Image:
    tensor = image.detach().cpu()
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div(255.0)
    elif not tensor.is_floating_point():
        tensor = tensor.float()
    padded = resize_with_pad(tensor.unsqueeze(0), size, size, pad_value=0)[0]
    array = padded.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array).convert("RGB")


def draw_box(draw: ImageDraw.ImageDraw, box: DecodedTargetBox, color: str, prefix: str, size: int) -> None:
    if box.visibility is not TargetVisibility.VISIBLE or box.coordinate_bins is None:
        return
    scale = (size - 1) / 255.0
    x1, y1, x2, y2 = (round(value * scale) for value in box.coordinate_bins)
    invalid_order = x2 < x1 or y2 < y1
    coordinates = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    draw.rectangle(coordinates, outline=color, width=3)
    x, y = coordinates[:2]
    status = " INVALID" if invalid_order else ""
    draw.text((x + 2, max(0, y - 12)), f"{prefix}{status}:{box.entity_name}", fill=color)


def save_preview(
    path: Path,
    raw_batch: dict[str, Any],
    batch_index: int,
    gt_boxes: list[DecodedTargetBox],
    predicted_boxes: list[DecodedTargetBox],
    task: str,
    parse_error: str | None,
    camera_names: Iterable[str],
) -> None:
    size = 512
    camera_names = list(camera_names)
    panel_rows: list[tuple[Image.Image, Image.Image]] = []
    for camera in camera_names:
        image_key = f"observation.images.{camera}"
        image = padded_image(raw_batch[image_key][batch_index], size=size)
        ground_truth_panel = image.copy()
        prediction_panel = image.copy()
        ground_truth_draw = ImageDraw.Draw(ground_truth_panel)
        prediction_draw = ImageDraw.Draw(prediction_panel)
        ground_truth_draw.text((6, 6), f"{camera} | Ground truth", fill="white")
        prediction_draw.text((6, 6), f"{camera} | Prediction", fill="white")
        for box in gt_boxes:
            if box.camera == camera:
                draw_box(ground_truth_draw, box, "#00ff66", "GT", size)
        for box in predicted_boxes:
            if box.camera == camera:
                draw_box(prediction_draw, box, "#ff4050", "P", size)
        panel_rows.append((ground_truth_panel, prediction_panel))

    heading_height = 46
    output = Image.new(
        "RGB",
        (size * 2, size * len(panel_rows) + heading_height),
        "#111111",
    )
    draw = ImageDraw.Draw(output)
    draw.text((6, 5), task[:160], fill="white")
    if parse_error:
        draw.text((6, 23), f"Parse error: {parse_error[:150]}", fill="#ff6060")
    for row_index, (ground_truth_panel, prediction_panel) in enumerate(panel_rows):
        y = heading_height + row_index * size
        output.paste(ground_truth_panel, (0, y))
        output.paste(prediction_panel, (size, y))
    output.save(path, quality=92)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    validate_args(args)
    prepare_output_dir(args.output_dir, args.overwrite)
    device = torch.device(args.device)

    print(f"Loading dataset: {args.dataset_root}", flush=True)
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        return_uint8=True,
    )
    episode_indices = parse_episode_spec(args.episodes, dataset.num_episodes)
    frame_indices = select_frame_indices(
        dataset,
        episode_indices,
        args.frame_stride,
        args.max_samples,
    )
    if not frame_indices:
        raise RuntimeError("Frame selection produced no samples.")
    subset = torch.utils.data.Subset(dataset, frame_indices)
    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lerobot_collate_fn if dataset.meta.has_language_columns else None,
        persistent_workers=args.num_workers > 0,
    )

    print(f"Loading policy: {args.policy_path}", flush=True)
    config = PreTrainedConfig.from_pretrained(args.policy_path, local_files_only=True)
    if not isinstance(config, ActionFlowV22Config):
        raise TypeError(
            f"Expected an ActionFlowV22Config checkpoint, got {type(config).__name__}."
        )
    config.device = str(device)
    config.compile_model = False
    policy = ActionFlowV22Policy.from_pretrained(
        args.policy_path,
        config=config,
        local_files_only=True,
    ).to(device)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        config,
        pretrained_path=str(args.policy_path),
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )
    tokenizer = policy.model.target_tokenizer
    max_new_tokens = args.max_new_tokens or config.max_grounding_tokens
    preview_ordinals = set(
        np.linspace(0, len(frame_indices) - 1, min(args.preview_count, len(frame_indices)), dtype=int).tolist()
    )

    sample_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    vlm_output_handle = (args.output_dir / "vlm_outputs.jsonl").open("w", encoding="utf-8")
    start_time = time.perf_counter()
    processed = 0
    for raw_batch in dataloader:
        batch_size = raw_batch[TARGET_BOX_BINS].shape[0]
        gt_sequences, gt_weights = serialize_ground_truth(raw_batch, tokenizer)
        model_batch = {key: value for key, value in raw_batch.items() if key != ACTION}
        for camera_key in dataset.meta.camera_keys:
            image = model_batch.get(camera_key)
            if isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
                model_batch[camera_key] = image.float().div(255.0)
        model_batch = preprocessor(model_batch)

        with autocast_context(device, args.amp_dtype):
            prefix = prepare_prefix(policy, model_batch)
            packed_targets = pad_grounding_targets(
                gt_sequences,
                gt_weights,
                pad_token_id=int(tokenizer.vlm_tokenizer.pad_token_id),
                action_end_token_id=config.structural_token_id("<action_end>"),
                device=device,
            )
            tf_rows = teacher_forced_metrics(policy, prefix, packed_targets)
            generated_sequences, termination = greedy_generate_grounding(
                policy,
                prefix,
                max_new_tokens=max_new_tokens,
                stop_on_action_token=args.stop_on_action_token,
            )

        for batch_index in range(batch_size):
            ordinal = processed + batch_index
            episode_index = tensor_scalar(raw_batch["episode_index"][batch_index])
            frame_index = tensor_scalar(raw_batch["frame_index"][batch_index])
            dataset_index = tensor_scalar(raw_batch["index"][batch_index])
            task_value = raw_batch["task"][batch_index]
            task = str(task_value)
            metadata = {
                "dataset_index": dataset_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "task": task,
            }
            gt_boxes = tokenizer.deserialize_grounding(gt_sequences[batch_index])
            generated = generated_sequences[batch_index]
            parse_error = None
            try:
                predicted_boxes = tokenizer.deserialize_grounding(generated)
                parse_success = 1
            except Exception as error:  # The exact parser failure is part of the evaluation output.
                predicted_boxes = []
                parse_success = 0
                parse_error = str(error)

            matched_rows, box_metrics = match_targets(gt_boxes, predicted_boxes, metadata)
            target_rows.extend(matched_rows)
            decoder_prompt = generated[:1]
            model_generated = generated[1:]
            ground_truth_payload = token_sequence_payload(
                tokenizer.vlm_tokenizer,
                gt_sequences[batch_index],
            )
            generated_payload = token_sequence_payload(tokenizer.vlm_tokenizer, generated)
            sample_row = {
                **metadata,
                "termination": termination[batch_index],
                "generated_token_count": len(generated),
                "decoder_start_token_id": generated[0] if generated else None,
                "decoder_start_token": (
                    tokenizer.vlm_tokenizer.convert_ids_to_tokens(generated[0]) if generated else ""
                ),
                "first_generated_token_id": generated[1] if len(generated) > 1 else None,
                "first_generated_token": (
                    tokenizer.vlm_tokenizer.convert_ids_to_tokens(generated[1])
                    if len(generated) > 1
                    else ""
                ),
                "ground_truth_token_count": len(gt_sequences[batch_index]),
                "parse_success": parse_success,
                "parse_error": parse_error or "",
                "free_sequence_exact": int(generated == gt_sequences[batch_index]),
                **box_metrics,
                **tf_rows[batch_index],
            }
            if not parse_success:
                sample_row["strict_box_success"] = 0
            sample_rows.append(sample_row)
            prediction_records.append(
                {
                    **metadata,
                    "termination": termination[batch_index],
                    "parse_error": parse_error,
                    "ground_truth_token_ids": gt_sequences[batch_index],
                    "generated_token_ids": generated,
                    "generated_tokens": tokenizer.vlm_tokenizer.convert_ids_to_tokens(generated),
                    "generated_text": generated_payload["text"],
                    "ground_truth_boxes": [decoded_box_dict(box) for box in gt_boxes],
                    "predicted_boxes": [decoded_box_dict(box) for box in predicted_boxes],
                }
            )
            vlm_output_handle.write(
                json.dumps(
                    {
                        **metadata,
                        "decoder_prompt": token_sequence_payload(
                            tokenizer.vlm_tokenizer,
                            decoder_prompt,
                        ),
                        "model_output": token_sequence_payload(
                            tokenizer.vlm_tokenizer,
                            model_generated,
                        ),
                        "full_grounding_sequence": generated_payload,
                        "ground_truth_sequence": ground_truth_payload,
                        "termination": termination[batch_index],
                        "parse_success": bool(parse_success),
                        "parse_error": parse_error,
                        "predicted_boxes": [decoded_box_dict(box) for box in predicted_boxes],
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            if ordinal in preview_ordinals:
                save_preview(
                    args.output_dir / "previews" / f"episode-{episode_index:06d}_frame-{frame_index:06d}.jpg",
                    raw_batch,
                    batch_index,
                    gt_boxes,
                    predicted_boxes,
                    task,
                    parse_error,
                    config.camera_names,
                )

        processed += batch_size
        elapsed = time.perf_counter() - start_time
        print(
            f"Evaluated {processed}/{len(frame_indices)} frames "
            f"({processed / max(elapsed, 1e-6):.2f} frames/s)",
            flush=True,
        )
        vlm_output_handle.flush()
        del prefix, packed_targets, model_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    vlm_output_handle.close()

    overall = aggregate_metrics(sample_rows, target_rows)
    summary = {
        "dataset": {
            "repo_id": args.dataset_repo_id,
            "root": str(args.dataset_root.resolve()),
            "total_episodes": dataset.num_episodes,
            "total_frames": dataset.num_frames,
            "selected_episodes": episode_indices,
            "selected_frames": len(frame_indices),
            "frame_stride": args.frame_stride,
            "max_samples": args.max_samples,
        },
        "policy": {
            "path": str(args.policy_path.resolve()),
            "max_new_tokens": max_new_tokens,
            "stop_on_action_token": args.stop_on_action_token,
            "device": str(device),
            "amp_dtype": args.amp_dtype,
        },
        "overall": overall,
        "generation_diagnostics": {
            "termination_counts": dict(Counter(str(row["termination"]) for row in sample_rows)),
            "first_token_counts": dict(
                Counter(str(row["first_generated_token"]) for row in sample_rows)
            ),
        },
        "by_task": grouped_task_metrics(sample_rows, target_rows),
        "by_camera": grouped_target_metrics(target_rows, "camera"),
        "by_entity": grouped_target_metrics(target_rows, "entity_name"),
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    write_csv(args.output_dir / "samples.csv", sample_rows)
    write_csv(args.output_dir / "targets.csv", target_rows)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in prediction_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print("\nOverall metrics", flush=True)
    for name, value in overall.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {name}: {formatted}", flush=True)
    print(f"Saved evaluation to: {args.output_dir}", flush=True)
    print(f"Saved complete VLM outputs to: {args.output_dir / 'vlm_outputs.jsonl'}", flush=True)


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
