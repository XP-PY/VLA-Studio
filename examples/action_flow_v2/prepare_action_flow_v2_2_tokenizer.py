#!/usr/bin/env python

"""Build and validate the Action Flow V2.2 Stage 4 tokenizer artifact."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.action_flow_v2_2 import (
    ActionFlowV22Config,
    ActionFlowV22TargetTokenizer,
)
from lerobot.policies.action_flow_v2_2.data_contracts_action_flow_v2_2 import (
    TARGET_BOX_BINS,
    TARGET_BOX_LOSS_WEIGHT,
    TARGET_BOX_VISIBILITY,
    TARGET_ENTITY_IDS,
    TARGET_ENTITY_ROLES,
    TARGET_ENTITY_VALID,
)
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import NormalizerProcessorStep, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features

DEFAULT_DATASET_ROOT = Path(
    "runs/dataset/rm65_single_table_preparation_and_bussing_afv2_2_yoloe_boxes"
)
DEFAULT_REPO_ID = "local/rm65_single_table_preparation_and_bussing_afv2_2_yoloe_boxes"
DEFAULT_OUTPUT_DIR = Path("artifacts/tokenizers/action_flow_v2_2_smolvlm_fast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--vlm-tokenizer", default=None)
    parser.add_argument("--action-tokenizer", default=None)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow missing tokenizer files to be downloaded from Hugging Face.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace tokenizer files in an existing generated artifact.",
    )
    return parser.parse_args()


def make_config(
    metadata: LeRobotDatasetMetadata,
    *,
    chunk_size: int,
) -> ActionFlowV22Config:
    features = dataset_to_policy_features(metadata.features)
    config = ActionFlowV22Config(
        device="cpu",
        chunk_size=chunk_size,
        n_action_steps=min(10, chunk_size),
    )
    config.output_features = {
        key: feature for key, feature in features.items() if key == ACTION
    }
    config.input_features = {
        key: feature for key, feature in features.items() if key != ACTION
    }
    config.set_dataset_feature_metadata(metadata.features)
    config.validate_features()
    return config


def get_validation_sample(
    dataset: LeRobotDataset,
    config: ActionFlowV22Config,
    metadata: LeRobotDatasetMetadata,
) -> tuple[dict, torch.Tensor]:
    rows = dataset.hf_dataset[: config.chunk_size]
    episode_indices = [
        int(value.item()) if isinstance(value, torch.Tensor) else int(value)
        for value in rows["episode_index"]
    ]
    if len(set(episode_indices)) != 1:
        raise ValueError("The first validation chunk crosses an episode boundary.")

    state = rows[OBS_STATE][0].float().unsqueeze(0)
    actions = torch.stack([value.float() for value in rows[ACTION]]).unsqueeze(0)
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=metadata.stats)
    transitions = preprocessor.step_through({OBS_STATE: state, ACTION: actions})
    next(transitions)  # Initial transition before any processor step.
    normalized_actions = None
    for step in preprocessor.steps:
        transition = next(transitions)
        if isinstance(step, NormalizerProcessorStep):
            normalized_actions = transition[TransitionKey.ACTION]
            break
    if normalized_actions is None:
        raise RuntimeError("Action Flow V2.2 preprocessor has no normalization step.")
    labels = {
        key: rows[key][0]
        for key in (
            TARGET_BOX_BINS,
            TARGET_BOX_VISIBILITY,
            TARGET_BOX_LOSS_WEIGHT,
            TARGET_ENTITY_IDS,
            TARGET_ENTITY_ROLES,
            TARGET_ENTITY_VALID,
        )
    }
    return labels, normalized_actions[0].cpu()


def scan_grounding_lengths(
    dataset_root: Path,
    target_tokenizer: ActionFlowV22TargetTokenizer,
) -> dict[str, object]:
    """Measure grounding sequence lengths for every frame using direct Parquet reads."""

    keys = (
        TARGET_BOX_BINS,
        TARGET_BOX_VISIBILITY,
        TARGET_BOX_LOSS_WEIGHT,
        TARGET_ENTITY_IDS,
        TARGET_ENTITY_ROLES,
        TARGET_ENTITY_VALID,
    )
    total_frames = 0
    maximum_length = 0
    maximum_index = -1
    histogram: dict[int, int] = {}
    for parquet_path in sorted((dataset_root / "data").rglob("*.parquet")):
        table = pq.read_table(parquet_path, columns=list(keys))
        columns = {key: table[key].to_pylist() for key in keys}
        for row_index in range(table.num_rows):
            tokens, _ = target_tokenizer.serialize_grounding(
                target_box_bins=columns[TARGET_BOX_BINS][row_index],
                target_box_visibility=columns[TARGET_BOX_VISIBILITY][row_index],
                target_box_loss_weight=columns[TARGET_BOX_LOSS_WEIGHT][row_index],
                target_entity_ids=columns[TARGET_ENTITY_IDS][row_index],
                target_entity_roles=columns[TARGET_ENTITY_ROLES][row_index],
                target_entity_valid=columns[TARGET_ENTITY_VALID][row_index],
            )
            length = len(tokens)
            histogram[length] = histogram.get(length, 0) + 1
            if length > maximum_length:
                maximum_length = length
                maximum_index = total_frames + row_index
        total_frames += table.num_rows
    if total_frames == 0:
        raise RuntimeError(f"No dataset rows found under {dataset_root / 'data'}.")
    return {
        "frames_scanned": total_frames,
        "maximum_grounding_tokens": maximum_length,
        "maximum_grounding_dataset_index": maximum_index,
        "grounding_length_histogram": {
            str(length): count for length, count in sorted(histogram.items())
        },
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    if args.chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    manifest_path = args.output_dir / "action_flow_v2_2_tokenizer.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Tokenizer artifact already exists at {args.output_dir}. Use `--overwrite`."
        )

    logging.info("Loading dataset metadata and one validation chunk")
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    config = make_config(metadata, chunk_size=args.chunk_size)
    entity_names = tuple(
        metadata.features[TARGET_BOX_BINS].get("entity_names", ())
    )
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, video_backend="pyav")
    labels, normalized_actions = get_validation_sample(dataset, config, metadata)

    logging.info("Loading and extending SmolVLM and FAST tokenizers")
    target_tokenizer = ActionFlowV22TargetTokenizer.from_pretrained_sources(
        config=config,
        entity_names=entity_names,
        vlm_tokenizer_name=args.vlm_tokenizer,
        action_tokenizer_name=args.action_tokenizer,
        local_files_only=not args.allow_download,
    )
    serialized = target_tokenizer.serialize_sample(
        normalized_relative_actions=normalized_actions,
        target_box_bins=labels[TARGET_BOX_BINS],
        target_box_visibility=labels[TARGET_BOX_VISIBILITY],
        target_box_loss_weight=labels[TARGET_BOX_LOSS_WEIGHT],
        target_entity_ids=labels[TARGET_ENTITY_IDS],
        target_entity_roles=labels[TARGET_ENTITY_ROLES],
        target_entity_valid=labels[TARGET_ENTITY_VALID],
    )
    decoded_boxes = target_tokenizer.deserialize_grounding(
        serialized.grounding_tokens,
        serialized.grounding_token_mask,
    )
    fast_tokens, fast_symbols = target_tokenizer.serialize_fast_actions(normalized_actions)
    assert target_tokenizer.unmap_fast_token_ids(fast_tokens[1:-1]) == fast_symbols
    decoded_actions = target_tokenizer.deserialize_fast_actions(
        serialized.action_tokens,
        token_mask=serialized.action_token_mask,
        time_horizon=config.chunk_size,
        action_dim=config.expected_action_dim,
    )
    logging.info("Checking grounding sequence length across all dataset frames")
    grounding_scan = scan_grounding_lengths(args.dataset_root, target_tokenizer)
    if grounding_scan["maximum_grounding_tokens"] > config.max_grounding_tokens:
        raise ValueError(
            "Grounding sequence budget is too small: "
            f"{grounding_scan['maximum_grounding_tokens']} > {config.max_grounding_tokens}."
        )

    logging.info("Saving tokenizer artifact to %s", args.output_dir)
    target_tokenizer.save_pretrained(args.output_dir)
    reloaded = ActionFlowV22TargetTokenizer.from_pretrained(
        args.output_dir,
        config=config,
    )
    serialized_reloaded = reloaded.serialize_sample(
        normalized_relative_actions=normalized_actions,
        target_box_bins=labels[TARGET_BOX_BINS],
        target_box_visibility=labels[TARGET_BOX_VISIBILITY],
        target_box_loss_weight=labels[TARGET_BOX_LOSS_WEIGHT],
        target_entity_ids=labels[TARGET_ENTITY_IDS],
        target_entity_roles=labels[TARGET_ENTITY_ROLES],
        target_entity_valid=labels[TARGET_ENTITY_VALID],
    )
    torch.testing.assert_close(
        serialized_reloaded.grounding_tokens,
        serialized.grounding_tokens,
    )
    torch.testing.assert_close(
        serialized_reloaded.action_tokens,
        serialized.action_tokens,
    )

    summary = {
        "extended_vocabulary_size": len(target_tokenizer.vlm_tokenizer),
        "grounding_tokens": int(serialized.grounding_token_mask.sum()),
        "fast_target_tokens": int(serialized.action_token_mask.sum()),
        "fast_symbols": len(fast_symbols),
        "decoded_boxes": len(decoded_boxes),
        "fast_decode_mae": float(
            np.mean(np.abs(decoded_actions - normalized_actions.numpy()))
        ),
        "checkpoint_reload_exact": True,
        **grounding_scan,
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    logging.info("Stage 4 tokenizer validation: %s", summary)


if __name__ == "__main__":
    main()
