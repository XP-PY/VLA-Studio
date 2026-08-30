#!/usr/bin/env python

"""Evaluate AFV2.3 FAST-token prediction on a saved episode split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.action_flow_v2_3.modeling_action_flow_v2_3 import ActionFlowV23Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn


METRIC_NAMES = (
    "fast_ce_loss",
    "fast_token_top1_accuracy",
    "fast_token_top5_accuracy",
    "fast_sequence_exact_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/rm65_single_table_preparation_and_bussing_afv2_2_planning_boxes")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def evaluate_split(
    *,
    name: str,
    episode_indices: list[int],
    args: argparse.Namespace,
    policy: ActionFlowV23Policy,
) -> dict[str, float | int]:
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        episodes=episode_indices,
        delta_timestamps={"action": [index / args.fps for index in range(policy.config.chunk_size)]},
        video_backend="pyav",
        return_uint8=True,
    )
    sampled_indices = list(range(0, len(dataset), args.sample_stride))
    dataloader = DataLoader(
        Subset(dataset, sampled_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=lerobot_collate_fn if dataset.meta.has_language_columns else None,
    )
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.policy_path),
        dataset_stats=dataset.meta.stats,
    )

    totals = {metric: 0.0 for metric in METRIC_NAMES}
    sample_count = 0
    for batch in dataloader:
        batch = preprocessor(batch)
        _, metrics = policy(batch)
        batch_size = int(batch["action"].shape[0])
        sample_count += batch_size
        for metric in METRIC_NAMES:
            totals[metric] += float(metrics[metric]) * batch_size

    result: dict[str, float | int] = {
        "episodes": len(episode_indices),
        "samples": sample_count,
        **{metric: total / sample_count for metric, total in totals.items()},
    }
    print(f"{name}: {json.dumps(result, sort_keys=True)}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be positive.")
    split = json.loads(args.split_file.read_text())

    policy = ActionFlowV23Policy.from_pretrained(
        args.policy_path,
        local_files_only=True,
    ).to(args.device)
    policy.eval()
    results = {
        "policy_path": str(args.policy_path.resolve()),
        "split_file": str(args.split_file.resolve()),
        "sample_stride": args.sample_stride,
        "train": evaluate_split(
            name="train",
            episode_indices=[int(index) for index in split["train_episodes"]],
            args=args,
            policy=policy,
        ),
        "validation": evaluate_split(
            name="validation",
            episode_indices=[int(index) for index in split["validation_episodes"]],
            args=args,
            policy=policy,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved evaluation to {args.output}")


if __name__ == "__main__":
    main()
