#!/usr/bin/env python

"""Create a prompt-balanced train/validation episode split for AFV2.3 FAST tests."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-task", type=int, default=1)
    parser.add_argument("--validation-per-task", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def task_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    raise ValueError(f"Episode has no usable task prompt: {value!r}")


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Split already exists: {args.output}")
    if args.train_per_task < 1 or args.validation_per_task < 1:
        raise ValueError("Both per-task episode counts must be positive.")

    episode_files = sorted((args.dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode metadata found under {args.dataset_root}")

    episodes = pd.concat((pd.read_parquet(path) for path in episode_files), ignore_index=True)
    required = {"episode_index", "tasks", "length"}
    missing = required.difference(episodes.columns)
    if missing:
        raise ValueError(f"Episode metadata is missing columns: {sorted(missing)}")
    episodes["task"] = episodes["tasks"].map(task_text)

    rng = random.Random(args.seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    task_splits: list[dict[str, object]] = []
    required_per_task = args.train_per_task + args.validation_per_task

    for task, group in sorted(episodes.groupby("task"), key=lambda item: item[0]):
        candidates = sorted(int(index) for index in group["episode_index"])
        if len(candidates) < required_per_task:
            raise ValueError(
                f"Task {task!r} has {len(candidates)} episodes; {required_per_task} are required."
            )
        selected = rng.sample(candidates, required_per_task)
        train = sorted(selected[: args.train_per_task])
        validation = sorted(selected[args.train_per_task :])
        train_indices.extend(train)
        validation_indices.extend(validation)
        task_splits.append(
            {
                "task": task,
                "available_episodes": len(candidates),
                "train_episodes": train,
                "validation_episodes": validation,
            }
        )

    length_by_episode = episodes.set_index("episode_index")["length"].to_dict()
    train_indices.sort()
    validation_indices.sort()
    split = {
        "dataset_root": str(args.dataset_root.resolve()),
        "seed": args.seed,
        "strategy": "sample_without_replacement_per_task",
        "num_tasks": len(task_splits),
        "train_per_task": args.train_per_task,
        "validation_per_task": args.validation_per_task,
        "train_episodes": train_indices,
        "validation_episodes": validation_indices,
        "train_frames": sum(int(length_by_episode[index]) for index in train_indices),
        "validation_frames": sum(int(length_by_episode[index]) for index in validation_indices),
        "tasks": task_splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n")
    print(
        f"Saved {len(train_indices)} train and {len(validation_indices)} validation episodes "
        f"covering {len(task_splits)} tasks to {args.output}"
    )
    print(f"Train frames: {split['train_frames']}; validation frames: {split['validation_frames']}")


if __name__ == "__main__":
    main()
