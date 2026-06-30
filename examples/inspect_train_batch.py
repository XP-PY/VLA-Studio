#!/usr/bin/env python
"""Inspect the actual batch structure used by LeRobot training.

This mirrors the important data path in ``lerobot_train.py``:

1. Build the policy config and dataset with ACT delta timestamps.
2. Create the same EpisodeAwareSampler/DataLoader style used by training.
3. Show the raw batch returned by the DataLoader.
4. Apply the same camera uint8 -> float conversion from training.
5. Run the policy preprocessor and show the batch the policy receives.

The output is summarized by shape, dtype, device, min/max/mean/std, and a few
sample values. It intentionally does not print full image tensors by default.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


CURRENT_FILE = Path(__file__).resolve()
WORKSPACE = CURRENT_FILE.parent.parent
LEROBOT_SRC = WORKSPACE / "repos" / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

os.environ.setdefault("HF_HOME", str(WORKSPACE / ".cache" / "huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(WORKSPACE / ".cache" / "lerobot"))

from lerobot.configs import FeatureType  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets import EpisodeAwareSampler, make_dataset  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.act import ACTConfig  # noqa: E402
from lerobot.utils.collate import lerobot_collate_fn  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.utils.feature_utils import dataset_to_policy_features  # noqa: E402


DEFAULT_DATASET_ROOT = WORKSPACE / "runs" / "dataset" / "rm65_aloha_pick_screwdriver_to_box"
DEFAULT_REPO_ID = "rm65_aloha_pick_screwdriver_to_box"


def clone_batch(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: clone_batch(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_batch(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_batch(v) for v in value)
    return copy.deepcopy(value)


def summarize_tensor(name: str, tensor: torch.Tensor, max_values: int) -> list[str]:
    lines = [
        f"{name}: Tensor shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}",
    ]
    if tensor.numel() == 0:
        return lines

    stats_tensor = tensor.detach()
    if not torch.is_floating_point(stats_tensor):
        stats_tensor = stats_tensor.float()
    else:
        stats_tensor = stats_tensor.float()

    lines.append(
        "  "
        f"min={stats_tensor.min().item():.6g} "
        f"max={stats_tensor.max().item():.6g} "
        f"mean={stats_tensor.mean().item():.6g} "
        f"std={stats_tensor.std(unbiased=False).item():.6g}"
    )
    flat = tensor.detach().flatten()
    values = flat[:max_values].cpu().tolist()
    lines.append(f"  first_values={values}")
    return lines


def summarize_array(name: str, array: np.ndarray, max_values: int) -> list[str]:
    lines = [f"{name}: ndarray shape={array.shape} dtype={array.dtype}"]
    if array.size == 0:
        return lines
    stats = array.astype(np.float32, copy=False)
    lines.append(
        "  "
        f"min={float(np.min(stats)):.6g} "
        f"max={float(np.max(stats)):.6g} "
        f"mean={float(np.mean(stats)):.6g} "
        f"std={float(np.std(stats)):.6g}"
    )
    lines.append(f"  first_values={array.reshape(-1)[:max_values].tolist()}")
    return lines


def summarize_value(name: str, value: Any, max_values: int, depth: int = 0) -> list[str]:
    indent = "  " * depth
    label = f"{indent}{name}"

    if isinstance(value, torch.Tensor):
        return [indent + line if i else line for i, line in enumerate(summarize_tensor(label, value, max_values))]
    if isinstance(value, np.ndarray):
        return [indent + line if i else line for i, line in enumerate(summarize_array(label, value, max_values))]
    if isinstance(value, dict):
        lines = [f"{label}: dict keys={list(value.keys())}"]
        for key in value:
            lines.extend(summarize_value(str(key), value[key], max_values, depth + 1))
        return lines
    if isinstance(value, (list, tuple)):
        lines = [f"{label}: {type(value).__name__} len={len(value)}"]
        for i, item in enumerate(value[: min(len(value), 3)]):
            lines.extend(summarize_value(f"[{i}]", item, max_values, depth + 1))
        if len(value) > 3:
            lines.append(f"{indent}  ... {len(value) - 3} more item(s)")
        return lines
    return [f"{label}: {type(value).__name__} value={value!r}"]


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_batch(title: str, batch: dict[str, Any], max_values: int) -> None:
    print_section(title)
    for key in sorted(batch):
        for line in summarize_value(key, batch[key], max_values):
            print(line)


def make_cfg(args: argparse.Namespace) -> TrainPipelineConfig:
    dataset_cfg = DatasetConfig(
        repo_id=args.dataset_repo_id,
        root=str(args.dataset_root),
        video_backend=args.video_backend,
        use_imagenet_stats=args.use_imagenet_stats,
    )
    policy_cfg = ACTConfig(
        device=args.device,
        push_to_hub=False,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        use_amp=args.use_amp,
    )
    return TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        seed=args.seed,
    )


def fill_policy_features_from_dataset(policy_cfg: ACTConfig, dataset) -> None:
    features = dataset_to_policy_features(dataset.meta.features)
    policy_cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    policy_cfg.input_features = {key: ft for key, ft in features.items() if key not in policy_cfg.output_features}


def make_training_dataloader(cfg: TrainPipelineConfig, dataset) -> torch.utils.data.DataLoader:
    if cfg.dataset.streaming:
        sampler = None
        shuffle = True
    else:
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
            shuffle=True,
            seed=cfg.seed if cfg.seed is not None else 0,
        )
        shuffle = False

    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=torch.device(cfg.policy.device).type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )


def convert_camera_uint8_like_training(batch: dict[str, Any], dataset) -> None:
    for cam_key in dataset.meta.camera_keys:
        if cam_key in batch and isinstance(batch[cam_key], torch.Tensor) and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0


def main() -> None:
    args = parse_args()
    cfg = make_cfg(args)

    print_section("Config")
    print(f"dataset.repo_id={cfg.dataset.repo_id}")
    print(f"dataset.root={cfg.dataset.root}")
    print(f"dataset.video_backend={cfg.dataset.video_backend}")
    print(f"policy.type={cfg.policy.type}")
    print(f"policy.device={cfg.policy.device}")
    print(f"policy.chunk_size={cfg.policy.chunk_size}")
    print(f"policy.n_action_steps={cfg.policy.n_action_steps}")
    print(f"batch_size={cfg.batch_size}")
    print(f"num_workers={cfg.num_workers}")

    dataset = make_dataset(cfg)
    fill_policy_features_from_dataset(cfg.policy, dataset)

    print_section("Dataset Metadata")
    print(f"dataset.num_frames={dataset.num_frames}")
    print(f"dataset.num_episodes={dataset.num_episodes}")
    print(f"dataset.fps={dataset.fps}")
    print(f"dataset.camera_keys={dataset.meta.camera_keys}")
    print(f"dataset.features[{ACTION}]={dataset.meta.features[ACTION]}")
    print(f"dataset.features['observation.state']={dataset.meta.features.get('observation.state')}")
    print(f"policy.input_features={cfg.policy.input_features}")
    print(f"policy.output_features={cfg.policy.output_features}")

    preprocessor, _ = make_pre_post_processors(cfg.policy, dataset_stats=dataset.meta.stats)
    dataloader = make_training_dataloader(cfg, dataset)

    iterator = iter(dataloader)
    for batch_index in range(args.num_batches):
        raw_batch = next(iterator)
        print_batch(f"Raw DataLoader Batch #{batch_index}", raw_batch, args.max_values)

        train_batch = clone_batch(raw_batch)
        convert_camera_uint8_like_training(train_batch, dataset)
        print_batch(f"After Training Camera Conversion #{batch_index}", train_batch, args.max_values)

        processed_batch = preprocessor(train_batch)
        print_batch(f"After Policy Preprocessor #{batch_index}", processed_batch, args.max_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--n-action-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--max-values", type=int, default=8)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--use-imagenet-stats", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
