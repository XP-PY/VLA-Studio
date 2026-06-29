#!/usr/bin/env python
"""Inspect the tensor shapes entering SmolVLAPolicy.forward.

This script reads one LeRobot training batch and prints:

1. The raw keys/shapes in the batch passed to SmolVLAPolicy.forward.
2. The internal shapes produced by SmolVLA's prepare_images/prepare_state/prepare_action logic.

It does not instantiate the full SmolVLA VLM by default, so it is safe for shape inspection
without downloading model weights.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

import torch


WORKSPACE = Path("/home/ubuntu/py/vla_studio")
LEROBOT_SRC = WORKSPACE / "repos" / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

os.environ.setdefault("HF_HOME", str(WORKSPACE / ".cache" / "huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(WORKSPACE / ".cache" / "lerobot"))

from lerobot.configs import FeatureType, PreTrainedConfig  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets import EpisodeAwareSampler, make_dataset  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import pad_vector, resize_with_pad  # noqa: E402
from lerobot.utils.collate import lerobot_collate_fn  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.feature_utils import dataset_to_policy_features  # noqa: E402


DEFAULT_DATASET_ID = "rm65_aloha_pick_screwdriver_to_fixed_box_closer_front"
DEFAULT_DATASET_ROOT = WORKSPACE / "runs" / "dataset" / DEFAULT_DATASET_ID


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


def shape_line(name: str, value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"{name}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
    return f"{name}: {type(value).__name__} value={value!r}"


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def convert_camera_uint8_like_training(batch: dict[str, Any], dataset) -> None:
    for cam_key in dataset.meta.camera_keys:
        if cam_key in batch and isinstance(batch[cam_key], torch.Tensor) and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0


def fill_policy_features_from_dataset(policy_cfg: SmolVLAConfig, dataset) -> None:
    features = dataset_to_policy_features(dataset.meta.features)
    policy_cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    policy_cfg.input_features = {key: ft for key, ft in features.items() if key not in policy_cfg.output_features}
    policy_cfg.validate_features()


def make_cfg(args: argparse.Namespace) -> TrainPipelineConfig:
    dataset_cfg = DatasetConfig(
        repo_id=args.dataset_repo_id,
        root=str(args.dataset_root),
        video_backend=args.video_backend,
        use_imagenet_stats=args.use_imagenet_stats,
    )
    if args.policy_path is not None:
        policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path, local_files_only=True)
        if not isinstance(policy_cfg, SmolVLAConfig):
            raise TypeError(f"Expected a SmolVLA policy config, got {type(policy_cfg).__name__}")
        policy_cfg.device = args.device
        policy_cfg.push_to_hub = False
        policy_cfg.chunk_size = args.chunk_size
        policy_cfg.n_action_steps = args.n_action_steps
        policy_cfg.max_state_dim = args.max_state_dim
        policy_cfg.max_action_dim = args.max_action_dim
        policy_cfg.tokenizer_max_length = args.tokenizer_max_length
        policy_cfg.empty_cameras = args.empty_cameras
    else:
        policy_cfg = SmolVLAConfig(
            device=args.device,
            push_to_hub=False,
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            max_state_dim=args.max_state_dim,
            max_action_dim=args.max_action_dim,
            tokenizer_max_length=args.tokenizer_max_length,
            empty_cameras=args.empty_cameras,
        )
    if args.vlm_model_name is not None:
        policy_cfg.vlm_model_name = str(args.vlm_model_name)
    return TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        seed=args.seed,
    )


def make_training_dataloader(cfg: TrainPipelineConfig, dataset) -> torch.utils.data.DataLoader:
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
        shuffle=True,
        seed=cfg.seed if cfg.seed is not None else 0,
    )
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        sampler=sampler,
        pin_memory=torch.device(cfg.policy.device).type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )


def prepare_smolvla_images(batch: dict[str, torch.Tensor], cfg: SmolVLAConfig):
    images = []
    img_masks = []
    present_img_keys = [key for key in cfg.image_features if key in batch]
    missing_img_keys = [key for key in cfg.image_features if key not in batch]
    if not present_img_keys:
        raise ValueError(f"No image keys found. Expected one of: {list(cfg.image_features)}")

    for key in present_img_keys:
        img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
        if cfg.resize_imgs_with_padding is not None:
            img = resize_with_pad(img, *cfg.resize_imgs_with_padding, pad_value=0)
        img = img * 2.0 - 1.0
        bsize = img.shape[0]
        if f"{key}_padding_mask" in batch:
            mask = batch[f"{key}_padding_mask"].bool()
        else:
            mask = torch.ones(bsize, dtype=torch.bool, device=img.device)
        images.append((key, img))
        img_masks.append((f"{key}_mask", mask))

    for num_empty_cameras in range(len(missing_img_keys)):
        if num_empty_cameras >= cfg.empty_cameras:
            break
        img = torch.ones_like(images[-1][1]) * -1
        mask = torch.zeros_like(img_masks[-1][1])
        images.append((f"empty_camera_{num_empty_cameras}", img))
        img_masks.append((f"empty_camera_{num_empty_cameras}_mask", mask))

    return images, img_masks


def print_forward_shapes(batch: dict[str, Any], cfg: SmolVLAConfig) -> None:
    print_section("Batch Passed To SmolVLAPolicy.forward(batch)")
    for key in sorted(batch):
        print(shape_line(key, batch[key]))

    print_section("Inside SmolVLAPolicy.forward")
    images, img_masks = prepare_smolvla_images(batch, cfg)
    state_raw = batch[OBS_STATE]
    state = state_raw[:, -1, :] if state_raw.ndim > 2 else state_raw
    state = pad_vector(state, cfg.max_state_dim)
    actions = pad_vector(batch[ACTION], cfg.max_action_dim)

    state_dim = cfg.robot_state_feature.shape[0] if cfg.robot_state_feature is not None else None
    action_dim = cfg.action_feature.shape[0] if cfg.action_feature is not None else None
    print(f"original state_dim={state_dim}")
    print(f"original action_dim={action_dim}")
    print(f"max_state_dim={cfg.max_state_dim}")
    print(f"max_action_dim={cfg.max_action_dim}")
    print(f"chunk_size={cfg.chunk_size}")
    for key, image in images:
        print(shape_line(f"images[{key}]", image))
    for key, mask in img_masks:
        print(shape_line(f"img_masks[{key}]", mask))
    print(shape_line("lang_tokens", batch[OBS_LANGUAGE_TOKENS]))
    print(shape_line("lang_masks", batch[OBS_LANGUAGE_ATTENTION_MASK]))
    print(shape_line("state after prepare_state", state))
    print(shape_line("actions after prepare_action", actions))
    if "action_is_pad" in batch:
        print(shape_line("action_is_pad", batch["action_is_pad"]))

    print_section("VLAFlowMatching.forward Receives")
    print("images: list[Tensor], one per camera, each (B, 3, 512, 512) after default resize")
    print("img_masks: list[Tensor], one per camera, each (B,)")
    print(shape_line("lang_tokens", batch[OBS_LANGUAGE_TOKENS]))
    print(shape_line("lang_masks", batch[OBS_LANGUAGE_ATTENTION_MASK]))
    print(shape_line("state", state))
    print(shape_line("actions", actions))
    print("noise: None by default, otherwise same shape as actions")
    print("time: None by default, otherwise shape=(B,)")


def add_placeholder_language_tokens(batch: dict[str, Any], cfg: SmolVLAConfig) -> None:
    if OBS_LANGUAGE_TOKENS in batch and OBS_LANGUAGE_ATTENTION_MASK in batch:
        return

    first_tensor = next(value for value in batch.values() if isinstance(value, torch.Tensor))
    bsize = first_tensor.shape[0]
    device = first_tensor.device
    seq_len = cfg.tokenizer_max_length
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros((bsize, seq_len), dtype=torch.long, device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones((bsize, seq_len), dtype=torch.bool, device=device)
    print(
        "WARNING: using placeholder language tokens because the SmolVLM tokenizer is not available locally. "
        f"Shape is fixed to tokenizer_max_length={seq_len}."
    )


def main() -> None:
    args = parse_args()
    cfg = make_cfg(args)
    dataset = make_dataset(cfg)
    fill_policy_features_from_dataset(cfg.policy, dataset)

    print_section("Dataset / Policy")
    print(f"dataset.root={dataset.root}")
    print(f"dataset.num_episodes={dataset.num_episodes}")
    print(f"dataset.num_frames={dataset.num_frames}")
    print(f"dataset.camera_keys={dataset.meta.camera_keys}")
    print(f"dataset.fps={dataset.fps}")
    print(f"policy.image_features={list(cfg.policy.image_features)}")
    print(f"policy.robot_state_feature={cfg.policy.robot_state_feature}")
    print(f"policy.action_feature={cfg.policy.action_feature}")
    print(f"policy.vlm_model_name={cfg.policy.vlm_model_name}")

    dataloader = make_training_dataloader(cfg, dataset)
    raw_batch = next(iter(dataloader))
    batch = clone_batch(raw_batch)
    convert_camera_uint8_like_training(batch, dataset)

    try:
        preprocessor, _ = make_pre_post_processors(cfg.policy, dataset_stats=dataset.meta.stats)
        batch = preprocessor(batch)
    except Exception as exc:
        print(
            "WARNING: could not run SmolVLA preprocessor. "
            "Continuing with camera conversion and placeholder language tokens."
        )
        print(f"Preprocessor error: {exc}")
        add_placeholder_language_tokens(batch, cfg.policy)

    print_forward_shapes(batch, cfg.policy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=WORKSPACE / "models" / "smolvla_base",
        help="Optional local SmolVLA policy directory containing config.json.",
    )
    parser.add_argument(
        "--vlm-model-name",
        default=None,
        help=(
            "Optional tokenizer/VLM path or repo id. Use this to point at a local "
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct snapshot if available."
        ),
    )
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-state-dim", type=int, default=32)
    parser.add_argument("--max-action-dim", type=int, default=32)
    parser.add_argument("--tokenizer-max-length", type=int, default=48)
    parser.add_argument("--empty-cameras", type=int, default=0)
    parser.add_argument("--use-imagenet-stats", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
