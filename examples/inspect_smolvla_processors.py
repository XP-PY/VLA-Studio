#!/usr/bin/env python
"""Inspect what SmolVLA policy processors do to a LeRobot sample.

The script mirrors ``lerobot/policies/smolvla/processor_smolvla.py`` and prints
the data structure before and after each preprocessor/postprocessor step.

Usage example:
python examples/inspect_smolvla_processors.py \
    2>&1 | tee ./examples/inspect_smolvla_processors.log
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

CURRENT_FILE = Path(__file__).resolve()
WORKSPACE = CURRENT_FILE.parent.parent
LEROBOT_SRC = WORKSPACE / "repos" / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

os.environ.setdefault("HF_HOME", str(WORKSPACE / ".cache" / "huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(WORKSPACE / ".cache" / "lerobot"))

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature, PreTrainedConfig  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets import EpisodeAwareSampler, make_dataset  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors  # noqa: E402
from lerobot.processor import (  # noqa: E402
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.converters import batch_to_transition, transition_to_batch  # noqa: E402
from lerobot.types import EnvTransition, TransitionKey  # noqa: E402
from lerobot.utils.collate import lerobot_collate_fn  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.feature_utils import dataset_to_policy_features  # noqa: E402


DEFAULT_DATASET_ID = "rm65_aloha_pick_screwdriver_to_fixed_box_closer_front"
DEFAULT_DATASET_ROOT = WORKSPACE / "runs" / "dataset" / DEFAULT_DATASET_ID


@dataclass
class PlaceholderTokenizerProcessorStep(ProcessorStep):
    """Tokenizer-shaped debug fallback used only when the real tokenizer is unavailable."""

    max_length: int = 48

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            observation = {}
            transition[TransitionKey.OBSERVATION] = observation

        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        task = complementary_data.get("task", [""])
        if isinstance(task, str):
            batch_size = 1
        elif isinstance(task, (list, tuple)):
            batch_size = len(task)
        else:
            batch_size = 1

        observation[OBS_LANGUAGE_TOKENS] = torch.zeros((batch_size, self.max_length), dtype=torch.long)
        observation[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(
            (batch_size, self.max_length), dtype=torch.bool
        )
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    return copy.deepcopy(value)


def summarize_value(name: str, value: Any, indent: int = 0, max_values: int = 4) -> list[str]:
    prefix = "  " * indent
    label = f"{prefix}{name}"
    if isinstance(value, torch.Tensor):
        line = f"{label}: Tensor shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
        if value.numel() > 0 and value.ndim <= 2:
            flat = value.detach().flatten()[:max_values].cpu().tolist()
            line += f" first_values={flat}"
        return [line]
    if isinstance(value, dict):
        lines = [f"{label}: dict keys={list(value.keys())}"]
        for key in sorted(value):
            lines.extend(summarize_value(str(key), value[key], indent + 1, max_values))
        return lines
    if isinstance(value, list):
        lines = [f"{label}: list len={len(value)}"]
        for idx, item in enumerate(value[:3]):
            lines.extend(summarize_value(f"[{idx}]", item, indent + 1, max_values))
        if len(value) > 3:
            lines.append(f"{prefix}  ... {len(value) - 3} more item(s)")
        return lines
    return [f"{label}: {type(value).__name__} value={value!r}"]


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def summarize_transition(transition: EnvTransition, max_values: int) -> None:
    for key in [
        TransitionKey.OBSERVATION,
        TransitionKey.ACTION,
        TransitionKey.COMPLEMENTARY_DATA,
        TransitionKey.REWARD,
        TransitionKey.DONE,
        TransitionKey.TRUNCATED,
        TransitionKey.INFO,
    ]:
        for line in summarize_value(str(key), transition.get(key), max_values=max_values):
            print(line)


def summarize_batch(batch: dict[str, Any], max_values: int) -> None:
    for key in sorted(batch):
        for line in summarize_value(key, batch[key], max_values=max_values):
            print(line)


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
            raise TypeError(f"Expected SmolVLAConfig, got {type(policy_cfg).__name__}")
    else:
        policy_cfg = SmolVLAConfig()

    policy_cfg.device = args.device
    policy_cfg.push_to_hub = False
    policy_cfg.chunk_size = args.chunk_size
    policy_cfg.n_action_steps = args.n_action_steps
    policy_cfg.tokenizer_max_length = args.tokenizer_max_length
    if args.vlm_model_name is not None:
        policy_cfg.vlm_model_name = str(args.vlm_model_name)

    return TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        batch_size=args.batch_size,
        num_workers=0,
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
        num_workers=0,
        batch_size=cfg.batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_fn,
    )


def make_debug_preprocessor(
    cfg: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]],
    allow_placeholder_tokenizer: bool,
) -> PolicyProcessorPipeline:
    steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NewLineTaskProcessorStep(),
    ]

    try:
        steps.append(
            TokenizerProcessorStep(
                tokenizer_name=cfg.vlm_model_name,
                padding=cfg.pad_language_to,
                padding_side="right",
                max_length=cfg.tokenizer_max_length,
            )
        )
    except Exception as exc:
        if not allow_placeholder_tokenizer:
            raise
        print(
            "WARNING: real TokenizerProcessorStep could not be constructed. "
            "Using placeholder token IDs for the rest of this inspection."
        )
        print(f"Tokenizer error: {exc}")
        steps.append(PlaceholderTokenizerProcessorStep(max_length=cfg.tokenizer_max_length))

    steps.extend(
        [
            DeviceProcessorStep(device=cfg.device),
            NormalizerProcessorStep(
                features={**cfg.input_features, **cfg.output_features},
                norm_map=cfg.normalization_mapping,
                stats=dataset_stats,
            ),
        ]
    )
    return PolicyProcessorPipeline(steps=steps, name="debug_smolvla_preprocessor")


def get_input_batch(args: argparse.Namespace, cfg: TrainPipelineConfig, dataset) -> dict[str, Any]:
    if args.source == "sample":
        sample = dataset[args.sample_index]
        convert_camera_uint8_like_training(sample, dataset)
        return sample

    dataloader = make_training_dataloader(cfg, dataset)
    batch = next(iter(dataloader))
    convert_camera_uint8_like_training(batch, dataset)
    return batch


def inspect_preprocessor(preprocessor: PolicyProcessorPipeline, batch: dict[str, Any], max_values: int) -> dict[str, Any]:
    print_section("Input Batch")
    summarize_batch(batch, max_values)

    transition = batch_to_transition(clone_value(batch))
    print_section("Initial EnvTransition From batch_to_transition")
    summarize_transition(transition, max_values)

    for idx, step in enumerate(preprocessor.steps):
        name = step.__class__.__name__
        print_section(f"Preprocessor Step {idx}: {name}")
        print("Before:")
        summarize_transition(transition, max_values)
        transition = step(transition)
        print("\nAfter:")
        summarize_transition(transition, max_values)

    output = transition_to_batch(transition)
    print_section("Final Preprocessor Output Batch")
    summarize_batch(output, max_values)
    return output


def inspect_postprocessor(postprocessor: PolicyProcessorPipeline, action: torch.Tensor, max_values: int) -> None:
    transition = policy_action_to_transition(action)
    print_section("Initial Postprocessor EnvTransition")
    summarize_transition(transition, max_values)

    for idx, step in enumerate(postprocessor.steps):
        name = step.__class__.__name__
        print_section(f"Postprocessor Step {idx}: {name}")
        print("Before:")
        summarize_transition(transition, max_values)
        transition = step(transition)
        print("\nAfter:")
        summarize_transition(transition, max_values)

    output_action = transition_to_policy_action(transition)
    print_section("Final Postprocessor Output Action")
    for line in summarize_value(ACTION, output_action, max_values=max_values):
        print(line)


def main() -> None:
    args = parse_args()
    cfg = make_cfg(args)
    dataset = make_dataset(cfg)
    fill_policy_features_from_dataset(cfg.policy, dataset)

    print_section("SmolVLA Processor Setup")
    print(f"dataset.root={dataset.root}")
    print(f"dataset.num_episodes={dataset.num_episodes}")
    print(f"dataset.num_frames={dataset.num_frames}")
    print(f"source={args.source}")
    print(f"policy.vlm_model_name={cfg.policy.vlm_model_name}")
    print(f"policy.input_features={cfg.policy.input_features}")
    print(f"policy.output_features={cfg.policy.output_features}")

    try:
        preprocessor, postprocessor = make_smolvla_pre_post_processors(
            cfg.policy,
            dataset_stats=dataset.meta.stats,
        )
        print("Constructed exact make_smolvla_pre_post_processors pipeline.")
    except Exception as exc:
        if not args.allow_placeholder_tokenizer:
            raise
        print("Could not construct exact SmolVLA processor pipeline.")
        print(f"Error: {exc}")
        preprocessor = make_debug_preprocessor(
            cfg.policy,
            dataset_stats=dataset.meta.stats,
            allow_placeholder_tokenizer=True,
        )
        postprocessor = PolicyProcessorPipeline(
            steps=[
                UnnormalizerProcessorStep(
                    features=cfg.policy.output_features,
                    norm_map=cfg.policy.normalization_mapping,
                    stats=dataset.meta.stats,
                ),
                DeviceProcessorStep(device="cpu"),
            ],
            name="debug_smolvla_postprocessor",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

    batch = get_input_batch(args, cfg, dataset)
    processed_batch = inspect_preprocessor(preprocessor, batch, args.max_values)

    if args.inspect_postprocessor:
        action = processed_batch[ACTION]
        if action.ndim == 3:
            action = action[:, 0, :]
        inspect_postprocessor(postprocessor, action, args.max_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--policy-path", type=Path, default=WORKSPACE / "models" / "smolvla_base")
    parser.add_argument("--vlm-model-name", default=WORKSPACE / "models" / "SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--tokenizer-max-length", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--source", choices=["sample", "batch"], default="sample")
    parser.add_argument("--max-values", type=int, default=4)
    parser.add_argument("--use-imagenet-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-placeholder-tokenizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inspect-postprocessor", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
