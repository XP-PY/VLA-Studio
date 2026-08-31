#!/usr/bin/env python
"""Inspect the tensors entering PI0FastPolicy.forward.

The script reads one training batch, runs the real PI0-FAST preprocessor, and
prints the image, prompt-token, action-token, and mask shapes used for training.
It does not instantiate the full PaliGemma model or run its expensive forward pass.

Example:
python examples/inspect_pi0_fast_forward_shapes.py \
    2>&1 | tee logs/inspect_pi0_fast_forward_shapes.log

or:
python examples/inspect_pi0_fast_forward_shapes.py --seed 42 \
    2>&1 | tee logs/inspect_pi0_fast_forward_shapes.log
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
from transformers import AutoProcessor, AutoTokenizer

CURRENT_FILE = Path(__file__).resolve()
WORKSPACE = CURRENT_FILE.parent.parent
LEROBOT_SRC = WORKSPACE / "repos" / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

os.environ.setdefault("HF_HOME", str(WORKSPACE / ".cache" / "huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(WORKSPACE / ".cache" / "lerobot"))

from lerobot.configs import FeatureType, NormalizationMode  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets import EpisodeAwareSampler, make_dataset  # noqa: E402
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig  # noqa: E402
from lerobot.policies.pi0_fast.modeling_pi0_fast import resize_with_pad_torch  # noqa: E402
from lerobot.policies.pi0_fast.processor_pi0_fast import make_pi0_fast_pre_post_processors  # noqa: E402
from lerobot.utils.collate import lerobot_collate_fn  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.feature_utils import dataset_to_policy_features  # noqa: E402


# DEFAULT_DATASET_ID = "rm65_single_table_preparation_and_bussing"
DEFAULT_DATASET_ID = "rm65_single_table_preparation_and_bussing_pi"
DEFAULT_DATASET_ROOT = WORKSPACE / "runs" / "dataset" / DEFAULT_DATASET_ID
DEFAULT_TEXT_TOKENIZER = WORKSPACE / "models" / "paligemma-3b-pt-224"
LOCAL_ACTION_TOKENIZER = WORKSPACE / "models" / "fast-action-tokenizer"
DEFAULT_ACTION_TOKENIZER = (
    str(LOCAL_ACTION_TOKENIZER) if LOCAL_ACTION_TOKENIZER.exists() else "lerobot/fast-action-tokenizer"
)
PALIGEMMA_IMAGE_TOKENS = 256
PALIGEMMA_HIDDEN_SIZE = 2048


def clone_batch(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_batch(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_batch(item) for item in value)
    return copy.deepcopy(value)


def shape_line(name: str, value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return (
            f"{name}: shape={tuple(value.shape)} dtype={value.dtype} "
            f"device={value.device}"
        )
    return f"{name}: {type(value).__name__} value={value!r}"


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def convert_camera_uint8_like_training(batch: dict[str, Any], dataset) -> None:
    for camera_key in dataset.meta.camera_keys:
        image = batch.get(camera_key)
        if isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            batch[camera_key] = image.to(dtype=torch.float32) / 255.0


def fill_policy_features_from_dataset(config: PI0FastConfig, dataset) -> None:
    features = dataset_to_policy_features(dataset.meta.features)
    config.output_features = {
        key: feature for key, feature in features.items() if feature.type is FeatureType.ACTION
    }
    config.input_features = {
        key: feature for key, feature in features.items() if key not in config.output_features
    }

    action_names = dataset.meta.features.get(ACTION, {}).get("names")
    if action_names is not None:
        config.action_feature_names = list(action_names)
    config.validate_features()


def make_config(args: argparse.Namespace) -> TrainPipelineConfig:
    normalization_mode = NormalizationMode(args.normalization_mode)
    dataset_config = DatasetConfig(
        repo_id=args.dataset_repo_id,
        root=str(args.dataset_root),
        video_backend=args.video_backend,
        return_uint8=True,
    )
    policy_config = PI0FastConfig(
        device=args.device,
        dtype=args.dtype,
        push_to_hub=False,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        max_state_dim=args.max_state_dim,
        max_action_dim=args.max_action_dim,
        max_action_tokens=args.max_action_tokens,
        tokenizer_max_length=args.tokenizer_max_length,
        text_tokenizer_name=str(args.text_tokenizer_name),
        action_tokenizer_name=str(args.action_tokenizer_name),
        use_relative_actions=args.use_relative_actions,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": normalization_mode,
            "ACTION": normalization_mode,
        },
    )
    return TrainPipelineConfig(
        dataset=dataset_config,
        policy=policy_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def make_training_dataloader(config: TrainPipelineConfig, dataset) -> torch.utils.data.DataLoader:
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=getattr(config.policy, "drop_n_last_frames", 0),
        shuffle=True,
        seed=config.seed if config.seed is not None else 0,
    )
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=torch.device(config.policy.device).type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )


def prepare_images_like_policy(
    batch: dict[str, torch.Tensor], config: PI0FastConfig
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    images = []
    image_masks = []

    for key in config.image_features:
        if key not in batch:
            continue

        image = batch[key].to(dtype=torch.float32)
        channels_first = image.shape[1] == 3
        if channels_first:
            image = image.permute(0, 2, 3, 1)
        if image.shape[1:3] != config.image_resolution:
            image = resize_with_pad_torch(image, *config.image_resolution)
        image = image * 2.0 - 1.0
        if channels_first:
            image = image.permute(0, 3, 1, 2)

        mask = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)
        images.append((key, image))
        image_masks.append((f"{key}_mask", mask))

    if not images:
        raise ValueError(f"No configured camera was found in the batch: {list(config.image_features)}")
    return images, image_masks


def decode_prompts(tokens: torch.Tensor, masks: torch.Tensor, tokenizer_name: str) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        add_eos_token=True,
        add_bos_token=False,
        local_files_only=Path(tokenizer_name).exists(),
    )
    return [
        tokenizer.decode(sample_tokens[sample_mask], skip_special_tokens=True)
        for sample_tokens, sample_mask in zip(tokens.cpu(), masks.cpu(), strict=True)
    ]


def print_decoded_action_tokens(
    action_tokens: torch.Tensor,
    action_masks: torch.Tensor,
    tokenizer_name: str,
    action_tokenizer_name: str,
    fast_skip_tokens: int,
    action_targets: torch.Tensor,
    action_names: list[str] | None,
) -> None:
    """Inspect the full PaliGemma-token-to-continuous-action decoding path."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        add_eos_token=True,
        add_bos_token=False,
        local_files_only=Path(tokenizer_name).exists(),
    )
    decoded_tokens = [
        tokenizer.convert_ids_to_tokens(sample.tolist()) for sample in action_tokens.cpu()
    ]

    print_section("PaliGemma Token Strings Before FAST Detokenization")
    print("Equivalent to modeling_pi0_fast.py detokenize_actions() decoded_tokens.")
    print(
        "decoded_tokens: Python list with equivalent shape="
        f"({len(decoded_tokens)}, {len(decoded_tokens[0]) if decoded_tokens else 0})"
    )

    if not action_masks.any():
        print("FAST detokenization skipped because the action tokens are placeholders.")
        return

    action_tokenizer = AutoProcessor.from_pretrained(
        action_tokenizer_name,
        trust_remote_code=True,
        local_files_only=Path(action_tokenizer_name).exists(),
    )
    action_prefix_ids = tokenizer.encode("Action: ", add_special_tokens=False)
    action_prefix_tokens = tokenizer.convert_ids_to_tokens(action_prefix_ids)
    bos_id = tokenizer.bos_token_id
    action_horizon = action_targets.shape[-2]
    action_dim = action_targets.shape[-1]
    paligemma_action_token_max = tokenizer.vocab_size - 1 - fast_skip_tokens

    for index, (sample_ids, sample_tokens, sample_mask) in enumerate(
        zip(action_tokens.cpu(), decoded_tokens, action_masks.cpu(), strict=True)
    ):
        valid_mask = sample_mask.bool()
        valid_ids = sample_ids[valid_mask].tolist()
        valid_tokens = [token for token, valid in zip(sample_tokens, valid_mask.tolist(), strict=True) if valid]
        padding_count = len(sample_tokens) - len(valid_tokens)

        print(f"sample[{index}].valid_length={len(valid_tokens)} padding_length={padding_count}")
        print(f"sample[{index}].valid_token_ids={valid_ids}")
        print(f"sample[{index}].valid_decoded_tokens={valid_tokens!r}")

        # During inference BOS is supplied as the generation seed, so it is not part
        # of the tokens returned by sample_actions_fast[_kv_cache].
        generated_ids = valid_ids[1:] if valid_ids and valid_ids[0] == bos_id else valid_ids
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        print(f"sample[{index}].expected_generated_tokens={generated_tokens!r}")

        if "|" in generated_tokens:
            generated_tokens = generated_tokens[: generated_tokens.index("|")]

        prefix_length = len(action_prefix_tokens)
        if generated_tokens[:prefix_length] == action_prefix_tokens:
            cleaned_tokens = generated_tokens[prefix_length:]
        else:
            cleaned_tokens = generated_tokens

        raw_action_token_ids = tokenizer.convert_tokens_to_ids(cleaned_tokens)
        fast_token_ids = [paligemma_action_token_max - token_id for token_id in raw_action_token_ids]
        decoded_action = action_tokenizer.decode(
            [fast_token_ids],
            time_horizon=action_horizon,
            action_dim=action_dim,
        )[0]
        target_action = action_targets[index].detach().cpu().numpy()

        print(f"sample[{index}].cleaned_action_tokens={cleaned_tokens!r}")
        print(f"sample[{index}].raw_paligemma_action_token_ids={raw_action_token_ids}")
        print(f"sample[{index}].fast_token_ids={fast_token_ids}")
        print(f"sample[{index}].decoded_continuous_action_shape={decoded_action.shape}")
        print(f"sample[{index}].decoded_action_first_step={decoded_action[0].tolist()}")
        print(f"sample[{index}].decoded_action_last_step={decoded_action[-1].tolist()}")
        print(f"sample[{index}].FAST_round_trip_MAE={np.abs(decoded_action - target_action).mean():.6f}")

        if action_names is not None:
            for gripper_index, name in enumerate(action_names[:action_dim]):
                if "gripper" not in name.lower():
                    continue
                gripper_values = decoded_action[:, gripper_index]
                print(
                    f"sample[{index}].decoded_{name}: shape={gripper_values.shape} "
                    f"min={gripper_values.min():.6f} max={gripper_values.max():.6f} "
                    f"first_10={gripper_values[:10].tolist()}"
                )


def prepare_without_action_tokenizer(
    batch: dict[str, Any],
    config: PI0FastConfig,
    dataset,
) -> dict[str, Any]:
    """Create real state-language tokens and shape-only FAST action placeholders."""
    state = batch[OBS_STATE].to(dtype=torch.float32)
    state_stats = dataset.meta.stats[OBS_STATE]
    mean = torch.as_tensor(state_stats["mean"], dtype=state.dtype, device=state.device)
    std = torch.as_tensor(state_stats["std"], dtype=state.dtype, device=state.device)
    normalized_state = (state - mean) / (std + 1e-8)
    batch[OBS_STATE] = normalized_state

    discretized_states = np.digitize(
        normalized_state.cpu().numpy(),
        bins=np.linspace(-1, 1, 256 + 1)[:-1],
    ) - 1
    tasks = batch.get("task")
    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(tasks, (list, tuple)):
        raise ValueError("The batch does not contain task strings required by PI0-FAST")

    prompts = []
    for index, task in enumerate(tasks):
        cleaned_task = task.strip().replace("_", " ").replace("\n", " ")
        state_text = " ".join(map(str, discretized_states[index]))
        prompts.append(f"Task: {cleaned_task}, State: {state_text};\n")

    tokenizer = AutoTokenizer.from_pretrained(
        config.text_tokenizer_name,
        local_files_only=Path(config.text_tokenizer_name).exists(),
    )
    tokenized_prompt = tokenizer(
        prompts,
        max_length=config.tokenizer_max_length,
        truncation=True,
        padding="max_length",
        padding_side="right",
        return_tensors="pt",
    )
    batch[OBS_LANGUAGE_TOKENS] = tokenized_prompt["input_ids"]
    batch[OBS_LANGUAGE_ATTENTION_MASK] = tokenized_prompt["attention_mask"].bool()

    batch_size = state.shape[0]
    batch[ACTION_TOKENS] = torch.zeros(
        batch_size,
        config.max_action_tokens,
        dtype=torch.long,
        device=state.device,
    )
    batch[ACTION_TOKEN_MASK] = torch.zeros(
        batch_size,
        config.max_action_tokens,
        dtype=torch.bool,
        device=state.device,
    )
    return batch


def print_forward_shapes(batch: dict[str, Any], config: PI0FastConfig) -> None:
    print_section("Batch Passed To PI0FastPolicy.forward(batch)")
    for key in sorted(batch):
        print(shape_line(key, batch[key]))

    images, image_masks = prepare_images_like_policy(batch, config)
    language_tokens = batch[OBS_LANGUAGE_TOKENS]
    language_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    action_tokens = batch[ACTION_TOKENS]
    action_masks = batch[ACTION_TOKEN_MASK]

    print_section("PI0FastPolicy.forward Prepares")
    for key, image in images:
        print(shape_line(f"images[{key}]", image))
    for key, mask in image_masks:
        print(shape_line(f"img_masks[{key}]", mask))
    print(shape_line("tokens", language_tokens))
    print(shape_line("masks", language_masks))
    print(shape_line("fast_action_tokens", action_tokens))
    print(shape_line("fast_action_masks", action_masks))
    print(f"normalized observation.state values: {batch[OBS_STATE].detach().cpu().tolist()}")
    print(f"valid language tokens per sample: {language_masks.sum(dim=1).tolist()}")
    print(f"valid FAST action tokens per sample: {action_masks.sum(dim=1).tolist()}")

    print_section("Decoded Language + Robot-State Prompt")
    prompts = decode_prompts(language_tokens, language_masks, config.text_tokenizer_name)
    for index, prompt in enumerate(prompts):
        print(f"sample[{index}]: {prompt!r}")

    print_decoded_action_tokens(
        action_tokens,
        action_masks,
        config.text_tokenizer_name,
        config.action_tokenizer_name,
        config.fast_skip_tokens,
        batch[ACTION],
        config.action_feature_names,
    )

    batch_size = language_tokens.shape[0]
    num_cameras = len(images)
    total_image_tokens = num_cameras * PALIGEMMA_IMAGE_TOKENS
    total_prefix_tokens = total_image_tokens + language_tokens.shape[1] + action_tokens.shape[1]

    print_section("PI0FastPytorch.forward Receives / Produces")
    print(f"images: list of {num_cameras} tensors")
    print(f"img_masks: list of {num_cameras} tensors, each shape=({batch_size},)")
    print(shape_line("tokens", language_tokens))
    print(shape_line("masks", language_masks))
    print(shape_line("fast_action_tokens", action_tokens))
    print(shape_line("fast_action_masks", action_masks))
    print(
        "embedded prefix: "
        f"({batch_size}, {total_prefix_tokens}, {PALIGEMMA_HIDDEN_SIZE}) = "
        f"{total_image_tokens} image + {language_tokens.shape[1]} prompt + "
        f"{action_tokens.shape[1]} FAST positions"
    )
    print(f"2D attention mask: ({batch_size}, {total_prefix_tokens}, {total_prefix_tokens})")
    print(
        "FAST logits before shifting: "
        f"({batch_size}, {action_tokens.shape[1]}, paligemma_vocab_size)"
    )
    print(
        "FAST logits/targets used by cross entropy after shifting: "
        f"({batch_size}, {action_tokens.shape[1] - 1}, paligemma_vocab_size) / "
        f"({batch_size}, {action_tokens.shape[1] - 1})"
    )
    print("PI0FastPolicy.forward returns: scalar loss and {'loss': float, 'ce_loss': float}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--text-tokenizer-name", default=DEFAULT_TEXT_TOKENIZER)
    parser.add_argument("--action-tokenizer-name", default=DEFAULT_ACTION_TOKENIZER)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-state-dim", type=int, default=32)
    parser.add_argument("--max-action-dim", type=int, default=32)
    parser.add_argument("--max-action-tokens", type=int, default=256)
    parser.add_argument("--tokenizer-max-length", type=int, default=200)
    parser.add_argument(
        "--normalization-mode",
        choices=("MEAN_STD", "MIN_MAX", "QUANTILES", "QUANTILE10"),
        default="QUANTILES",
    )
    parser.add_argument("--use-relative-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-placeholder-action-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue shape inspection when the FAST action tokenizer is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_config(args)
    dataset = make_dataset(config)
    fill_policy_features_from_dataset(config.policy, dataset)

    print_section("Dataset / PI0-FAST Configuration")
    print(f"dataset.root={dataset.root}")
    print(f"dataset.num_episodes={dataset.num_episodes}")
    print(f"dataset.num_frames={dataset.num_frames}")
    print(f"dataset.camera_keys={dataset.meta.camera_keys}")
    print(f"dataset.fps={dataset.fps}")
    print(f"policy.image_features={list(config.policy.image_features)}")
    print(f"policy.text_tokenizer_name={config.policy.text_tokenizer_name}")
    print(f"policy.action_tokenizer_name={config.policy.action_tokenizer_name}")
    print(f"policy.chunk_size={config.policy.chunk_size}")
    print(f"policy.max_action_tokens={config.policy.max_action_tokens}")
    print(f"policy.normalization_mapping={config.policy.normalization_mapping}")

    dataloader = make_training_dataloader(config, dataset)
    batch = clone_batch(next(iter(dataloader)))
    convert_camera_uint8_like_training(batch, dataset)

    print_section("Batch Passed To Preprocessor")
    for key in sorted(batch):
        print(shape_line(key, batch[key]))

    using_placeholder_action_tokens = False
    try:
        preprocessor, _ = make_pi0_fast_pre_post_processors(
            config.policy,
            dataset_stats=dataset.meta.stats,
        )
    except Exception as error:
        if not args.allow_placeholder_action_tokens:
            raise RuntimeError(
                "Could not create the PI0-FAST preprocessor. Ensure the FAST action tokenizer is available. "
                "Download 'lerobot/fast-action-tokenizer' locally and pass "
                "--action-tokenizer-name models/fast-action-tokenizer."
            ) from error
        print_section("FAST Action Tokenizer Warning")
        print(f"{type(error).__name__}: {error}")
        print("Language + state tokens are real; FAST action tokens below are shape-only placeholders.")
        batch = prepare_without_action_tokenizer(batch, config.policy, dataset)
        using_placeholder_action_tokens = True
    else:
        batch = preprocessor(batch)

    if using_placeholder_action_tokens:
        print("[WARNING] Valid FAST action-token counts are zero because placeholders are being used.")
    print_forward_shapes(batch, config.policy)


if __name__ == "__main__":
    main()
