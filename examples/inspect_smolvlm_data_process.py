#!/usr/bin/env python
"""Inspect image and language embedding paths in SmolVLMWithExpertModel.

This script builds a pseudo image tensor in the format expected by
`SmolVLMWithExpertModel.embed_image`, tokenizes a pseudo task prompt for
`SmolVLMWithExpertModel.embed_language_tokens`, then prints:

1. The input image tensor passed to `embed_image`.
2. The output of the SmolVLM vision encoder before the connector.
3. The output of the connector, which is also the output of `embed_image`.
4. The language token ids and attention mask.
5. The output language token embeddings.

Usage example:

python examples/inspect_smolvlm_data_process.py \
    --policy-path models/smolvla_base \
    --model-id models/SmolVLM2-500M-Video-Instruct \
    --load-vlm-weights \
    --task "grasp screwdriver and place it in box" \
    --tokenizer-max-length 48 \
    --padding max_length \
    --padding-side right \
    --device cuda \
    2>&1 | tee ./logs/inspect_smolvlm_data_process.log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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

from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel  # noqa: E402


DEFAULT_POLICY_PATH = WORKSPACE / "models" / "smolvla_base"
DEFAULT_VLM_PATH = WORKSPACE / "models" / "SmolVLM2-500M-Video-Instruct"


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def load_policy_config(policy_path: Path | None) -> dict[str, Any]:
    if policy_path is None:
        return {}
    config_path = policy_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Policy config not found: {config_path}")
    with config_path.open() as f:
        return json.load(f)


def resolve_model_id(args: argparse.Namespace, policy_cfg: dict[str, Any]) -> str:
    if args.model_id is not None:
        return str(args.model_id)
    model_id = policy_cfg.get("vlm_model_name")
    if model_id:
        return str(model_id)
    return "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def summarize_tensor(name: str, tensor: torch.Tensor, max_values: int = 8) -> None:
    detached = tensor.detach()
    print(f"{name}:")
    print(f"  shape={tuple(detached.shape)}")
    print(f"  dtype={detached.dtype}")
    print(f"  device={detached.device}")
    print(f"  requires_grad={tensor.requires_grad}")

    if detached.numel() == 0:
        print("  empty tensor")
        return

    values = detached.float()
    finite = torch.isfinite(values)
    if finite.any():
        finite_values = values[finite]
        print(f"  min={finite_values.min().item():.6f}")
        print(f"  max={finite_values.max().item():.6f}")
        print(f"  mean={finite_values.mean().item():.6f}")
        print(f"  std={finite_values.std(unbiased=False).item():.6f}")
    else:
        print("  no finite values")

    flat = values.flatten()[:max_values].cpu().tolist()
    print(f"  first_values={flat}")


def make_pseudo_image(args: argparse.Namespace) -> torch.Tensor:
    batch_size = args.batch_size
    height = args.height
    width = args.width

    if args.image_mode == "random":
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        image = torch.rand((batch_size, 3, height, width), generator=generator)
    elif args.image_mode == "zeros":
        image = torch.zeros((batch_size, 3, height, width))
    elif args.image_mode == "ones":
        image = torch.ones((batch_size, 3, height, width))
    else:
        x = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width).expand(batch_size, 1, height, width)
        y = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1).expand(batch_size, 1, height, width)
        image = torch.cat([x, y, (x + y) * 0.5], dim=1)

    if args.smolvla_range:
        # SmolVLAPolicy.prepare_images converts [0, 1] images to [-1, 1]
        # before calling VLAFlowMatching.embed_prefix -> embed_image.
        image = image * 2.0 - 1.0

    return image


def make_language_inputs(
    model: SmolVLMWithExpertModel,
    args: argparse.Namespace,
    tokenizer_max_length: int,
    padding: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    task = args.task
    if args.add_newline and not task.endswith("\n"):
        task = f"{task}\n"
    texts = [task] * args.batch_size

    tokenizer = getattr(model.processor, "tokenizer", model.processor)
    tokenized = tokenizer(
        texts,
        max_length=tokenizer_max_length,
        truncation=True,
        padding=padding,
        padding_side=args.padding_side,
        return_tensors="pt",
    )
    return tokenized["input_ids"], tokenized["attention_mask"], texts


def make_model(args: argparse.Namespace, policy_cfg: dict[str, Any], model_id: str) -> SmolVLMWithExpertModel:
    kwargs = {
        "model_id": model_id,
        "load_vlm_weights": args.load_vlm_weights,
        "train_expert_only": policy_cfg.get("train_expert_only", True),
        "freeze_vision_encoder": policy_cfg.get("freeze_vision_encoder", True),
        "attention_mode": policy_cfg.get("attention_mode", "cross_attn"),
        "num_expert_layers": policy_cfg.get("num_expert_layers", -1),
        "num_vlm_layers": policy_cfg.get("num_vlm_layers", 16),
        "self_attn_every_n_layers": policy_cfg.get("self_attn_every_n_layers", 2),
        "expert_width_multiplier": policy_cfg.get("expert_width_multiplier", 0.75),
        "device": args.device,
    }

    print_section("Model Construction")
    print(f"model_id={model_id}")
    print(f"load_vlm_weights={args.load_vlm_weights}")
    for key, value in kwargs.items():
        if key not in {"model_id", "load_vlm_weights"}:
            print(f"{key}={value}")

    try:
        model = SmolVLMWithExpertModel(**kwargs)
    except Exception as exc:
        print("\nFailed to construct SmolVLMWithExpertModel.")
        print("Common causes:")
        print("  1. The SmolVLM backbone config/processor is not available locally.")
        print("  2. `models/smolvla_base` is a SmolVLA policy checkpoint, not the full SmolVLM tokenizer/processor.")
        print("  3. Network access is disabled and Hugging Face files are not cached.")
        print("\nFix: pass --model-id /path/to/local/SmolVLM2-500M-Video-Instruct or download/cache that backbone.")
        raise exc

    model.to(torch.device(args.device))
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--model-id", type=str, default=DEFAULT_VLM_PATH)
    parser.add_argument("--load-vlm-weights", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--image-mode", choices=["gradient", "random", "zeros", "ones"], default="gradient")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smolvla-range", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task", type=str, default="grasp screwdriver and place it in box")
    parser.add_argument("--add-newline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenizer-max-length", type=int, default=None)
    parser.add_argument("--padding", type=str, default=None)
    parser.add_argument("--padding-side", choices=["left", "right"], default="right")
    parser.add_argument("--max-values", type=int, default=8)
    args = parser.parse_args()

    policy_cfg = load_policy_config(args.policy_path) if args.policy_path is not None else {}
    model_id = resolve_model_id(args, policy_cfg)
    model = make_model(args, policy_cfg, model_id)

    image = make_pseudo_image(args).to(device=torch.device(args.device), dtype=torch.float32)
    tokenizer_max_length = args.tokenizer_max_length or int(policy_cfg.get("tokenizer_max_length", 48))
    padding = args.padding or str(policy_cfg.get("pad_language_to", "max_length"))
    lang_tokens, lang_attention_mask, prompt_texts = make_language_inputs(
        model,
        args,
        tokenizer_max_length=tokenizer_max_length,
        padding=padding,
    )
    lang_tokens = lang_tokens.to(device=torch.device(args.device))
    lang_attention_mask = lang_attention_mask.to(device=torch.device(args.device))

    print_section("embed_image Input")
    summarize_tensor("image passed to embed_image", image, max_values=args.max_values)
    print("Expected range here is [-1, 1] when --smolvla-range is enabled.")

    with torch.no_grad():
        vision_model = model.get_vlm_model().vision_model
        connector = model.get_vlm_model().connector

        image_for_vision = image.to(dtype=vision_model.dtype)
        vision_output = vision_model(
            pixel_values=image_for_vision,
            patch_attention_mask=None,
        ).last_hidden_state

        connector_output = connector(vision_output)
        embed_image_output = model.embed_image(image)
        embed_language_output = model.embed_language_tokens(lang_tokens)

    print_section("Inside embed_image: Vision Encoder Input")
    summarize_tensor("image converted to vision_model.dtype", image_for_vision, max_values=args.max_values)

    print_section("Inside embed_image: Vision Encoder Output / Connector Input")
    summarize_tensor("vision_model(...).last_hidden_state", vision_output, max_values=args.max_values)

    print_section("Inside embed_image: Connector Output")
    summarize_tensor("connector(vision_hidden_states)", connector_output, max_values=args.max_values)

    print_section("embed_image Output")
    summarize_tensor("model.embed_image(image)", embed_image_output, max_values=args.max_values)

    if connector_output.shape == embed_image_output.shape:
        diff = (connector_output.float() - embed_image_output.float()).abs().max().item()
        print(f"\nmax_abs_diff(connector_output, embed_image_output)={diff:.8f}")

    print_section("embed_language_tokens Input")
    print(f"task_text={prompt_texts[0]!r}")
    print(f"tokenizer_max_length={tokenizer_max_length}")
    print(f"padding={padding}")
    print(f"padding_side={args.padding_side}")
    summarize_tensor("lang_tokens passed to embed_language_tokens", lang_tokens, max_values=args.max_values)
    summarize_tensor("lang_attention_mask", lang_attention_mask, max_values=args.max_values)

    print_section("embed_language_tokens Output")
    summarize_tensor("model.embed_language_tokens(lang_tokens)", embed_language_output, max_values=args.max_values)


if __name__ == "__main__":
    main()
