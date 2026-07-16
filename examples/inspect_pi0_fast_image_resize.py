#!/usr/bin/env python

from pathlib import Path

import torch
from torchvision.utils import save_image

from lerobot.policies.pi0_fast.modeling_pi0_fast import resize_with_pad_torch


OUTPUT_DIR = Path(__file__).resolve().parent
TARGET_HEIGHT = 224
TARGET_WIDTH = 224


def make_pseudo_image(height: int = 480, width: int = 640) -> torch.Tensor:
    """Create one float32 RGB image in PI0-FAST's channels-last input format."""
    image = torch.zeros(1, height, width, 3, dtype=torch.float32)
    image[..., 0] = 0.8  # Red background.
    image[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4, 1] = 1.0
    image[:, :, : width // 8, 2] = 1.0
    image[:, : height // 6, 3 * width // 4 :, 0:2] = 1.0
    return image


def print_resize_geometry(input_height: int, input_width: int) -> None:
    ratio = max(input_width / TARGET_WIDTH, input_height / TARGET_HEIGHT)
    resized_height = int(input_height / ratio)
    resized_width = int(input_width / ratio)

    pad_h0, pad_h1_extra = divmod(TARGET_HEIGHT - resized_height, 2)
    pad_w0, pad_w1_extra = divmod(TARGET_WIDTH - resized_width, 2)

    print(f"[INFO] resize ratio: {ratio:.6f}")
    print(f"[INFO] resized content: height={resized_height}, width={resized_width}")
    print(
        "[INFO] padding: "
        f"top={pad_h0}, bottom={pad_h0 + pad_h1_extra}, "
        f"left={pad_w0}, right={pad_w0 + pad_w1_extra}"
    )


def print_tensor(name: str, tensor: torch.Tensor) -> None:
    print(
        f"[INFO] {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"min={tensor.min().item():.3f}, max={tensor.max().item():.3f}"
    )


def main() -> None:
    input_bhwc = make_pseudo_image()
    input_bchw = input_bhwc.permute(0, 3, 1, 2)

    output_bhwc = resize_with_pad_torch(input_bhwc, TARGET_HEIGHT, TARGET_WIDTH)
    output_bchw = resize_with_pad_torch(input_bchw, TARGET_HEIGHT, TARGET_WIDTH)
    output_bchw_as_bhwc = output_bchw.permute(0, 2, 3, 1)

    print_resize_geometry(input_bhwc.shape[1], input_bhwc.shape[2])
    print_tensor("channels-last input", input_bhwc)
    print_tensor("channels-last output", output_bhwc)
    print_tensor("channels-first input", input_bchw)
    print_tensor("channels-first output", output_bchw)

    max_format_difference = (output_bhwc - output_bchw_as_bhwc).abs().max().item()
    print(f"[INFO] maximum difference between format outputs: {max_format_difference:.6f}")
    print(f"[INFO] first output row maximum (padding check): {output_bhwc[:, 0].max().item():.3f}")

    input_path = OUTPUT_DIR / "inspect_pi0_fast_image_resize_input.png"
    output_path = OUTPUT_DIR / "inspect_pi0_fast_image_resize_output.png"
    save_image(input_bchw, input_path)
    save_image(output_bchw, output_path)

    print(f"[INFO] saved input image: {input_path}")
    print(f"[INFO] saved resized image: {output_path}")


if __name__ == "__main__":
    main()
