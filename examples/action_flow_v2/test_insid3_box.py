#!/usr/bin/env python3
"""Transfer a reference mask with INSID3 and convert it to a target box.

This is the acceptance test used before running INSID3 on RM65 keyframes. It
saves the binary prediction, an overlay with the largest-component box, and a
JSON record containing raw-image ``xyxy`` coordinates.

Example:

    HF_HOME=artifacts/huggingface \
    conda run -n insid3 python examples/action_flow_v2/test_insid3_box.py \
        --reference-image repos/INSID3/assets/ref_cat_image.jpg \
        --reference-mask repos/INSID3/assets/ref_cat_mask.png \
        --target-image repos/INSID3/assets/target_cat_image.jpg \
        --entity cat \
        --output-dir runs/insid3/test_cat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from insid3_compat import DEFAULT_INSID3_ROOT, DEFAULT_TIMM_MODEL, build_insid3_with_timm


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Retain the largest 8-connected foreground component."""

    labels, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if component_count == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.reshape(-1))
    counts[0] = 0
    return labels == int(counts.argmax())


def mask_to_xyxy(mask: np.ndarray) -> list[int] | None:
    """Return an inclusive-exclusive box in original target-image pixels."""

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def save_overlay(
    image: Image.Image,
    mask: np.ndarray,
    box: list[int] | None,
    output_path: Path,
) -> None:
    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    overlay = base.copy()
    color = np.array([20, 210, 120], dtype=np.uint8)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    rendered = Image.fromarray(overlay)
    if box is not None:
        draw = ImageDraw.Draw(rendered)
        draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline="#ff304f", width=3)
    rendered.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--reference-mask", type=Path, required=True)
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--insid3-root", type=Path, default=DEFAULT_INSID3_ROOT)
    parser.add_argument("--model-name", default=DEFAULT_TIMM_MODEL)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--svd-components", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--merge-threshold", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.reference_image, args.reference_mask, args.target_image):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_insid3_with_timm(
        insid3_root=args.insid3_root,
        model_name=args.model_name,
        image_size=args.image_size,
        svd_components=args.svd_components,
        tau=args.tau,
        merge_threshold=args.merge_threshold,
        device=args.device,
    )
    model.set_reference(str(args.reference_image), str(args.reference_mask))
    model.set_target(str(args.target_image))
    raw_mask = model.segment().detach().cpu().numpy().astype(bool)
    mask = largest_component(raw_mask)

    target = Image.open(args.target_image).convert("RGB")
    expected_shape = (target.height, target.width)
    if mask.shape != expected_shape:
        raise RuntimeError(f"INSID3 returned mask shape {mask.shape}; expected {expected_shape}")
    box = mask_to_xyxy(mask)
    foreground_pixels = int(mask.sum())

    mask_path = args.output_dir / "predicted_mask.png"
    overlay_path = args.output_dir / "prediction_overlay.jpg"
    metadata_path = args.output_dir / "prediction.json"
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    save_overlay(target, mask, box, overlay_path)
    metadata = {
        "entity": args.entity,
        "reference_image": str(args.reference_image.resolve()),
        "reference_mask": str(args.reference_mask.resolve()),
        "target_image": str(args.target_image.resolve()),
        "image_width": target.width,
        "image_height": target.height,
        "bbox_format": "xyxy_inclusive_exclusive",
        "bbox_xyxy": box,
        "visible": box is not None,
        "foreground_pixels": foreground_pixels,
        "foreground_fraction": foreground_pixels / (target.width * target.height),
        "model_backend": "timm_dinov3",
        "model_name": args.model_name,
        "image_size": args.image_size,
        "svd_components": args.svd_components,
        "tau": args.tau,
        "merge_threshold": args.merge_threshold,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Mask: {mask_path}")
    print(f"Overlay: {overlay_path}")


if __name__ == "__main__":
    main()
