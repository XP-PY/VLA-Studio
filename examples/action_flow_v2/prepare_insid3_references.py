#!/usr/bin/env python3
"""Create an INSID3 reference manifest from reviewed YOLOE reference boxes.

The generated rectangle masks are provisional visual prompts. They are useful
for testing INSID3 because the existing boxes are tight, but they must be
reviewed before full dataset generation. A later precise mask can replace any
PNG without changing the manifest schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = [
    WORKSPACE / "examples/action_flow_v2/ref/yoloe_reference_lerobot.yaml",
    WORKSPACE
    / "examples/action_flow_v2/ref/yoloe_reference_realsense_viewer_with_collection_box.yaml",
]
DEFAULT_OUTPUT_ROOT = WORKSPACE / "artifacts/insid3/references/rm65_tabletop"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-configs",
        type=Path,
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Earlier files take precedence when an entity appears more than once.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.output_root / "references.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite to rebuild it")

    args.output_root.mkdir(parents=True, exist_ok=True)
    mask_root = args.output_root / "masks"
    overlay_root = args.output_root / "overlays"
    mask_root.mkdir(exist_ok=True)
    overlay_root.mkdir(exist_ok=True)

    references: dict[str, dict] = {}
    for config_path in args.reference_configs:
        config_path = config_path.resolve()
        payload = yaml.safe_load(config_path.read_text())
        image_path = (config_path.parent / payload["reference"]["image"]).resolve()
        image = Image.open(image_path).convert("RGB")
        expected_size = (
            int(payload["reference"]["width"]),
            int(payload["reference"]["height"]),
        )
        if image.size != expected_size:
            raise ValueError(f"{image_path} has size {image.size}; expected {expected_size}")

        for item in payload["objects"]:
            entity = str(item.get("display_name") or item["name"]).replace("_", " ")
            if entity in references:
                continue
            box = [int(value) for value in item["bbox_xyxy"]]
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
                raise ValueError(f"Invalid reference box for {entity}: {box}")

            slug = str(item["name"])
            mask_path = mask_root / f"{slug}.png"
            overlay_path = overlay_root / f"{slug}.jpg"
            mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(mask).rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
            mask.save(mask_path)

            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="#ff304f", width=3)
            draw.text((x0, max(0, y0 - 16)), entity, fill="#ff304f")
            overlay.save(overlay_path)
            references[entity] = {
                "reference_images": [str(image_path)],
                "reference_masks": [str(mask_path.resolve())],
                "source_bbox_xyxy": box,
                "mask_method": "reviewed_box_rectangle",
            }

    manifest = {
        "format": "insid3_rm65_references",
        "format_version": 1,
        "bbox_format": "xyxy_inclusive_exclusive",
        "warning": (
            "Rectangle masks are provisional. Review or replace them with precise masks before "
            "full-dataset generation."
        ),
        "references": references,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(references)} references to {manifest_path}")
    print("Entities:", ", ".join(sorted(references)))


if __name__ == "__main__":
    main()

