#!/usr/bin/env python3
"""Draw all configured YOLOE visual-prompt boxes on the reference image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from test_yoloe_visual_prompt_video import COLORS, draw_detection, load_reference_config




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("yoloe_reference_realsense_viewer_with_collection_box.yaml"),
        help="YOLOE reference YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("reference_boxes_realsense_viewer_with_collection_box.jpg"),
        # default=Path(__file__).parent / "runs" / "reference_boxes.jpg",
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_image, objects = load_reference_config(args.config.resolve())
    image = cv2.imread(str(reference_image))
    if image is None:
        raise RuntimeError(f"OpenCV could not read reference image: {reference_image}")

    for item in objects:
        draw_detection(
            image,
            item.bbox_xyxy,
            f"{item.class_id}: {item.display_name}",
            COLORS[item.class_id % len(COLORS)],
        )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not save reference visualization: {output_path}")

    print(f"Reference visualization: {output_path}")


if __name__ == "__main__":
    main()
