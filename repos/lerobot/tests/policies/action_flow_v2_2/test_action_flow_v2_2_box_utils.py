#!/usr/bin/env python

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "action_flow_v2"))

from action_flow_v2_2_box_utils import (  # noqa: E402
    compute_resize_geometry,
    load_episode_definitions,
    load_review_visibility_decisions,
    mask_bounds_from_runs,
    transform_bounds,
)


def encode_runs(mask: np.ndarray) -> bytes:
    flat = mask.reshape(-1).astype(np.uint8)
    padded = np.pad(flat, (1, 1))
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    runs = np.column_stack((transitions[::2], transitions[1::2] - transitions[::2]))
    return zlib.compress(runs.astype(np.uint32).tobytes())


def test_resize_geometry_matches_smolvla_left_top_padding():
    geometry = compute_resize_geometry(640, 480, 512, 512)

    assert geometry.resized_width == 512
    assert geometry.resized_height == 384
    assert geometry.pad_left == 0
    assert geometry.pad_top == 128


def test_mask_bounds_are_read_directly_from_compressed_runs():
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1:5, 2:7] = 1

    bounds = mask_bounds_from_runs(encode_runs(mask), height=6, width=8)

    assert bounds == (2, 1, 6, 4, 20)


def test_empty_mask_has_no_bounds():
    mask = np.zeros((4, 5), dtype=np.uint8)

    assert mask_bounds_from_runs(encode_runs(mask), height=4, width=5) is None


def test_full_native_box_maps_to_padded_policy_image_and_coordinate_bins():
    geometry = compute_resize_geometry(640, 480, 512, 512)

    normalized, bins = transform_bounds(
        (0, 0, 639, 479),
        geometry,
        coordinate_bins=256,
    )

    np.testing.assert_allclose(
        normalized,
        np.array([0.0, 128 / 511, 1.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(bins, np.array([0, 64, 255, 255]))


def test_quantized_box_preserves_xyxy_order():
    geometry = compute_resize_geometry(640, 480, 512, 512)

    normalized, bins = transform_bounds(
        (101, 52, 278, 311),
        geometry,
        coordinate_bins=256,
    )

    assert normalized[0] <= normalized[2]
    assert normalized[1] <= normalized[3]
    assert bins[0] <= bins[2]
    assert bins[1] <= bins[3]


def test_reviewed_manual_point_overrides_raw_not_visible(tmp_path):
    selected = [
        {
            "object_id": 1,
            "object_name": "tomato",
            "object_role": "source",
            "task_step": 1,
            "task": "Place the tomato.",
            "visible": False,
        }
    ]
    for camera in ("wrist", "top"):
        path = tmp_path / "json" / "episode-000000" / f"camera-{camera}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"selected": selected}))
    review_path = tmp_path / "review" / "decisions.json"
    review_path.parent.mkdir(parents=True)
    review_path.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "episode_index": 0,
                        "camera": camera,
                        "object_id": 1,
                        "status": "manual_point",
                    }
                    for camera in ("wrist", "top")
                ]
            }
        )
    )

    decisions = load_review_visibility_decisions(tmp_path)
    _, definitions, not_visible = load_episode_definitions(
        tmp_path,
        episode_index=0,
        cameras=("wrist", "top"),
        entity_name_to_id={"tomato": 0},
        review_decisions=decisions,
    )

    assert len(definitions) == 1
    assert not not_visible["wrist"]
    assert not not_visible["top"]
