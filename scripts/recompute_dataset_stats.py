#!/usr/bin/env python
"""Recompute exact tabular statistics for a LeRobot dataset.

LeRobot can merge means and variances exactly from per-episode statistics, but
quantiles cannot be merged by taking a weighted average. This utility reads the
final parquet data and computes every numeric feature's statistics globally.
It can also replace action statistics with the relative-action distribution
seen by PI0-family policies during training.

Image and video statistics are preserved because LeRobot intentionally computes
them from sampled pixels. They are not used to normalize PI0-family inputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}
NON_TABULAR_DTYPES = {"image", "video", "string", "language"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="LeRobot dataset root.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Defaults to <root>/meta/stats.json.",
    )
    parser.add_argument(
        "--relative-actions",
        action="store_true",
        help="Compute action statistics after subtracting the state at each chunk start.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Training action chunk size used for relative-action statistics.",
    )
    parser.add_argument(
        "--relative-exclude-joints",
        nargs="*",
        default=["gripper"],
        help="Action-name tokens that remain absolute in relative-action mode.",
    )
    return parser.parse_args()


def series_to_array(series: pd.Series, feature_shape: list[int]) -> np.ndarray:
    values = series.to_numpy()
    if len(values) == 0:
        raise ValueError(f"Feature {series.name!r} has no values")

    first = values[0]
    if np.isscalar(first):
        array = np.asarray(values)
        if feature_shape == [1]:
            array = array.reshape(-1, 1)
        return array

    return np.stack(values)


def compute_exact_stats(values: np.ndarray) -> dict[str, list]:
    if values.shape[0] == 0:
        raise ValueError("Cannot compute statistics for an empty array")

    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        bad_count = int(values.size - np.isfinite(values).sum())
        raise ValueError(f"Found {bad_count} non-finite values")

    stats = {
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "count": np.array([values.shape[0]], dtype=np.int64),
    }
    for name, quantile in QUANTILES.items():
        stats[name] = np.quantile(values, quantile, axis=0)

    return {name: np.asarray(value).tolist() for name, value in stats.items()}


def load_tabular_features(root: Path, features: dict) -> dict[str, np.ndarray]:
    feature_names = [
        name for name, feature in features.items() if feature["dtype"] not in NON_TABULAR_DTYPES
    ]
    parquet_files = sorted((root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in feature_names}
    for parquet_path in parquet_files:
        frame = pd.read_parquet(parquet_path, columns=feature_names)
        for name in feature_names:
            chunks[name].append(series_to_array(frame[name], features[name]["shape"]))

    arrays = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Numeric feature lengths do not match: {lengths}")
    return arrays


def build_relative_mask(action_names: list[str] | None, action_dim: int, excluded: list[str]) -> np.ndarray:
    if not excluded or action_names is None:
        return np.ones(action_dim, dtype=bool)

    tokens = [token.lower() for token in excluded if token]
    mask = []
    for name in action_names[:action_dim]:
        normalized_name = str(name).lower()
        mask.append(not any(token == normalized_name or token in normalized_name for token in tokens))
    mask.extend([True] * (action_dim - len(mask)))
    return np.asarray(mask, dtype=bool)


def compute_relative_action_stats(
    arrays: dict[str, np.ndarray],
    features: dict,
    chunk_size: int,
    excluded: list[str],
) -> dict[str, list]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    required = {"action", "observation.state", "episode_index"}
    missing = required - arrays.keys()
    if missing:
        raise KeyError(f"Relative-action statistics require features: {sorted(missing)}")

    actions = arrays["action"]
    states = arrays["observation.state"]
    episode_indices = arrays["episode_index"].reshape(-1)
    if actions.ndim != 2 or states.ndim != 2:
        raise ValueError(
            "Relative-action statistics require 2-D action and state arrays, "
            f"got {actions.shape} and {states.shape}"
        )

    max_start = len(actions) - chunk_size
    if max_start < 0:
        raise ValueError(f"Dataset has fewer frames ({len(actions)}) than chunk_size ({chunk_size})")

    starts = np.arange(max_start + 1, dtype=np.int64)
    starts = starts[episode_indices[starts] == episode_indices[starts + chunk_size - 1]]
    if len(starts) == 0:
        raise ValueError(f"No complete action chunks found for chunk_size={chunk_size}")

    action_dim = actions.shape[1]
    if states.shape[1] < action_dim:
        raise ValueError(f"State dimension {states.shape[1]} is smaller than action dimension {action_dim}")

    mask = build_relative_mask(features["action"].get("names"), action_dim, excluded)
    frame_indices = starts[:, None] + np.arange(chunk_size, dtype=np.int64)[None, :]

    per_stat = {name: np.empty(action_dim, dtype=np.float64) for name in ["min", "max", "mean", "std"]}
    per_stat.update({name: np.empty(action_dim, dtype=np.float64) for name in QUANTILES})

    for dim in range(action_dim):
        values = actions[frame_indices, dim].astype(np.float64, copy=True)
        if mask[dim]:
            values -= states[starts, dim, None]
        values = values.reshape(-1)

        per_stat["min"][dim] = np.min(values)
        per_stat["max"][dim] = np.max(values)
        per_stat["mean"][dim] = np.mean(values)
        per_stat["std"][dim] = np.std(values)
        for name, quantile in QUANTILES.items():
            per_stat[name][dim] = np.quantile(values, quantile)

    result = {name: value.tolist() for name, value in per_stat.items()}
    result["count"] = [int(len(starts) * chunk_size)]
    return result


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    output_path = args.output.expanduser().resolve() if args.output else stats_path

    with info_path.open() as file:
        features = json.load(file)["features"]
    with stats_path.open() as file:
        stats = json.load(file)

    arrays = load_tabular_features(root, features)
    for name, values in arrays.items():
        stats[name] = compute_exact_stats(values)

    if args.relative_actions:
        stats["action"] = compute_relative_action_stats(
            arrays,
            features,
            args.chunk_size,
            args.relative_exclude_joints,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w") as file:
        json.dump(stats, file, indent=4)
        file.write("\n")
    os.replace(temporary_path, output_path)

    mode = "relative-action" if args.relative_actions else "absolute-action"
    print(f"Wrote exact numeric statistics ({mode}) to {output_path}")


if __name__ == "__main__":
    main()
