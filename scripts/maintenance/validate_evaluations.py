#!/usr/bin/env python3
"""Validate rollout and offline-evaluation groups against outputs on disk."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Invalid evaluation registry: {path}")
    return config


def directory_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except FileNotFoundError:
                pass
    return total


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def dataset_counts(path: Path) -> tuple[int, int]:
    info_path = path / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return 0, 0
    return int(info.get("total_episodes") or 0), int(info.get("total_frames") or 0)


def validate_groups(
    *,
    root: Path,
    groups: dict[str, Any],
    valid_retention: set[str],
    label: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    discovered = {path.name: path for path in root.iterdir() if path.is_dir()} if root.is_dir() else {}
    owners: dict[str, list[str]] = {name: [] for name in discovered}
    summaries = []

    for group_id, entry in groups.items():
        if entry.get("retention") not in valid_retention:
            errors.append(f"{label}.{group_id}: invalid retention {entry.get('retention')!r}")
        documented_in = entry.get("documented_in")
        if documented_in and not Path(documented_in).is_file():
            errors.append(f"{label}.{group_id}: missing document {documented_in}")

        matched: dict[str, Path] = {}
        for pattern in entry.get("patterns", []):
            for path in root.glob(pattern):
                if path.is_dir():
                    matched[path.name] = path
        if not matched:
            errors.append(f"{label}.{group_id}: patterns matched no directories")

        episodes = 0
        frames = 0
        size_bytes = 0
        for name, path in matched.items():
            owners.setdefault(name, []).append(group_id)
            path_episodes, path_frames = dataset_counts(path)
            episodes += path_episodes
            frames += path_frames
            size_bytes += directory_size(path)
        summaries.append(
            {
                "id": group_id,
                "status": entry.get("status", "unknown"),
                "retention": entry.get("retention"),
                "directories": len(matched),
                "episodes_in_metadata": episodes,
                "frames_in_metadata": frames,
                "size_bytes": size_bytes,
            }
        )

    for name, matched_groups in sorted(owners.items()):
        if not matched_groups:
            errors.append(f"{label}: unregistered directory {name}")
        elif len(matched_groups) > 1:
            errors.append(f"{label}: {name} belongs to multiple groups {matched_groups}")
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/evaluations.yaml"))
    parser.add_argument("--rollout-root", type=Path)
    parser.add_argument("--offline-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.registry)
    rollout_root = (args.rollout_root or Path(config["rollout_root"])).resolve()
    offline_root = (args.offline_root or Path(config["offline_evaluation_root"])).resolve()
    valid_retention = set(config.get("retention_states", {}))
    errors: list[str] = []

    rollout_summaries = validate_groups(
        root=rollout_root,
        groups=config.get("rollout_groups", {}),
        valid_retention=valid_retention,
        label="rollout_groups",
        errors=errors,
    )
    offline_summaries = validate_groups(
        root=offline_root,
        groups=config.get("offline_evaluation_groups", {}),
        valid_retention=valid_retention,
        label="offline_evaluation_groups",
        errors=errors,
    )

    if errors:
        print(f"Evaluation registry validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print(f"Validated rollout groups under {rollout_root}")
    for summary in rollout_summaries:
        print(
            f"  {summary['id']}: {summary['directories']} dirs, "
            f"metadata episodes={summary['episodes_in_metadata']}, "
            f"{human_size(summary['size_bytes'])}"
        )
    print(f"Validated offline evaluation groups under {offline_root}")
    for summary in offline_summaries:
        print(f"  {summary['id']}: {summary['directories']} dirs, {human_size(summary['size_bytes'])}")


if __name__ == "__main__":
    main()
