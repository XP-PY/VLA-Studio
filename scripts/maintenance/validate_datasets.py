#!/usr/bin/env python3
"""Validate the curated dataset registry against datasets stored on disk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml


def load_registry(path: Path) -> dict[str, Any]:
    registry = yaml.safe_load(path.read_text())
    if not isinstance(registry, dict) or not isinstance(registry.get("datasets"), dict):
        raise ValueError(f"Invalid dataset registry: {path}")
    return registry


def lerobot_cameras(info: dict[str, Any]) -> list[str]:
    prefix = "observation.images."
    return [
        key.removeprefix(prefix)
        for key, feature in info.get("features", {}).items()
        if key.startswith(prefix) and feature.get("dtype") in {"image", "video"}
    ]


def check_equal(errors: list[str], dataset_id: str, field: str, expected: Any, actual: Any) -> None:
    if expected != actual:
        errors.append(f"{dataset_id}: {field} expected {expected!r}, found {actual!r}")


def validate_lerobot_dataset(
    dataset_id: str,
    root: Path,
    expected: dict[str, Any],
    errors: list[str],
) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        errors.append(f"{dataset_id}: missing {info_path}")
        return
    info = json.loads(info_path.read_text())
    check_equal(errors, dataset_id, "format", "v3.0", info.get("codebase_version"))
    for registry_field, info_field in (
        ("episodes", "total_episodes"),
        ("frames", "total_frames"),
        ("tasks", "total_tasks"),
        ("fps", "fps"),
    ):
        check_equal(errors, dataset_id, registry_field, expected.get(registry_field), info.get(info_field))
    check_equal(errors, dataset_id, "cameras", expected.get("cameras", []), lerobot_cameras(info))


def validate_annotation_dataset(
    dataset_id: str,
    root: Path,
    expected: dict[str, Any],
    errors: list[str],
) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        errors.append(f"{dataset_id}: missing {info_path}")
        return
    info = json.loads(info_path.read_text())
    check_equal(errors, dataset_id, "format_version", 1, info.get("format_version"))
    check_equal(errors, dataset_id, "fps", expected.get("fps"), info.get("fps"))
    check_equal(errors, dataset_id, "cameras", expected.get("cameras", []), info.get("cameras", []))

    data_root = root / "data"
    episode_dirs = sorted(path for path in data_root.glob("episode-*") if path.is_dir())
    parquet_files = sorted(data_root.glob("episode-*/camera-*.parquet"))
    annotation_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_files)
    check_equal(errors, dataset_id, "episodes", expected.get("episodes"), len(episode_dirs))
    check_equal(errors, dataset_id, "annotation_rows", expected.get("annotation_rows"), annotation_rows)
    check_equal(
        errors,
        dataset_id,
        "camera files",
        expected.get("episodes", 0) * len(expected.get("cameras", [])),
        len(parquet_files),
    )


def validate_lineage(datasets: dict[str, Any], errors: list[str]) -> None:
    for dataset_id, entry in datasets.items():
        for parent in entry.get("parents", []):
            if parent not in datasets:
                errors.append(f"{dataset_id}: unknown parent {parent!r}")

        if entry.get("build", {}).get("method") != "concatenate":
            continue
        parents = [datasets[parent] for parent in entry.get("parents", []) if parent in datasets]
        for field in ("episodes", "frames"):
            parent_total = sum(parent.get("metadata", {}).get(field, 0) for parent in parents)
            expected = entry.get("metadata", {}).get(field)
            check_equal(errors, dataset_id, f"sum(parent {field})", expected, parent_total)


def validate_coverage(dataset_root: Path, datasets: dict[str, Any], errors: list[str]) -> None:
    registered = {Path(entry["path"]) for entry in datasets.values()}
    discovered = {
        info_path.parent.parent.relative_to(dataset_root)
        for info_path in dataset_root.glob("**/meta/info.json")
    }
    for path in sorted(discovered - registered):
        errors.append(f"unregistered dataset on disk: {path}")
    for path in sorted(registered - discovered):
        errors.append(f"registered dataset missing on disk: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/datasets.yaml"))
    parser.add_argument("--dataset-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_registry(args.registry)
    dataset_root = (args.dataset_root or Path(registry["dataset_root"])).resolve()
    datasets = registry["datasets"]
    valid_roles = set(registry.get("roles", {}))
    valid_retention = set(registry.get("retention_states", {}))
    errors: list[str] = []

    for dataset_id, entry in datasets.items():
        if entry.get("role") not in valid_roles:
            errors.append(f"{dataset_id}: invalid role {entry.get('role')!r}")
        if entry.get("retention") not in valid_retention:
            errors.append(f"{dataset_id}: invalid retention {entry.get('retention')!r}")

        build_tool = entry.get("build", {}).get("tool")
        if build_tool and not Path(build_tool).is_file():
            errors.append(f"{dataset_id}: missing build tool {build_tool}")
        for source in entry.get("annotation_sources", []):
            if not Path(source).exists():
                errors.append(f"{dataset_id}: missing annotation source {source}")

        relative_path = Path(entry["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"{dataset_id}: path must remain relative to dataset_root")
            continue
        root = dataset_root / relative_path
        if entry.get("format") == "lerobot_v3":
            validate_lerobot_dataset(dataset_id, root, entry.get("metadata", {}), errors)
        elif entry.get("format") == "afv2_mask_annotation_v1":
            validate_annotation_dataset(dataset_id, root, entry.get("metadata", {}), errors)
        else:
            errors.append(f"{dataset_id}: unsupported format {entry.get('format')!r}")

    validate_lineage(datasets, errors)
    validate_coverage(dataset_root, datasets, errors)

    if errors:
        print(f"Dataset registry validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    retention_counts: dict[str, int] = {}
    for entry in datasets.values():
        retention = entry["retention"]
        retention_counts[retention] = retention_counts.get(retention, 0) + 1
    counts = ", ".join(f"{key}={value}" for key, value in sorted(retention_counts.items()))
    print(f"Validated {len(datasets)} datasets under {dataset_root}")
    print(f"Retention summary: {counts}")


if __name__ == "__main__":
    main()
