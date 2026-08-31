#!/usr/bin/env python3
"""Prune training checkpoints according to an explicit retention policy.

The command is a dry-run unless ``--execute`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_MODES = {"keep_all", "full_latest", "model_only_latest"}


@dataclass(frozen=True)
class Removal:
    path: Path
    size_bytes: int
    reason: str


def directory_size(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
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


def load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text())
    default_mode = policy.get("default_mode")
    if default_mode not in VALID_MODES:
        raise ValueError(f"Invalid default_mode: {default_mode!r}")
    for name, mode in policy.get("runs", {}).items():
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode for {name}: {mode!r}")
    return policy


def checkpoint_dirs(run_dir: Path) -> list[Path]:
    checkpoints_root = run_dir / "checkpoints"
    if not checkpoints_root.is_dir():
        return []
    return sorted(
        (path for path in checkpoints_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )


def plan_run(run_dir: Path, mode: str) -> list[Removal]:
    checkpoints = checkpoint_dirs(run_dir)
    if mode == "keep_all" or not checkpoints:
        return []

    latest = checkpoints[-1]
    removals = [
        Removal(path, directory_size(path), f"older than latest checkpoint {latest.name}")
        for path in checkpoints[:-1]
    ]
    if mode == "model_only_latest":
        pretrained_model = latest / "pretrained_model"
        if not pretrained_model.is_dir():
            raise RuntimeError(f"Latest checkpoint has no pretrained_model: {latest}")
        removals.extend(
            Removal(path, directory_size(path), "historical run keeps inference model only")
            for path in latest.iterdir()
            if path.name != "pretrained_model"
        )
    return removals


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=Path("runs/train"))
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("scripts/maintenance/checkpoint_retention.json"),
    )
    parser.add_argument("--execute", action="store_true", help="Apply the displayed deletion plan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_root = args.train_root.resolve()
    policy = load_policy(args.policy)
    default_mode = policy["default_mode"]
    overrides = policy.get("runs", {})

    if not train_root.is_dir() or train_root.name != "train":
        raise RuntimeError(f"Refusing unexpected training root: {train_root}")

    existing_runs = {path.name for path in train_root.iterdir() if path.is_dir()}
    missing_overrides = sorted(set(overrides) - existing_runs)
    if missing_overrides:
        raise RuntimeError(f"Retention policy references missing runs: {missing_overrides}")

    removals: list[Removal] = []
    run_summaries = []
    for run_dir in sorted(path for path in train_root.iterdir() if path.is_dir()):
        mode = overrides.get(run_dir.name, default_mode)
        run_removals = plan_run(run_dir, mode)
        removals.extend(run_removals)
        run_summaries.append((run_dir.name, mode, len(run_removals), sum(item.size_bytes for item in run_removals)))

    print("Checkpoint retention plan")
    print(f"Training root: {train_root}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    for name, mode, count, size in run_summaries:
        print(f"  {mode:17} {count:3d} paths {human_size(size):>10}  {name}")
    print(f"Total: {len(removals)} paths, {human_size(sum(item.size_bytes for item in removals))}")

    if not args.execute:
        print("No files changed. Add --execute after reviewing this plan.")
        return

    for removal in removals:
        remove_path(removal.path)
    print(f"Removed {len(removals)} paths.")


if __name__ == "__main__":
    main()
