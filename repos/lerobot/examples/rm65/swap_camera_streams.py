#!/usr/bin/env python
"""Swap two camera streams inside an existing LeRobot dataset.

Use this when camera serials were assigned to the wrong semantic names during
recording. The script swaps the video directories and the stored image
statistics for the two camera feature keys. It does not change robot state,
actions, episode indices, or task labels.

By default this is a dry run. Pass --apply to modify the dataset. With --apply,
the script creates a full backup next to the dataset unless --no-backup is set.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def _camera_key(name: str) -> str:
    if name.startswith("observation.images."):
        return name
    return f"observation.images.{name}"


def _swap_video_dirs(root: Path, first: str, second: str, apply: bool) -> None:
    videos_dir = root / "videos"
    first_dir = videos_dir / first
    second_dir = videos_dir / second
    tmp_dir = videos_dir / f"__swap_tmp_{first.replace('.', '_')}__"

    for path in (first_dir, second_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing video directory: {path}")
    if tmp_dir.exists():
        raise FileExistsError(f"Temporary swap path already exists: {tmp_dir}")

    print(f"Swap video directories:")
    print(f"  {first_dir}")
    print(f"  {second_dir}")

    if apply:
        first_dir.rename(tmp_dir)
        second_dir.rename(first_dir)
        tmp_dir.rename(second_dir)


def _swap_stats_json(root: Path, first: str, second: str, apply: bool) -> None:
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        print(f"Skip missing stats file: {stats_path}")
        return

    stats = json.loads(stats_path.read_text())
    if first not in stats or second not in stats:
        print(f"Skip stats.json swap; one key is missing: {first}, {second}")
        return

    print(f"Swap stats.json keys: {first} <-> {second}")
    if apply:
        stats[first], stats[second] = stats[second], stats[first]
        stats_path.write_text(json.dumps(stats, indent=4) + "\n")


def _swap_episode_stats(root: Path, first: str, second: str, apply: bool) -> None:
    episodes_dir = root / "meta" / "episodes"
    if not episodes_dir.exists():
        print(f"Skip missing episode metadata directory: {episodes_dir}")
        return

    first_prefix = f"stats/{first}/"
    second_prefix = f"stats/{second}/"
    parquet_paths = sorted(episodes_dir.rglob("*.parquet"))
    if not parquet_paths:
        print(f"Skip episode stats swap; no parquet files under: {episodes_dir}")
        return

    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        swapped = 0

        for first_col in [col for col in df.columns if col.startswith(first_prefix)]:
            suffix = first_col[len(first_prefix) :]
            second_col = f"{second_prefix}{suffix}"
            if second_col not in df.columns:
                continue

            swapped += 1
            if apply:
                tmp_col = f"__swap_tmp__{suffix}"
                while tmp_col in df.columns:
                    tmp_col = f"_{tmp_col}"
                df[tmp_col] = df[first_col]
                df[first_col] = df[second_col]
                df[second_col] = df[tmp_col]
                df = df.drop(columns=[tmp_col])

        print(f"Swap {swapped} episode stat columns in {parquet_path}")
        if apply and swapped:
            df.to_parquet(parquet_path, index=False)


def _create_backup(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = root.with_name(f"{root.name}.backup_before_camera_swap_{timestamp}")
    if backup_path.exists():
        raise FileExistsError(f"Backup path already exists: {backup_path}")

    print(f"Create dataset backup: {backup_path}")
    shutil.copytree(root, backup_path)
    return backup_path


def swap_camera_streams(root: Path, first: str, second: str, apply: bool, backup: bool) -> None:
    root = root.expanduser().resolve()
    first = _camera_key(first)
    second = _camera_key(second)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if first == second:
        raise ValueError("Camera keys must be different.")

    print(f"Dataset root: {root}")
    print(f"Mode: {'apply' if apply else 'dry-run'}")
    print(f"Camera keys: {first} <-> {second}")

    if apply and backup:
        _create_backup(root)

    _swap_video_dirs(root, first, second, apply=apply)
    _swap_stats_json(root, first, second, apply=apply)
    _swap_episode_stats(root, first, second, apply=apply)

    if apply:
        print("Camera stream swap complete.")
    else:
        print("Dry run complete. Re-run with --apply to modify the dataset.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="LeRobot dataset root to repair.",
    )
    parser.add_argument(
        "--first",
        default="front",
        help="First camera key or short name. Default: front.",
    )
    parser.add_argument(
        "--second",
        default="wrist",
        help="Second camera key or short name. Default: wrist.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Modify the dataset. Without this flag, only print planned changes.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a full dataset backup before applying changes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    swap_camera_streams(
        root=args.root,
        first=args.first,
        second=args.second,
        apply=args.apply,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
