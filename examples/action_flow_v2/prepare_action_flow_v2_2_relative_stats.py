#!/usr/bin/env python

"""Compute and install Action Flow V2.2 relative-action statistics.

Only the dataset's ``action`` statistics are replaced. Joint actions are
converted relative to the state at the start of each future chunk, while
dimensions selected by ``--exclude-joints`` remain absolute. Observation,
image, and auxiliary-label statistics are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.compute_stats import compute_relative_action_stats
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.processor import RelativeActionsProcessorStep
from lerobot.utils.constants import ACTION

DEFAULT_DATASET_ROOT = Path(
    "runs/dataset/rm65_single_table_preparation_and_bussing_afv2_2_yoloe_boxes"
)
DEFAULT_REPO_ID = "local/rm65_single_table_preparation_and_bussing_afv2_2_yoloe_boxes"
PROVENANCE_PATH = Path("meta/action_flow_v2_2_relative_stats.json")
ABSOLUTE_STATS_BACKUP_PATH = Path("meta/stats.absolute_actions.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--exclude-joints", nargs="*", default=["gripper"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute an already prepared dataset without replacing its absolute-statistics backup.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and validate statistics without modifying dataset metadata.",
    )
    return parser.parse_args()


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_stats(stats: dict[str, Any]) -> str:
    mean = np.asarray(stats["mean"])
    std = np.asarray(stats["std"])
    q01 = np.asarray(stats["q01"])
    q99 = np.asarray(stats["q99"])
    return (
        f"mean={np.array2string(mean, precision=4)}, "
        f"std={np.array2string(std, precision=4)}, "
        f"q01={np.array2string(q01, precision=4)}, "
        f"q99={np.array2string(q99, precision=4)}"
    )


def validate_dataset_contract(
    metadata: LeRobotDatasetMetadata,
    *,
    exclude_joints: list[str],
) -> tuple[list[str], list[bool]]:
    features = metadata.features
    if ACTION not in features or "observation.state" not in features:
        raise ValueError("Dataset must contain `action` and `observation.state`.")
    action_shape = tuple(features[ACTION]["shape"])
    state_shape = tuple(features["observation.state"]["shape"])
    if action_shape != (7,) or state_shape != (7,):
        raise ValueError(
            f"Expected RM65 action/state shapes `(7,)`, got {action_shape} and {state_shape}."
        )
    action_names = list(features[ACTION].get("names") or [])
    if len(action_names) != 7:
        raise ValueError("Every RM65 action dimension must have a name.")
    mask_step = RelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=exclude_joints,
        action_names=action_names,
    )
    relative_mask = mask_step._build_mask(len(action_names))
    if sum(relative_mask) != 6:
        raise ValueError(
            "Action Flow V2.2 expects six relative joint dimensions and one "
            f"absolute gripper dimension; mask is {relative_mask}."
        )
    return action_names, relative_mask


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    if args.chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    if args.num_workers < 0:
        raise ValueError("Number of workers cannot be negative.")

    provenance_path = args.dataset_root / PROVENANCE_PATH
    if provenance_path.is_file() and not args.force and not args.dry_run:
        provenance = json.loads(provenance_path.read_text())
        raise FileExistsError(
            f"Relative statistics already exist at {provenance_path} "
            f"(chunk_size={provenance.get('chunk_size')}). Use `--force` to recompute."
        )

    logging.info("Loading dataset metadata from %s", args.dataset_root)
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    action_names, relative_mask = validate_dataset_contract(
        metadata,
        exclude_joints=args.exclude_joints,
    )
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        video_backend="pyav",
    )
    logging.info(
        "Computing chunk-relative action statistics: chunk_size=%d, relative_mask=%s",
        args.chunk_size,
        relative_mask,
    )
    relative_stats = compute_relative_action_stats(
        hf_dataset=dataset.hf_dataset,
        features=metadata.features,
        chunk_size=args.chunk_size,
        exclude_joints=args.exclude_joints,
        num_workers=args.num_workers,
    )
    logging.info("Computed relative action statistics: %s", summarize_stats(relative_stats))

    if args.dry_run:
        logging.info("Dry run complete; dataset metadata was not modified.")
        return

    stats_path = args.dataset_root / "meta" / "stats.json"
    backup_path = args.dataset_root / ABSOLUTE_STATS_BACKUP_PATH
    if not backup_path.exists():
        shutil.copy2(stats_path, backup_path)
        logging.info("Backed up absolute-action statistics to %s", backup_path)

    previous_action_stats = metadata.stats[ACTION]
    updated_stats = dict(metadata.stats)
    updated_stats[ACTION] = relative_stats
    write_stats(updated_stats, args.dataset_root)

    provenance = {
        "format": "action_flow_v2_2_relative_action_stats",
        "version": 1,
        "dataset_root": str(args.dataset_root.resolve()),
        "repo_id": args.repo_id,
        "chunk_size": args.chunk_size,
        "action_names": action_names,
        "exclude_joints": args.exclude_joints,
        "relative_mask": relative_mask,
        "relative_dimensions": int(sum(relative_mask)),
        "absolute_dimensions": int(len(relative_mask) - sum(relative_mask)),
        "absolute_stats_backup": str(ABSOLUTE_STATS_BACKUP_PATH),
        "absolute_stats_backup_sha256": file_sha256(backup_path),
        "previous_action_stats": previous_action_stats,
        "relative_action_stats": relative_stats,
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, default=json_value) + "\n"
    )

    reloaded = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    for key, expected in relative_stats.items():
        np.testing.assert_allclose(
            np.asarray(reloaded.stats[ACTION][key]),
            np.asarray(expected),
            rtol=1e-6,
            atol=1e-6,
        )
    logging.info(
        "Installed relative-action statistics and verified reload. Provenance: %s",
        provenance_path,
    )


if __name__ == "__main__":
    main()
