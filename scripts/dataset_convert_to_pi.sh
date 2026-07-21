#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

PI_TASK_ID="${TASK_ID}_pi"
PI_TASK_ROOT="${TASK_ROOT}_pi"
CHUNK_SIZE="${CHUNK_SIZES[0]}"

lerobot-edit-dataset \
    --repo_id="local/${TASK_ID}" \
    --root="${TASK_ROOT}" \
    --new_repo_id="local/${PI_TASK_ID}" \
    --new_root="${PI_TASK_ROOT}" \
    --operation.type=recompute_stats \
    --operation.relative_action=true \
    --operation.chunk_size="${CHUNK_SIZE}" \
    --operation.relative_exclude_joints='["gripper"]' \
    --operation.num_workers=4

python "$SCRIPT_DIR/recompute_dataset_stats.py" \
    --root "$PI_TASK_ROOT" \
    --relative-actions \
    --chunk-size "$CHUNK_SIZE" \
    --relative-exclude-joints gripper
