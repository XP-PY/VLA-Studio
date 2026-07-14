#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo $PROJECT_ROOT
echo $STORAGE_ROOT

export PI0_TASK_ID="${TASK_ID}_three_cameras"
export PI0_TASK_ROOT="$STORAGE_ROOT/runs/dataset/$PI0_TASK_ID"

echo "Creating three-camera PI0 dataset: $PI0_TASK_ID"
HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_HOME/datasets" \
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id="local/$TASK_ID" \
    --root="$TASK_ROOT" \
    --new_repo_id="local/$PI0_TASK_ID" \
    --new_root="$PI0_TASK_ROOT" \
    --operation.type=remove_feature \
    --operation.feature_names="['observation.images.front']" \
    --push_to_hub=false