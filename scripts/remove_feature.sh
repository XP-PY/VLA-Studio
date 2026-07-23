SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

SOURCE_ID=rm65_bimanual_farthest_to_closest_assembly_parts_to_center
TARGET_ID=${SOURCE_ID}_no_top

PYTHONPATH=repos/lerobot/src python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/$SOURCE_ID \
    --root runs/dataset/$SOURCE_ID \
    --new_repo_id local/$TARGET_ID \
    --new_root runs/dataset/$TARGET_ID \
    --operation.type remove_feature \
    --operation.feature_names "['observation.images.top']" \
    --push_to_hub false

python "$SCRIPT_DIR/recompute_dataset_stats.py" \
    --root "runs/dataset/$TARGET_ID"