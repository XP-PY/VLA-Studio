#!/bin/bash

HF_HOME=./.cache/huggingface \
HF_DATASETS_CACHE=./.cache/huggingface/datasets \
PYTHONPATH=repos/lerobot/src

# Input subset dataset directory
SUBSET_ROOT="runs/dataset/rm65_bimanual_farthest_to_closest_assembly_parts_to_center_subset"

# Remove "_subset" to create output name
NEW_ROOT="${SUBSET_ROOT%_subset}"

# Repo id name
DATASET_NAME=$(basename "$NEW_ROOT")
NEW_REPO_ID="local/${DATASET_NAME}"

# Automatically collect subfolders
ROOTS=()
REPO_IDS=()

for dir in "$SUBSET_ROOT"/*; do
    if [ -d "$dir" ]; then
        ROOTS+=("$dir")

        SUBDIR_NAME=$(basename "$dir")
        REPO_IDS+=("local/${SUBDIR_NAME}")
    fi
done

# Convert bash arrays to python list format
ROOTS_STR=$(printf "'%s', " "${ROOTS[@]}")
ROOTS_STR="[${ROOTS_STR%, }]"

REPO_IDS_STR=$(printf "'%s', " "${REPO_IDS[@]}")
REPO_IDS_STR="[${REPO_IDS_STR%, }]"

echo "New repo id:"
echo "$NEW_REPO_ID"

echo "New root:"
echo "$NEW_ROOT"

echo "Roots:"
echo "$ROOTS_STR"

echo "Repo ids:"
echo "$REPO_IDS_STR"


python -m lerobot.scripts.lerobot_edit_dataset \
    --new_repo_id "$NEW_REPO_ID" \
    --new_root "$NEW_ROOT" \
    --operation.type merge \
    --operation.repo_ids "$REPO_IDS_STR" \
    --operation.roots "$ROOTS_STR" \
    --operation.concatenate_videos false \
    --operation.concatenate_data false \
    --push_to_hub false