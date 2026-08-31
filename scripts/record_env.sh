#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "$SCRIPT_DIR/common_env.sh"

export RECORD_CONFIG="${RECORD_CONFIG:-configs/recordings/single_table_place_one_vegetable.env}"
if [[ "$RECORD_CONFIG" != /* ]]; then
    export RECORD_CONFIG="$PROJECT_ROOT/$RECORD_CONFIG"
fi
if [[ ! -f "$RECORD_CONFIG" ]]; then
    echo "Recording config does not exist: $RECORD_CONFIG" >&2
    return 1
fi
# shellcheck source=/dev/null
source "$RECORD_CONFIG"

export RECORD_DATASET_ROOT="${RECORD_DATASET_ROOT:-$STORAGE_ROOT/runs/dataset/$RECORD_DATASET_ID}"
export RECORD_SUBSET_ID="${RECORD_SUBSET_ID:-}"
export RECORD_NUM_EPISODES="${RECORD_NUM_EPISODES:-1}"
export RECORD_FPS="${RECORD_FPS:-30}"
export RECORD_ROBOT_CONFIG_PATH="${RECORD_ROBOT_CONFIG_PATH:-$ROBOT_CONFIG_PATH}"
export RECORD_IO_READ_SOURCE="${RECORD_IO_READ_SOURCE:-aloha}"
export RECORD_TELEOP_ACTIVE_IO="${RECORD_TELEOP_ACTIVE_IO:-1}"
export RECORD_BUTTON_IO="${RECORD_BUTTON_IO:-2}"
export RECORD_BUTTON_MODE="${RECORD_BUTTON_MODE:-toggle}"
export RECORD_FUTURE_OFFSET="${RECORD_FUTURE_OFFSET:-1}"
export RECORD_RESET_TIME_S="${RECORD_RESET_TIME_S:-5.0}"
export RECORD_ASK_SAVE="${RECORD_ASK_SAVE:-true}"

for required_variable in RECORDING_NAME RECORD_REGISTRY_KEY RECORD_DATASET_ID RECORD_TASK; do
    if [[ -z "${!required_variable:-}" ]]; then
        echo "Recording config is missing $required_variable: $RECORD_CONFIG" >&2
        return 1
    fi
done

if [[ ! "$RECORD_NUM_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "RECORD_NUM_EPISODES must be a positive integer: $RECORD_NUM_EPISODES" >&2
    return 1
fi
if [[ ! "$RECORD_FPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "RECORD_FPS must be a positive integer: $RECORD_FPS" >&2
    return 1
fi
if [[ "$RECORD_ASK_SAVE" != "true" && "$RECORD_ASK_SAVE" != "false" ]]; then
    echo "RECORD_ASK_SAVE must be true or false: $RECORD_ASK_SAVE" >&2
    return 1
fi

if [[ "$RECORD_ROBOT_CONFIG_PATH" != /* ]]; then
    export RECORD_ROBOT_CONFIG_PATH="$PROJECT_ROOT/$RECORD_ROBOT_CONFIG_PATH"
fi

if [[ -n "$RECORD_SUBSET_ID" ]]; then
    export RECORD_OUTPUT_ID="${RECORD_DATASET_ID}_${RECORD_SUBSET_ID}"
    export RECORD_OUTPUT_ROOT="${RECORD_DATASET_ROOT}_subsets/${RECORD_OUTPUT_ID}"
else
    export RECORD_OUTPUT_ID="$RECORD_DATASET_ID"
    export RECORD_OUTPUT_ROOT="$RECORD_DATASET_ROOT"
fi
