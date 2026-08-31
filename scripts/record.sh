#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/record_env.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_PATH="${STORAGE_ROOT}/logs/record_${RECORD_OUTPUT_ID}.log"

COMMAND=(
    "$PYTHON_BIN"
    "$PROJECT_ROOT/repos/lerobot/examples/rm65/passive_record.py"
    "--config-path=$RECORD_ROBOT_CONFIG_PATH"
    "--repo-id=local/$RECORD_OUTPUT_ID"
    "--root=$RECORD_OUTPUT_ROOT"
    "--task=$RECORD_TASK"
    "--fps=$RECORD_FPS"
    "--num-episodes=$RECORD_NUM_EPISODES"
    "--reset-time-s=$RECORD_RESET_TIME_S"
    "--future-offset=$RECORD_FUTURE_OFFSET"
    "--log-path=$LOG_PATH"
    "--io-read-source=$RECORD_IO_READ_SOURCE"
    "--teleop-active-io=$RECORD_TELEOP_ACTIVE_IO"
    "--record-button-io=$RECORD_BUTTON_IO"
    "--record-button-mode=$RECORD_BUTTON_MODE"
)

if [[ "$RECORD_ASK_SAVE" == "true" ]]; then
    COMMAND+=(--ask-save)
else
    COMMAND+=(--no-ask-save)
fi

echo "Recording profile: $RECORDING_NAME"
echo "Config: $RECORD_CONFIG"
echo "Dataset: $RECORD_OUTPUT_ROOT"
echo "Task: $RECORD_TASK"
echo "New episodes: $RECORD_NUM_EPISODES | FPS: $RECORD_FPS"
echo "Robot config: $RECORD_ROBOT_CONFIG_PATH"

if [[ "${RECORD_DRY_RUN:-false}" == "true" ]]; then
    printf "Command:"
    printf " %q" "${COMMAND[@]}"
    printf "\n"
    exit 0
fi

mkdir -p "$STORAGE_ROOT/logs"
"${COMMAND[@]}"
