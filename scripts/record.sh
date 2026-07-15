#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

if [ -z "$TASK_SUBSET_ID" ]; then
    MODIFIED_ID="${TASK_ID}"
    MODIFIED_ROOT="${TASK_ROOT}"
else
    MODIFIED_ID="${TASK_ID}_${TASK_SUBSET_ID}"
    MODIFIED_ROOT="${TASK_ROOT}_subsets/${MODIFIED_ID}"
fi

python "$PROJECT_ROOT/repos/lerobot/examples/rm65/passive_record.py" \
    --config-path="${ROBOT_CONFIG_PATH}" \
    --repo-id="local/${MODIFIED_ID}" \
    --root="${MODIFIED_ROOT}" \
    --task="${TASK_PROMPT}" \
    --fps=$TASK_FPS \
    --num-episodes="${TASK_EPISODES}" \
    --log-path="${STORAGE_ROOT}/logs/record_${MODIFIED_ID}.log" \
    --io-read-source=aloha \
    --teleop-active-io=1 \
    --record-button-io=2 \
    --record-button-mode=toggle
