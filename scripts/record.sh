#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

python "$PROJECT_ROOT/repos/lerobot/examples/rm65/passive_record.py" \
    --config-path="$ROBOT_CONFIG_PATH" \
    --repo-id="local/$TASK_ID" \
    --root="$TASK_ROOT" \
    --task="$TASK_PROMPT" \
    --fps="$TASK_FPS" \
    --num-episodes="$TASK_EPISODES" \
    --log-path="$STORAGE_ROOT/logs/record_$TASK_ID.log" \
    --io-read-source=aloha \
    --control-arm=right \
    --teleop-active-io=1 \
    --record-button-io=2 \
    --record-button-mode=toggle