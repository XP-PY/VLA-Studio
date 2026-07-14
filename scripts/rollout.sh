#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# ===== Check dir whether already exist =====

if [ -d "$ROLLOUT_DIR" ]; then
    RESUME=true
else
    RESUME=false
fi

# ===== Automatic choose checkpoint =====

CHECKPOINT_ROOT="$STORAGE_ROOT/runs/train/$POLICY_ID/checkpoints"

if [[ -d "$CHECKPOINT_ROOT/last/pretrained_model" ]]; then
    # Prefer the "last" checkpoint if it exists
    CHECKPOINT_NAME="last"
else
    # Find numeric checkpoint directories and select the largest one
    CHECKPOINT_NAME=$(
        find "$CHECKPOINT_ROOT" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -printf '%f\n' |
        grep -E '^[0-9]+$' |
        sort -n |
        tail -n 1
    )

    if [[ -z "$CHECKPOINT_NAME" ]]; then
        echo "Error: no 'last' or numeric checkpoint found in:"
        echo "$CHECKPOINT_ROOT"
        exit 1
    fi
fi

POLICY_PATH="$CHECKPOINT_ROOT/$CHECKPOINT_NAME/pretrained_model"

# Verify that the pretrained model directory exists
if [[ ! -d "$POLICY_PATH" ]]; then
    echo "Error: pretrained model directory does not exist:"
    echo "$POLICY_PATH"
    exit 1
fi

# ===== Rollout with RTC =====

lerobot-rollout \
    --config_path="$ROBOT_CONFIG_PATH" \
    --strategy.type=$ROLLOUT_STRATEGY_TYPE \
    --inference.type=$ROLLOUT_INFERENCE_TYPE \
    --inference.rtc.execution_horizon=$RTC_EXECUTION_HORIZON \
    --inference.rtc.max_guidance_weight=$RTC_MAX_GUIDANCE_WEIGHT \
    --inference.rtc.prefix_attention_schedule="$RTC_PREFIX_ATTENTION_SCHEDULE" \
    --inference.queue_threshold=$RTC_QUEUE_THRESHOLD \
    --robot.left_arm.canfd_follow=$ROLLOUT_CANFD_FOLLOW \
    --robot.right_arm.canfd_follow=$ROLLOUT_CANFD_FOLLOW \
    --robot.left_arm.max_relative_target=1000.0 \
    --robot.right_arm.max_relative_target=1000.0 \
    --policy.path="$POLICY_PATH" \
    --policy.n_action_steps=$N_ACTION_STEPS \
    --policy.num_inference_steps=5 \
    --dataset.repo_id="local/rollout_$POLICY_ID" \
    --dataset.root="$ROLLOUT_DIR" \
    --dataset.single_task="$TASK_PROMPT" \
    --dataset.push_to_hub=false \
    --resume=$RESUME \
    --device=cuda \
    --fps=$ROLLOUT_FPS \
    --duration=$ROLLOUT_DURATION \
    --task="$TASK_PROMPT" \
    --return_to_initial_position=false \
    2>&1 | tee "$STORAGE_ROOT/logs/rollout_$POLICY_ID.log"
