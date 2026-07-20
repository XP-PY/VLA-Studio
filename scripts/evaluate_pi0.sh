#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

CHECKPOINT_ROOT="$STORAGE_ROOT/runs/train/$POLICY_ID/checkpoints"
if [[ -d "$CHECKPOINT_ROOT/last/pretrained_model" ]]; then
    CHECKPOINT_NAME="last"
else
    CHECKPOINT_NAME=$(
        find "$CHECKPOINT_ROOT" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -printf '%f\n' |
        grep -E '^[0-9]+$' |
        sort -n |
        tail -n 1 || true
    )
fi

if [[ -z "$CHECKPOINT_NAME" ]]; then
    echo "Error: no checkpoint found under $CHECKPOINT_ROOT"
    exit 1
fi

POLICY_PATH="$CHECKPOINT_ROOT/$CHECKPOINT_NAME/pretrained_model"
if [[ ! -d "$POLICY_PATH" ]]; then
    echo "Error: pretrained model directory does not exist: $POLICY_PATH"
    exit 1
fi

SESSION_TIME="$(date +%Y%m%d_%H%M%S)"
EVAL_ID="rollout_${POLICY_ID}_multi_prompt_${SESSION_TIME}"
EVAL_DIR="$STORAGE_ROOT/runs/rollout_dataset/$EVAL_ID"
LOG_PATH="$STORAGE_ROOT/logs/evaluate_${POLICY_ID}_${SESSION_TIME}.log"
mkdir -p "$STORAGE_ROOT/runs/rollout_dataset" "$STORAGE_ROOT/logs"

python -u "$PROJECT_ROOT/examples/evaluate_pi0_prompts.py" \
    --eval-prompt="$TASK_PROMPT" \
    --config_path="$ROBOT_CONFIG_PATH" \
    --strategy.type=episodic \
    --inference.type="$ROLLOUT_INFERENCE_TYPE" \
    --inference.rtc.execution_horizon="$RTC_EXECUTION_HORIZON" \
    --inference.rtc.max_guidance_weight="$RTC_MAX_GUIDANCE_WEIGHT" \
    --inference.rtc.prefix_attention_schedule="$RTC_PREFIX_ATTENTION_SCHEDULE" \
    --inference.queue_threshold="$RTC_QUEUE_THRESHOLD" \
    --robot.single_arm.canfd_follow="$ROLLOUT_CANFD_FOLLOW" \
    --robot.single_arm.max_relative_target=1000.0 \
    --policy.path="$POLICY_PATH" \
    --policy.n_action_steps="$N_ACTION_STEPS" \
    --policy.num_inference_steps="$NUM_INFERENCE_STEP" \
    --dataset.repo_id="local/$EVAL_ID" \
    --dataset.root="$EVAL_DIR" \
    --dataset.single_task="$TASK_PROMPT" \
    --dataset.fps="$ROLLOUT_FPS" \
    --dataset.episode_time_s="${PI0_EVAL_EPISODE_TIME_S:-30}" \
    --dataset.push_to_hub=false \
    --device=cuda \
    --fps="$ROLLOUT_FPS" \
    --task="$TASK_PROMPT" \
    --return_to_initial_position=false \
    "$@" \
    2>&1 | tee "$LOG_PATH"
