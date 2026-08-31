#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

SELECTED_POLICY_ID="${ROLLOUT_POLICY_ID:-$POLICY_ID}"
CHECKPOINT_ROOT="${ROLLOUT_CHECKPOINT_ROOT:-$STORAGE_ROOT/runs/train/$SELECTED_POLICY_ID/checkpoints}"

if [[ -n "${ROLLOUT_POLICY_PATH:-}" ]]; then
    POLICY_PATH="$ROLLOUT_POLICY_PATH"
    CHECKPOINT_NAME="custom"
elif [[ -d "$CHECKPOINT_ROOT/last/pretrained_model" ]]; then
    CHECKPOINT_NAME="last"
    POLICY_PATH="$CHECKPOINT_ROOT/$CHECKPOINT_NAME/pretrained_model"
else
    CHECKPOINT_NAME=$(
        find "$CHECKPOINT_ROOT" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -printf '%f\n' 2>/dev/null |
        grep -E '^[0-9]+$' |
        sort -n |
        tail -n 1 || true
    )
    if [[ -z "$CHECKPOINT_NAME" ]]; then
        echo "Error: no checkpoint found under $CHECKPOINT_ROOT" >&2
        exit 1
    fi
    POLICY_PATH="$CHECKPOINT_ROOT/$CHECKPOINT_NAME/pretrained_model"
fi

if [[ ! -d "$POLICY_PATH" ]]; then
    echo "Error: pretrained model directory does not exist: $POLICY_PATH" >&2
    exit 1
fi

POLICY_CONFIG="$POLICY_PATH/config.json"
if [[ ! -f "$POLICY_CONFIG" ]]; then
    echo "Error: policy config does not exist: $POLICY_CONFIG" >&2
    exit 1
fi

DETECTED_POLICY_TYPE=$(
    sed -nE 's/^[[:space:]]*"type":[[:space:]]*"([^"]+)".*/\1/p' "$POLICY_CONFIG" |
        head -n 1
)
DETECTED_POLICY_TYPE="${DETECTED_POLICY_TYPE:-unknown}"

POLICY_ARGS=("--policy.path=$POLICY_PATH")
if grep -q '"n_action_steps"[[:space:]]*:' "$POLICY_CONFIG"; then
    POLICY_ARGS+=("--policy.n_action_steps=${ROLLOUT_N_ACTION_STEPS:-$N_ACTION_STEPS}")
fi

STEP_OVERRIDE="checkpoint default"
REQUESTED_NUM_STEPS="${ROLLOUT_NUM_STEPS:-$NUM_INFERENCE_STEP}"
if [[ -n "$REQUESTED_NUM_STEPS" ]]; then
    if grep -q '"num_inference_steps"[[:space:]]*:' "$POLICY_CONFIG"; then
        POLICY_ARGS+=("--policy.num_inference_steps=$REQUESTED_NUM_STEPS")
        STEP_OVERRIDE="num_inference_steps=$REQUESTED_NUM_STEPS"
    elif grep -q '"num_steps"[[:space:]]*:' "$POLICY_CONFIG"; then
        POLICY_ARGS+=("--policy.num_steps=$REQUESTED_NUM_STEPS")
        STEP_OVERRIDE="num_steps=$REQUESTED_NUM_STEPS"
    fi
fi

ROBOT_CONFIG_FILE="$ROBOT_CONFIG_PATH"
if [[ "$ROBOT_CONFIG_FILE" != /* ]]; then
    ROBOT_CONFIG_FILE="$PROJECT_ROOT/$ROBOT_CONFIG_FILE"
fi
if [[ ! -f "$ROBOT_CONFIG_FILE" ]]; then
    echo "Error: robot config does not exist: $ROBOT_CONFIG_FILE" >&2
    exit 1
fi

ROBOT_MODE=$(
    sed -nE 's/^[[:space:]]*mode:[[:space:]]*([^#[:space:]]+).*/\1/p' "$ROBOT_CONFIG_FILE" |
        head -n 1
)
ROBOT_ARGS=()
case "$ROBOT_MODE" in
    single)
        ROBOT_ARGS+=(
            "--robot.single_arm.canfd_follow=$ROLLOUT_CANFD_FOLLOW"
            "--robot.single_arm.max_relative_target=1000.0"
        )
        ;;
    bimanual)
        ROBOT_ARGS+=(
            "--robot.left_arm.canfd_follow=$ROLLOUT_CANFD_FOLLOW"
            "--robot.right_arm.canfd_follow=$ROLLOUT_CANFD_FOLLOW"
            "--robot.left_arm.max_relative_target=1000.0"
            "--robot.right_arm.max_relative_target=1000.0"
        )
        ;;
    *)
        echo "Error: unsupported or missing robot mode in $ROBOT_CONFIG_FILE: ${ROBOT_MODE:-missing}" >&2
        exit 1
        ;;
esac

SELECTED_INFERENCE_TYPE="${ROLLOUT_INFERENCE_TYPE:-rtc}"
INFERENCE_ARGS=("--inference.type=$SELECTED_INFERENCE_TYPE")
if [[ "$SELECTED_INFERENCE_TYPE" == "rtc" ]]; then
    INFERENCE_ARGS+=(
        "--inference.rtc.execution_horizon=$RTC_EXECUTION_HORIZON"
        "--inference.rtc.max_guidance_weight=$RTC_MAX_GUIDANCE_WEIGHT"
        "--inference.rtc.prefix_attention_schedule=$RTC_PREFIX_ATTENTION_SCHEDULE"
        "--inference.queue_threshold=$RTC_QUEUE_THRESHOLD"
    )
fi

SELECTED_TASK="${ROLLOUT_TASK:-$TASK_PROMPT}"
if [[ -n "${ROLLOUT_OUTPUT_DIR:-}" ]]; then
    SELECTED_ROLLOUT_DIR="$ROLLOUT_OUTPUT_DIR"
elif [[ -n "${ROLLOUT_POLICY_ID:-}" ]]; then
    SELECTED_ROLLOUT_DIR="$STORAGE_ROOT/runs/rollout_dataset/rollout_$SELECTED_POLICY_ID"
else
    SELECTED_ROLLOUT_DIR="$ROLLOUT_DIR"
fi
RESUME=false
if [[ -e "$SELECTED_ROLLOUT_DIR" ]]; then
    if [[ ! -d "$SELECTED_ROLLOUT_DIR" || ! -f "$SELECTED_ROLLOUT_DIR/meta/info.json" ]]; then
        echo "Error: rollout path exists but is not a resumable LeRobot dataset: $SELECTED_ROLLOUT_DIR" >&2
        exit 1
    fi
    RESUME=true
fi

LOG_PATH="${ROLLOUT_LOG_PATH:-$STORAGE_ROOT/logs/rollout_$SELECTED_POLICY_ID.log}"
COMMAND=(
    lerobot-rollout
    "--config_path=$ROBOT_CONFIG_FILE"
    "--strategy.type=$ROLLOUT_STRATEGY_TYPE"
    "${INFERENCE_ARGS[@]}"
    "${ROBOT_ARGS[@]}"
    "${POLICY_ARGS[@]}"
    "--dataset.repo_id=local/rollout_$SELECTED_POLICY_ID"
    "--dataset.root=$SELECTED_ROLLOUT_DIR"
    "--dataset.single_task=$SELECTED_TASK"
    "--dataset.push_to_hub=false"
    "--resume=$RESUME"
    "--device=cuda"
    "--fps=$ROLLOUT_FPS"
    "--duration=$ROLLOUT_DURATION"
    "--task=$SELECTED_TASK"
    "--return_to_initial_position=${ROLLOUT_RETURN_TO_INITIAL_POSITION:-false}"
    "$@"
)

echo "Experiment profile: $EXPERIMENT_NAME"
echo "Policy type: $DETECTED_POLICY_TYPE"
echo "Checkpoint: $CHECKPOINT_NAME ($POLICY_PATH)"
echo "Policy step override: $STEP_OVERRIDE"
echo "Robot mode: $ROBOT_MODE ($ROBOT_CONFIG_FILE)"
echo "Inference: $SELECTED_INFERENCE_TYPE"
echo "Rollout dataset: $SELECTED_ROLLOUT_DIR"
echo "Dataset mode: $([[ "$RESUME" == "true" ]] && echo resume || echo create)"
echo "Log: $LOG_PATH"

if [[ "${ROLLOUT_DRY_RUN:-false}" == "true" ]]; then
    printf "Command:"
    printf " %q" "${COMMAND[@]}"
    printf "\n"
    exit 0
fi

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$SELECTED_ROLLOUT_DIR")"
"${COMMAND[@]}" 2>&1 | tee "$LOG_PATH"
