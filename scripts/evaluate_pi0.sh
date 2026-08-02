#!/usr/bin/env bash
# Multi-prompt evaluator for any compatible LeRobot policy checkpoint.
#
# By default, settings come from scripts/env.sh. To evaluate another policy
# without editing that file:
#   EVAL_POLICY_ID=<training-output-name> ./scripts/evaluate_pi0.sh
#
# EVAL_POLICY_PATH may instead point directly to a pretrained_model directory.
set -eo pipefail

unset http_proxy
unset https_proxy
unset no_proxy
unset NO_PROXY
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
set -u

SELECTED_POLICY_ID="${EVAL_POLICY_ID:-$POLICY_ID}"
CHECKPOINT_ROOT="${EVAL_CHECKPOINT_ROOT:-$STORAGE_ROOT/runs/train/$SELECTED_POLICY_ID/checkpoints}"
if [[ -n "${EVAL_POLICY_PATH:-}" ]]; then
    POLICY_PATH="$EVAL_POLICY_PATH"
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
            -printf '%f\n' |
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

EVAL_N_ACTION_STEPS="${EVAL_N_ACTION_STEPS:-${N_ACTION_STEPS:-}}"
EVAL_NUM_STEPS="${EVAL_NUM_STEPS:-${NUM_INFERENCE_STEP:-}}"

POLICY_ARGS=("--policy.path=$POLICY_PATH")
if grep -q '"n_action_steps"[[:space:]]*:' "$POLICY_CONFIG" && [[ -n "$EVAL_N_ACTION_STEPS" ]]; then
    POLICY_ARGS+=("--policy.n_action_steps=$EVAL_N_ACTION_STEPS")
fi

STEP_OVERRIDE="checkpoint default"
if [[ -n "$EVAL_NUM_STEPS" ]]; then
    if grep -q '"num_inference_steps"[[:space:]]*:' "$POLICY_CONFIG"; then
        POLICY_ARGS+=("--policy.num_inference_steps=$EVAL_NUM_STEPS")
        STEP_OVERRIDE="num_inference_steps=$EVAL_NUM_STEPS"
    elif grep -q '"num_steps"[[:space:]]*:' "$POLICY_CONFIG"; then
        POLICY_ARGS+=("--policy.num_steps=$EVAL_NUM_STEPS")
        STEP_OVERRIDE="num_steps=$EVAL_NUM_STEPS"
    fi
fi

EVAL_INFERENCE_TYPE="${EVAL_INFERENCE_TYPE:-$ROLLOUT_INFERENCE_TYPE}"
INFERENCE_ARGS=("--inference.type=$EVAL_INFERENCE_TYPE")
if [[ "$EVAL_INFERENCE_TYPE" == "rtc" ]]; then
    INFERENCE_ARGS+=(
        "--inference.rtc.execution_horizon=$RTC_EXECUTION_HORIZON"
        "--inference.rtc.max_guidance_weight=$RTC_MAX_GUIDANCE_WEIGHT"
        "--inference.rtc.prefix_attention_schedule=$RTC_PREFIX_ATTENTION_SCHEDULE"
        "--inference.queue_threshold=$RTC_QUEUE_THRESHOLD"
    )
fi

HEATMAP_ARGS=()
if [[ "$DETECTED_POLICY_TYPE" =~ ^action_flow_v2(_1)?$ && "${EVAL_SAVE_HEATMAP_VIDEOS:-true}" == "true" ]]; then
    HEATMAP_ARGS+=(
        "--save-heatmap-videos"
        "--heatmap-overlay-alpha=${EVAL_HEATMAP_OVERLAY_ALPHA:-0.45}"
    )
fi

SESSION_TIME="$(date +%Y%m%d_%H%M%S)"
EVAL_ID="rollout_rm65_action_flow_v22_multi_prompt_${SESSION_TIME}"
# EVAL_ID="rollout_${SELECTED_POLICY_ID}_multi_prompt"
EVAL_DIR="$STORAGE_ROOT/runs/rollout_dataset/$EVAL_ID"
LOG_PATH="$STORAGE_ROOT/logs/evaluate_${SELECTED_POLICY_ID}.log"
mkdir -p "$STORAGE_ROOT/runs/rollout_dataset" "$STORAGE_ROOT/logs"

EVAL_RESUME=false
if [[ -e "$EVAL_DIR" ]]; then
    if [[ ! -d "$EVAL_DIR" || ! -f "$EVAL_DIR/meta/info.json" ]]; then
        echo "Error: evaluation path exists but is not a resumable LeRobot dataset: $EVAL_DIR" >&2
        exit 1
    fi
    EVAL_RESUME=true
fi

EVAL_EPISODE_TIME_S="${EVAL_EPISODE_TIME_S:-${PI0_EVAL_EPISODE_TIME_S:-300}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

COMMAND=(
    "$PYTHON_BIN"
    -u
    "$PROJECT_ROOT/examples/evaluate_pi0_prompts.py"
    "--eval-prompt=$TASK_PROMPT"
    "${HEATMAP_ARGS[@]}"
    "--config_path=$ROBOT_CONFIG_PATH"
    "--strategy.type=episodic"
    "--strategy.reset_to_initial_position=${EVAL_RESET_TO_INITIAL_POSITION:-true}"
    "--resume=$EVAL_RESUME"
    "${INFERENCE_ARGS[@]}"
    "--robot.single_arm.canfd_follow=$ROLLOUT_CANFD_FOLLOW"
    "--robot.single_arm.max_relative_target=1000.0"
    "${POLICY_ARGS[@]}"
    "--dataset.repo_id=local/$EVAL_ID"
    "--dataset.root=$EVAL_DIR"
    "--dataset.single_task=$TASK_PROMPT"
    "--dataset.fps=$ROLLOUT_FPS"
    "--dataset.episode_time_s=$EVAL_EPISODE_TIME_S"
    "--dataset.one_video_per_episode=true"
    "--dataset.push_to_hub=false"
    "--device=cuda"
    "--fps=$ROLLOUT_FPS"
    "--task=$TASK_PROMPT"
    "--return_to_initial_position=${EVAL_RETURN_TO_INITIAL_POSITION:-true}"
    "$@"
)

echo "Policy type: $DETECTED_POLICY_TYPE"
echo "Checkpoint: $CHECKPOINT_NAME ($POLICY_PATH)"
echo "Inference: $EVAL_INFERENCE_TYPE; policy step override: $STEP_OVERRIDE"
echo "Evaluation dataset: $EVAL_DIR"
echo "Dataset mode: $([[ "$EVAL_RESUME" == "true" ]] && echo "resume" || echo "create")"
echo "Log: $LOG_PATH"

if [[ "${EVALUATE_DRY_RUN:-false}" == "true" ]]; then
    printf "Command:"
    printf " %q" "${COMMAND[@]}"
    printf "\n"
    exit 0
fi

"${COMMAND[@]}" 2>&1 | tee "$LOG_PATH"
