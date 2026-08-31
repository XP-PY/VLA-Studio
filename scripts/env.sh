#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "$SCRIPT_DIR/common_env.sh"

# ===== Experiment profile =====
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-configs/experiments/action_flow_v2_3_no_ki.env}"
if [[ "$EXPERIMENT_CONFIG" != /* ]]; then
    export EXPERIMENT_CONFIG="$PROJECT_ROOT/$EXPERIMENT_CONFIG"
fi
if [[ ! -f "$EXPERIMENT_CONFIG" ]]; then
    echo "Experiment config does not exist: $EXPERIMENT_CONFIG" >&2
    return 1
fi
# shellcheck source=/dev/null
source "$EXPERIMENT_CONFIG"

# Common fallbacks for custom profiles. Policy-specific values belong in the
# selected profile rather than this file.
export TASK_ROOT="${TASK_ROOT:-$STORAGE_ROOT/runs/dataset/$TASK_ID}"
export TASK_SUBSET_ID="${TASK_SUBSET_ID:-}"
export TASK_FPS="${TASK_FPS:-30}"
export SAVE_FREQ="${SAVE_FREQ:-20000}"
export LOG_FREQ="${LOG_FREQ:-100}"
export GPU_IDS="${GPU_IDS:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"
export TRAIN_MIXED_PRECISION="${TRAIN_MIXED_PRECISION:-bf16}"
if ! declare -p CHUNK_SIZES >/dev/null 2>&1; then
    CHUNK_SIZES=(30)
fi
if [[ -n "${CHUNK_SIZE:-}" ]]; then
    CHUNK_SIZES=("$CHUNK_SIZE")
fi

for required_variable in EXPERIMENT_NAME POLICY_ID POLICY_TYPE TASK_ID TASK_PROMPT TASK_EPISODES STEPS PER_DEVICE_BATCH_SIZE; do
    if [[ -z "${!required_variable:-}" ]]; then
        echo "Experiment config is missing $required_variable: $EXPERIMENT_CONFIG" >&2
        return 1
    fi
done

# ===== Resume settings =====
# Resume keeps using POLICY_ID's existing output directory and extends the
# saved training target without changing the fixed post-warmup learning rate.
export RESUME_STEPS="${RESUME_STEPS:-10000}"
export RESUME_SAVE_FREQ="${RESUME_SAVE_FREQ:-2000}"

# ===== Rollout common settings =====
export N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
export NUM_INFERENCE_STEP="${NUM_INFERENCE_STEP:-10}"
export ROLLOUT_FPS="${ROLLOUT_FPS:-10}"
export ROLLOUT_DURATION="${ROLLOUT_DURATION:-360}"
export ROLLOUT_CANFD_FOLLOW="${ROLLOUT_CANFD_FOLLOW:-false}"
export ROLLOUT_STRATEGY_TYPE="${ROLLOUT_STRATEGY_TYPE:-highlight}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$STORAGE_ROOT/runs/rollout_dataset/rollout_${POLICY_ID}}"
export ROLLOUT_INFERENCE_TYPE="${ROLLOUT_INFERENCE_TYPE:-rtc}"
export RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-2}"
export RTC_QUEUE_THRESHOLD="${RTC_QUEUE_THRESHOLD:-10}"
export RTC_MAX_GUIDANCE_WEIGHT="${RTC_MAX_GUIDANCE_WEIGHT:-10}"
# EXP changes direction more quickly; LINEAR is more stable around grasp closure.
# export RTC_PREFIX_ATTENTION_SCHEDULE="EXP"
export RTC_PREFIX_ATTENTION_SCHEDULE="${RTC_PREFIX_ATTENTION_SCHEDULE:-LINEAR}"
export EVAL_NUM_STEPS="${EVAL_NUM_STEPS:-4}"

# ===== W&B =====
export WANDB_ENABLE="${WANDB_ENABLE:-true}"

# ===== CPU Restriction =====
# CORE_LIST=0-15
# for pid in $(pgrep -u $USER -f "lerobot_train|accelerate"); do
#     taskset -cp $CORE_LIST $pid
#     renice +10 -p $pid
# done
