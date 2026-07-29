#!/usr/bin/env bash

# ===== Project paths =====
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ===== Storage root =====
export STORAGE_ROOT="/media/ubuntu/BW_500G/vla_studio_storage"
if [ ! -d "$STORAGE_ROOT" ]; then
    STORAGE_ROOT="${PROJECT_ROOT}"
fi

# ===== HuggingFace / LeRobot cache =====
export TORCH_HOME="$STORAGE_ROOT/.cache/torch"
export HF_HOME="$STORAGE_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_LEROBOT_HOME="$STORAGE_ROOT/.cache/lerobot"
export HF_ENDPOINT="https://hf-mirror.com"
# export HF_ENDPOINT="https://huggingface.co"

export YOLOE_CACHE_DIR="$STORAGE_ROOT/.cache/yoloe"
export YOLOE_WEIGHT_DIR="$YOLOE_CACHE_DIR/weights"
export YOLOE_DATASET_DIR="$YOLOE_CACHE_DIR/datasets"
export YOLOE_RUNS_DIR="$YOLOE_CACHE_DIR/runs"
# yolo settings weights_dir="${YOLOE_WEIGHT_DIR}"
# yolo settings datasets_dir="${YOLOE_DATASET_DIR}"
# yolo settings runs_dir="${YOLOE_RUNS_DIR}"


# ===== Python path =====
export PYTHONPATH="$PROJECT_ROOT/repos/lerobot/src:${PYTHONPATH}"

# ===== Robot =====
export ROBOT_CONFIG_PATH="configs/rm65_wrist_top.yaml"
# export ROBOT_CONFIG_PATH="configs/rm65_left_right_wrist_top.yaml"
# export ROBOT_CONFIG_PATH="configs/rm65_left_right_wrist_front_top.yaml"

# ===== Dataset =====
export TASK_ID="rm65_single_table_preparation_and_bussing_afv2_yoloe_dense_segmentation"
export TASK_PROMPT=""
export TASK_ROOT="$STORAGE_ROOT/runs/dataset/$TASK_ID"
export TASK_SUBSET_ID=""
export TASK_EPISODES=203
export TASK_FPS=30

# ===== Training common settings =====
export POLICY_ID=""
export POLICY_TYPE="action_flow_v2_1"
export STEPS=40000
export PER_DEVICE_BATCH_SIZE=8
export SAVE_FREQ=10000
export LOG_FREQ=100
export CHUNK_SIZES=(50)
export GPU_IDS=0
export NUM_GPUS=1

# ===== Rollout common settings =====
export N_ACTION_STEPS=1
export NUM_INFERENCE_STEP=5
export ROLLOUT_FPS=20
export ROLLOUT_DURATION=360
export ROLLOUT_CANFD_FOLLOW=false
export ROLLOUT_STRATEGY_TYPE=highlight
export ROLLOUT_DIR="$STORAGE_ROOT/runs/rollout_dataset/rollout_${POLICY_ID}"
export ROLLOUT_INFERENCE_TYPE=rtc
export RTC_EXECUTION_HORIZON=15
export RTC_MAX_GUIDANCE_WEIGHT=15
# export RTC_PREFIX_ATTENTION_SCHEDULE="EXP"
export RTC_PREFIX_ATTENTION_SCHEDULE="LINEAR"
export RTC_QUEUE_THRESHOLD=25

# ===== W&B =====
export WANDB_ENABLE=true

# ===== CPU Restriction =====
# CORE_LIST=0-15
# for pid in $(pgrep -u $USER -f "lerobot_train|accelerate"); do
#     taskset -cp $CORE_LIST $pid
#     renice +10 -p $pid
# done
