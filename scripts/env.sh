#!/usr/bin/env bash

unset http_proxy
unset https_proxy
unset no_proxy
unset NO_PROXY
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

# ===== Project paths =====
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ===== Storage root =====
# Override before sourcing this file on a cloud host, for example:
# export STORAGE_ROOT=/root/autodl-tmp/vla-studio
# source scripts/env.sh
export STORAGE_ROOT="${STORAGE_ROOT:-/media/ubuntu/BW_500G/vla_studio_storage}"
if [ ! -d "$STORAGE_ROOT" ]; then
    STORAGE_ROOT="${PROJECT_ROOT}"
fi

# ===== HuggingFace / LeRobot cache =====
export TORCH_HOME="$STORAGE_ROOT/.cache/torch"
export HF_HOME="$STORAGE_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_LEROBOT_HOME="$STORAGE_ROOT/.cache/lerobot"
# export HF_ENDPOINT="https://hf-mirror.com"
export HF_ENDPOINT="https://huggingface.co"

export YOLOE_CACHE_DIR="$STORAGE_ROOT/.cache/yoloe"
export YOLOE_WEIGHT_DIR="$YOLOE_CACHE_DIR/weights"
export YOLOE_DATASET_DIR="$YOLOE_CACHE_DIR/datasets"
export YOLOE_RUNS_DIR="$YOLOE_CACHE_DIR/runs"
# yolo settings weights_dir="${YOLOE_WEIGHT_DIR}"
# yolo settings datasets_dir="${YOLOE_DATASET_DIR}"
# yolo settings runs_dir="${YOLOE_RUNS_DIR}"


# ===== Python path =====
export PYTHONPATH="$PROJECT_ROOT/repos/lerobot/src:${PYTHONPATH:-}"

# ===== Robot =====
export ROBOT_CONFIG_PATH="configs/rm65_wrist_top.yaml"
# export ROBOT_CONFIG_PATH="configs/rm65_left_right_wrist_top.yaml"
# export ROBOT_CONFIG_PATH="configs/rm65_left_right_wrist_front_top.yaml"

# ===== Dataset =====
export TASK_ID="rm65_single_table_preparation_and_bussing_afv2_2_planning_boxes"
export TASK_PROMPT="Put all vegetables into the collection box."
export TASK_ROOT="$STORAGE_ROOT/runs/dataset/$TASK_ID"
export TASK_SUBSET_ID="bus_the_table"
export TASK_EPISODES=442
export TASK_FPS=30

# ===== Training common settings =====
export POLICY_ID="rm65_action_flow_v24_pi05_plan_boxes_fast_episodes_442_steps_80000_global_batch_8_chunk_30"
export POLICY_TYPE="action_flow_v2_4"
export STEPS=80000
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export SAVE_FREQ=20000
export LOG_FREQ=100
export CHUNK_SIZES=(30)
export GPU_IDS="${GPU_IDS:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"

# ===== Resume settings =====
# Resume keeps using POLICY_ID's existing output directory and extends the
# saved training target without changing the fixed post-warmup learning rate.
export RESUME_STEPS=100000
export RESUME_SAVE_FREQ=20000

# ===== Action Flow V2.3 settings =====
export PLAN_LOSS_WEIGHT=1.0
export BOX_LOSS_WEIGHT=1.0
export FAST_LOSS_WEIGHT=1.0
export FLOW_LOSS_WEIGHT=1.0
export KEYFRAME_SAMPLING_FRACTION=0.05
export OPTIMIZER_LR=3e-5
export SCHEDULER_WARMUP_STEPS=2000
# Keep the LR constant after warmup. Using the same peak and decay LR also
# makes extending a resumed run independent of a newly chosen total step count.
export SCHEDULER_DECAY_LR="$OPTIMIZER_LR"
export RESTRICT_FAST_VOCABULARY=true
export TRAIN_MIXED_PRECISION=bf16

# Controlled 500M experiment matching the FAST vocabulary A/B test.
export AFV23_VLM_MODEL_NAME="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
export AFV23_NUM_VLM_LAYERS=16
export AFV23_TOKENIZER_ARTIFACT="$PROJECT_ROOT/artifacts/tokenizers/action_flow_v2_3_rm65_fast_h30"
export AFV23_RUN_SUFFIX="smolvlm500m_smolvla_init_fast_vocab_restricted_constant_lr"
export SMOLVLA_PRETRAINED_PATH="lerobot/smolvla_base"

# ===== Action Flow V2.4 cloud-training settings =====
# Recommended starting profile: 2 x 80 GB GPUs with DDP. Each process keeps a
# full PI0.5 model replica, so this is not intended for the local 24 GB GPU.
export PI05_PRETRAINED_PATH="lerobot/pi05_base"
export AFV24_TARGET_TOKENIZER="google/paligemma-3b-pt-224"
export AFV24_FAST_TOKENIZER="artifacts/tokenizers/rm65_fast_h30"
export AFV24_PLAN_LOSS_WEIGHT=1.0
export AFV24_BOX_LOSS_WEIGHT=1.0
export AFV24_FAST_LOSS_WEIGHT=1.0
export AFV24_FLOW_LOSS_WEIGHT=1.0
export AFV24_RESTRICT_FAST_VOCABULARY=true
export AFV24_KEYFRAME_SAMPLING_FRACTION=0.05
export AFV24_OPTIMIZER_LR=2.5e-5
export AFV24_SCHEDULER_WARMUP_STEPS=1000
export AFV24_SCHEDULER_DECAY_LR=2.5e-6
export AFV24_GRADIENT_CHECKPOINTING=true

# ===== Rollout common settings =====
export N_ACTION_STEPS=1
export NUM_INFERENCE_STEP=10
export ROLLOUT_FPS=10
export ROLLOUT_DURATION=360
export ROLLOUT_CANFD_FOLLOW=false
export ROLLOUT_STRATEGY_TYPE=highlight
export ROLLOUT_DIR="$STORAGE_ROOT/runs/rollout_dataset/rollout_${POLICY_ID}"
export ROLLOUT_INFERENCE_TYPE=rtc
export RTC_EXECUTION_HORIZON=2
export RTC_QUEUE_THRESHOLD=10
export RTC_MAX_GUIDANCE_WEIGHT=10
# EXP changes direction more quickly; LINEAR is more stable around grasp closure.
# export RTC_PREFIX_ATTENTION_SCHEDULE="EXP"
export RTC_PREFIX_ATTENTION_SCHEDULE="LINEAR"
export EVAL_NUM_STEPS=4

# ===== W&B =====
export WANDB_ENABLE=true

# ===== CPU Restriction =====
# CORE_LIST=0-15
# for pid in $(pgrep -u $USER -f "lerobot_train|accelerate"); do
#     taskset -cp $CORE_LIST $pid
#     renice +10 -p $pid
# done
