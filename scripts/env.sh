#!/usr/bin/env bash

# ===== Project paths =====
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ===== HuggingFace / LeRobot cache =====
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT_ROOT/.cache/lerobot"
export HF_ENDPOINT="https://hf-mirror.com"

# ===== Python path =====
export PYTHONPATH="$PROJECT_ROOT/repos/lerobot/src:${PYTHONPATH}"

# ===== Robot =====
export ROBOT_CONFIG_PATH="configs/rm65_left_right_wrist_front_top.yaml"

# ===== Dataset =====
export DATASET_ID="rm65_bimanual_assembly_parts_to_center"
export DATASET_ROOT="$PROJECT_ROOT/runs/dataset/$DATASET_ID"

export TASK_ID="rm65_bimanual_farthest_to_closest_assembly_parts_to_center"
export TASK_PROMPT="grasp the visible assembly parts from farthest to closest relative to the front camera, and place each pair in the center"
export TASK_ROOT="$PROJECT_ROOT/runs/dataset/$TASK_ID"
export TASK_EPISODES=240
export TASK_FPS=30

# ===== Training common settings =====
export POLICY_ID="rm65_act_bimanual_assembly_parts_to_center_episodes_240_steps_60000_batch_8_chunk_100"
export STEPS=60000
export BATCH_SIZE=8
export SAVE_FREQ=20000
export LOG_FREQ=100
export CHUNK_SIZES=(20 30 40)

# ===== Rollout common settings =====
export ROLLOUT_FPS=10
export ROLLOUT_DURATION=360
export ROLLOUT_CANFD_FOLLOW=false

# ===== W&B =====
export WANDB_PROJECT="lerobot_act_rm65"
export WANDB_ENABLE=true