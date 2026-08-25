#!/usr/bin/env bash

set -euo pipefail

# Diagnostic: determine whether Action Flow V2.3 can memorize FAST targets from
# one episode. This is intentionally separate from the normal training script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# env.sh appends to PYTHONPATH, which may be unset in a fresh shell while
# nounset checking is active.
export PYTHONPATH="${PYTHONPATH:-}"
source "${SCRIPT_DIR}/env.sh"

EPISODES="${OVERFIT_EPISODES:-[0]}"
CHUNK_SIZE="${OVERFIT_CHUNK_SIZE:-30}"
STEPS="${OVERFIT_STEPS:-5000}"
PER_DEVICE_BATCH_SIZE="${OVERFIT_BATCH_SIZE:-16}"
OPTIMIZER_LR="${OVERFIT_LR:-3e-5}"
WARMUP_STEPS="${OVERFIT_WARMUP_STEPS:-100}"
LOG_FREQ="${OVERFIT_LOG_FREQ:-50}"
GPU_IDS="${OVERFIT_GPU_IDS:-${GPU_IDS:-0}}"
NUM_GPUS="${OVERFIT_NUM_GPUS:-1}"
WANDB_ENABLE="${OVERFIT_WANDB_ENABLE:-false}"

AFV23_DATASET_ID="${AFV23_DATASET_ID:-rm65_single_table_preparation_and_bussing_afv2_2_planning_boxes}"
AFV23_DATASET_ROOT="${AFV23_DATASET_ROOT:-${STORAGE_ROOT}/runs/dataset/${AFV23_DATASET_ID}}"
AFV23_TOKENIZER_ARTIFACT="${OVERFIT_TOKENIZER_ARTIFACT:-${AFV23_TOKENIZER_ARTIFACT:-${PROJECT_ROOT}/artifacts/tokenizers/action_flow_v2_3_smolvlm_fast}}"
AFV23_VLM_MODEL_NAME="${OVERFIT_VLM_MODEL_NAME:-${AFV23_VLM_MODEL_NAME:-HuggingFaceTB/SmolVLM2-500M-Video-Instruct}}"
AFV23_NUM_VLM_LAYERS="${OVERFIT_NUM_VLM_LAYERS:-${AFV23_NUM_VLM_LAYERS:-16}}"
RESTRICT_FAST_VOCABULARY="${OVERFIT_RESTRICT_FAST_VOCABULARY:-true}"
# Preserve an explicitly empty value from env.sh; 500M smolvla_base weights
# cannot initialize a differently sized VLM backbone.
SMOLVLA_PRETRAINED_PATH="${OVERFIT_SMOLVLA_PRETRAINED_PATH-${SMOLVLA_PRETRAINED_PATH-lerobot/smolvla_base}}"

EPISODE_TAG="$(printf '%s' "$EPISODES" | tr -cd '0-9,' | tr ',' '-')"
EPISODE_TAG="${EPISODE_TAG:-custom}"
RUN_SUFFIX="${OVERFIT_RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
TRAINING_ID="overfit_afv23_fast_episode_${EPISODE_TAG}_steps_${STEPS}_batch_${PER_DEVICE_BATCH_SIZE}_chunk_${CHUNK_SIZE}_${RUN_SUFFIX}"
OUTPUT_DIR="${STORAGE_ROOT}/runs/train/${TRAINING_ID}"
LOG_PATH="${STORAGE_ROOT}/logs/train_${TRAINING_ID}.log"

mkdir -p "${STORAGE_ROOT}/logs"

SMOLVLA_INIT_ARGS=()
if [ -n "$SMOLVLA_PRETRAINED_PATH" ]; then
    SMOLVLA_INIT_ARGS+=(--policy.smolvla_pretrained_path="$SMOLVLA_PRETRAINED_PATH")
fi

ACCELERATE_ARGS=(--num_processes="$NUM_GPUS" --mixed_precision=bf16)
if [ "$NUM_GPUS" -gt 1 ]; then
    ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
fi

echo "=========================================="
echo "AFV2.3 FAST overfit diagnostic"
echo "dataset: $AFV23_DATASET_ROOT"
echo "episodes: $EPISODES"
echo "chunk_size: $CHUNK_SIZE"
echo "steps: $STEPS"
echo "batch_size: $PER_DEVICE_BATCH_SIZE"
echo "optimizer_lr: $OPTIMIZER_LR (constant after $WARMUP_STEPS warmup steps)"
echo "restrict_fast_vocabulary: $RESTRICT_FAST_VOCABULARY"
echo "vlm_model_name: $AFV23_VLM_MODEL_NAME"
echo "num_vlm_layers: $AFV23_NUM_VLM_LAYERS"
echo "output_dir: $OUTPUT_DIR"
echo "log_path: $LOG_PATH"
echo "Only FAST loss is enabled, so the printed loss is FAST CE."
echo "=========================================="

CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch \
    "${ACCELERATE_ARGS[@]}" \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$AFV23_DATASET_ID" \
    --dataset.root="$AFV23_DATASET_ROOT" \
    --dataset.episodes="$EPISODES" \
    --dataset.return_uint8=true \
    --dataset.use_imagenet_stats=false \
    --policy.type=action_flow_v2_3 \
    --policy.vlm_model_name="$AFV23_VLM_MODEL_NAME" \
    --policy.num_vlm_layers="$AFV23_NUM_VLM_LAYERS" \
    --policy.tokenizer_artifact_path="$AFV23_TOKENIZER_ARTIFACT" \
    "${SMOLVLA_INIT_ARGS[@]}" \
    --policy.input_features=null \
    --policy.output_features=null \
    --policy.device=cuda \
    --policy.chunk_size="$CHUNK_SIZE" \
    --policy.n_action_steps=1 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.train_state_proj=true \
    --policy.load_vlm_weights=true \
    --policy.knowledge_insulation=true \
    --policy.detach_expert_prefix_kv=true \
    --policy.keyframe_sampling_fraction=0.0 \
    --policy.plan_loss_weight=0.0 \
    --policy.box_loss_weight=0.0 \
    --policy.fast_loss_weight=1.0 \
    --policy.flow_loss_weight=0.0 \
    --policy.restrict_fast_vocabulary="$RESTRICT_FAST_VOCABULARY" \
    --policy.optimizer_lr="$OPTIMIZER_LR" \
    --policy.scheduler_warmup_steps="$WARMUP_STEPS" \
    --policy.scheduler_decay_steps="$STEPS" \
    --policy.scheduler_decay_lr="$OPTIMIZER_LR" \
    --policy.push_to_hub=false \
    --batch_size="$PER_DEVICE_BATCH_SIZE" \
    --steps="$STEPS" \
    --eval_freq=0 \
    --save_freq="$STEPS" \
    --log_freq="$LOG_FREQ" \
    --output_dir="$OUTPUT_DIR" \
    --wandb.enable="$WANDB_ENABLE" \
    --wandb.project=lerobot_action_flow_v2_rm65 \
    --wandb.disable_artifact=true \
    --job_name="$TRAINING_ID" \
    2>&1 | tee "$LOG_PATH"

echo "Finished. Inspect the printed loss or: $LOG_PATH"
