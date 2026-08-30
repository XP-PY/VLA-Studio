#!/usr/bin/env bash

set -euo pipefail

# Intermediate generalization tests after the 16-episode FAST overfit diagnostic.
# Training and validation episode counts are balanced independently per prompt.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}"
source "${SCRIPT_DIR}/env.sh"

PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:-/home/ubuntu/miniconda3/envs/lerobot}/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_PREFIX:-/home/ubuntu/miniconda3/envs/lerobot}/bin/accelerate}"
DATASET_ID="${AFV23_DATASET_ID:-rm65_single_table_preparation_and_bussing_afv2_2_planning_boxes}"
DATASET_ROOT="${AFV23_DATASET_ROOT:-${STORAGE_ROOT}/runs/dataset/${DATASET_ID}}"
EXPERIMENT_ROOT="${STORAGE_ROOT}/runs/evaluation/action_flow_v2_3_fast_balanced"
SPLIT_SEED="${AFV23_BALANCED_SPLIT_SEED:-42}"
TRAIN_PER_TASK="${AFV23_BALANCED_TRAIN_PER_TASK:-1}"
VALIDATION_PER_TASK="${AFV23_BALANCED_VALIDATION_PER_TASK:-1}"
DEFAULT_SPLIT_FILE="${EXPERIMENT_ROOT}/split_seed${SPLIT_SEED}_train${TRAIN_PER_TASK}_val${VALIDATION_PER_TASK}.json"
SPLIT_FILE="${AFV23_BALANCED_SPLIT_FILE:-${DEFAULT_SPLIT_FILE}}"

STEPS="${AFV23_BALANCED_STEPS:-10000}"
BATCH_SIZE="${AFV23_BALANCED_BATCH_SIZE:-16}"
CHUNK_SIZE="${AFV23_BALANCED_CHUNK_SIZE:-30}"
LEARNING_RATE="${AFV23_BALANCED_LR:-6e-5}"
WARMUP_STEPS="${AFV23_BALANCED_WARMUP_STEPS:-100}"
SAVE_FREQ="${AFV23_BALANCED_SAVE_FREQ:-5000}"
FAST_TOKEN_COUNT="${AFV23_BALANCED_FAST_TOKEN_COUNT:-1024}"
TOKENIZER_ARTIFACT="${AFV23_BALANCED_TOKENIZER_ARTIFACT:-\
${PROJECT_ROOT}/artifacts/tokenizers/action_flow_v2_3_rm65_fast_h30_vocab1024}"
VLM_MODEL_NAME="${AFV23_BALANCED_VLM_MODEL_NAME:-HuggingFaceTB/SmolVLM2-500M-Video-Instruct}"
NUM_VLM_LAYERS="${AFV23_BALANCED_NUM_VLM_LAYERS:-16}"
SMOLVLA_PATH="${AFV23_BALANCED_SMOLVLA_PATH:-lerobot/smolvla_base}"
TRAIN_MIXED_PRECISION="${AFV23_BALANCED_MIXED_PRECISION:-bf16}"
GPU_IDS="${AFV23_BALANCED_GPU_IDS:-${GPU_IDS:-0}}"
NUM_GPUS="${AFV23_BALANCED_NUM_GPUS:-1}"
PLAN_LOSS_WEIGHT="${AFV23_BALANCED_PLAN_LOSS_WEIGHT:-0.0}"
BOX_LOSS_WEIGHT="${AFV23_BALANCED_BOX_LOSS_WEIGHT:-0.0}"
FAST_LOSS_WEIGHT="${AFV23_BALANCED_FAST_LOSS_WEIGHT:-1.0}"
FLOW_LOSS_WEIGHT="${AFV23_BALANCED_FLOW_LOSS_WEIGHT:-0.0}"
KEYFRAME_SAMPLING_FRACTION="${AFV23_BALANCED_KEYFRAME_SAMPLING_FRACTION:-0.0}"
RUN_SUFFIX="${AFV23_BALANCED_RUN_SUFFIX:-}"

mkdir -p "${EXPERIMENT_ROOT}" "${STORAGE_ROOT}/logs"
if [ ! -f "${SPLIT_FILE}" ]; then
    "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/action_flow_v2/prepare_afv23_fast_split.py" \
        --dataset-root "${DATASET_ROOT}" \
        --output "${SPLIT_FILE}" \
        --seed "${SPLIT_SEED}" \
        --train-per-task "${TRAIN_PER_TASK}" \
        --validation-per-task "${VALIDATION_PER_TASK}"
fi

TRAIN_EPISODES="$("${PYTHON_BIN}" -c \
    'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["train_episodes"]))' \
    "${SPLIT_FILE}")"
TRAIN_EPISODE_COUNT="$("${PYTHON_BIN}" -c \
    'import json,sys; print(len(json.load(open(sys.argv[1]))["train_episodes"]))' \
    "${SPLIT_FILE}")"
LR_TAG="$(printf '%s' "${LEARNING_RATE}" | tr -cd '[:alnum:]')"
RUN_ID="diagnostic_afv23_fast_balanced_${TRAIN_EPISODE_COUNT}_steps_${STEPS}"
RUN_ID+="_batch_${BATCH_SIZE}_chunk_${CHUNK_SIZE}_vocab${FAST_TOKEN_COUNT}_lr${LR_TAG}"
if [ -n "${RUN_SUFFIX}" ]; then
    RUN_ID+="_${RUN_SUFFIX}"
fi
OUTPUT_DIR="${STORAGE_ROOT}/runs/train/${RUN_ID}"
LOG_PATH="${STORAGE_ROOT}/logs/train_${RUN_ID}.log"

if [ -e "${OUTPUT_DIR}" ]; then
    echo "Output directory already exists: ${OUTPUT_DIR}" >&2
    exit 1
fi

ACCELERATE_ARGS=(--num_processes="${NUM_GPUS}" --mixed_precision="${TRAIN_MIXED_PRECISION}")
if [ "${NUM_GPUS}" -gt 1 ]; then
    ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
fi

echo "=========================================="
echo "AFV2.3 balanced FAST generalization test"
echo "split: ${SPLIT_FILE}"
echo "train episodes: ${TRAIN_EPISODE_COUNT} (${TRAIN_PER_TASK} per task prompt)"
echo "validation episodes: ${VALIDATION_PER_TASK} per task prompt"
echo "steps: ${STEPS}; batch: ${BATCH_SIZE}; chunk: ${CHUNK_SIZE}"
echo "FAST vocabulary: ${FAST_TOKEN_COUNT}; learning rate: ${LEARNING_RATE}"
echo "loss weights: plan=${PLAN_LOSS_WEIGHT} box=${BOX_LOSS_WEIGHT} fast=${FAST_LOSS_WEIGHT} flow=${FLOW_LOSS_WEIGHT}"
echo "keyframe sampling fraction: ${KEYFRAME_SAMPLING_FRACTION}"
echo "output: ${OUTPUT_DIR}"
echo "=========================================="

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch \
    "${ACCELERATE_ARGS[@]}" \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATASET_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.episodes="${TRAIN_EPISODES}" \
    --dataset.return_uint8=true \
    --dataset.use_imagenet_stats=false \
    --policy.type=action_flow_v2_3 \
    --policy.vlm_model_name="${VLM_MODEL_NAME}" \
    --policy.num_vlm_layers="${NUM_VLM_LAYERS}" \
    --policy.tokenizer_artifact_path="${TOKENIZER_ARTIFACT}" \
    --policy.fast_token_count="${FAST_TOKEN_COUNT}" \
    --policy.smolvla_pretrained_path="${SMOLVLA_PATH}" \
    --policy.input_features=null \
    --policy.output_features=null \
    --policy.device=cuda \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.n_action_steps=1 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.train_state_proj=true \
    --policy.load_vlm_weights=true \
    --policy.knowledge_insulation=true \
    --policy.detach_expert_prefix_kv=true \
    --policy.keyframe_sampling_fraction="${KEYFRAME_SAMPLING_FRACTION}" \
    --policy.plan_loss_weight="${PLAN_LOSS_WEIGHT}" \
    --policy.box_loss_weight="${BOX_LOSS_WEIGHT}" \
    --policy.fast_loss_weight="${FAST_LOSS_WEIGHT}" \
    --policy.flow_loss_weight="${FLOW_LOSS_WEIGHT}" \
    --policy.restrict_fast_vocabulary=true \
    --policy.optimizer_lr="${LEARNING_RATE}" \
    --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
    --policy.scheduler_decay_steps="${STEPS}" \
    --policy.scheduler_decay_lr="${LEARNING_RATE}" \
    --policy.push_to_hub=false \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --eval_freq=0 \
    --save_freq="${SAVE_FREQ}" \
    --log_freq=50 \
    --output_dir="${OUTPUT_DIR}" \
    --wandb.enable="${AFV23_BALANCED_WANDB_ENABLE:-${WANDB_ENABLE:-true}}" \
    --wandb.project=lerobot_action_flow_v2_rm65 \
    --wandb.disable_artifact=true \
    --job_name="${RUN_ID}" \
    2>&1 | tee "${LOG_PATH}"

if [ "${AFV23_BALANCED_SKIP_EVAL:-false}" != "true" ]; then
    for EVAL_STEP in "${SAVE_FREQ}" "${STEPS}"; do
        CHECKPOINT="${OUTPUT_DIR}/checkpoints/$(printf '%06d' "${EVAL_STEP}")/pretrained_model"
        [ -d "${CHECKPOINT}" ] || continue
        EVAL_OUTPUT="${EXPERIMENT_ROOT}/${RUN_ID}_step_$(printf '%06d' "${EVAL_STEP}").json"
        "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/action_flow_v2/evaluate_afv23_fast.py" \
            --policy-path "${CHECKPOINT}" \
            --dataset-root "${DATASET_ROOT}" \
            --repo-id "local/${DATASET_ID}" \
            --split-file "${SPLIT_FILE}" \
            --output "${EVAL_OUTPUT}" \
            --sample-stride 10 \
            --batch-size "${BATCH_SIZE}"
    done
fi

echo "Training and evaluation complete. Results: ${EXPERIMENT_ROOT}/${RUN_ID}_step_*.json"
