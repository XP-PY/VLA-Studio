#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "Experiment profile: $EXPERIMENT_NAME"
echo "Config: $EXPERIMENT_CONFIG"
echo "Policy: $POLICY_TYPE"
echo "Dataset: $TASK_ROOT"
echo "Steps: $STEPS | per-device batch: $PER_DEVICE_BATCH_SIZE | chunks: ${CHUNK_SIZES[*]}"

# ========================== ACT ==========================
if [ "$POLICY_TYPE" = "act" ]; then
for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    TRAINING_ID=rm65_act_${TASK_ID#rm65_}_episodes_${TASK_EPISODES}_steps_${STEPS}_batch_${PER_DEVICE_BATCH_SIZE}_chunk_${CHUNK_SIZE}

    echo "=========================================="
    echo "Start training: $TRAINING_ID"
    echo "chunk_size: $CHUNK_SIZE"
    echo "=========================================="

    lerobot-train \
        --dataset.repo_id="$TASK_ID" \
        --dataset.root="$TASK_ROOT" \
        --policy.type="act" \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --policy.chunk_size=$CHUNK_SIZE \
        --policy.n_action_steps=1 \
        --batch_size=$PER_DEVICE_BATCH_SIZE \
        --steps=$STEPS \
        --eval_freq=0 \
        --save_freq=$SAVE_FREQ \
        --log_freq=$LOG_FREQ \
        --output_dir="$STORAGE_ROOT/runs/train/$TRAINING_ID" \
        --wandb.enable=$WANDB_ENABLE \
        --wandb.project="lerobot_act_rm65" \
        --job_name="$TRAINING_ID" \
        2>&1 | tee "$STORAGE_ROOT/logs/train_$TRAINING_ID.log"

    echo "Finished training: $TRAINING_ID"
done

# ========================== Action Flow V1 ==========================

elif [ "$POLICY_TYPE" = "action_flow_v1" ]; then
for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    TRAINING_ID=rm65_action_flow_v1_${TASK_ID#rm65_}_episodes_${TASK_EPISODES}_steps_${STEPS}_batch_${PER_DEVICE_BATCH_SIZE}_chunk_${CHUNK_SIZE}

    echo "=========================================="
    echo "Start training: $TRAINING_ID"
    echo "chunk_size: $CHUNK_SIZE"
    echo "=========================================="

    lerobot-train \
        --dataset.repo_id="$TASK_ID" \
        --dataset.root="$TASK_ROOT" \
        --policy.type="action_flow_v1" \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --policy.chunk_size=$CHUNK_SIZE \
        --policy.n_action_steps=1 \
        --batch_size=$PER_DEVICE_BATCH_SIZE \
        --steps=$STEPS \
        --eval_freq=0 \
        --save_freq=$SAVE_FREQ \
        --log_freq=$LOG_FREQ \
        --output_dir="$STORAGE_ROOT/runs/train/$TRAINING_ID" \
        --wandb.enable=$WANDB_ENABLE \
        --wandb.project="lerobot_action_flow_rm65" \
        --wandb.disable_artifact=true \
        --job_name=$TRAINING_ID \
        2>&1 | tee "$STORAGE_ROOT/logs/train_$TRAINING_ID.log"

    echo "Finished training: $TRAINING_ID"
done

# ========================== Pi0 ==========================

elif [ "$POLICY_TYPE" = "pi0" ]; then
for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    TRAINING_ID=rm65_pi0_${TASK_ID#rm65_}_episodes_${TASK_EPISODES}_steps_${STEPS}_batch_${PER_DEVICE_BATCH_SIZE}_chunk_${CHUNK_SIZE}_freeze_gemma
    ACCELERATE_ARGS=(--num_processes=$NUM_GPUS --mixed_precision=bf16)
    if [ "$NUM_GPUS" -gt 1 ]; then
        ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
    fi

    echo "=========================================="
    echo "Start training: $TRAINING_ID"
    echo "chunk_size: $CHUNK_SIZE"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
      "${ACCELERATE_ARGS[@]}" \
      -m lerobot.scripts.lerobot_train \
      --dataset.repo_id="${TASK_ID}_pi" \
      --dataset.root="${TASK_ROOT}_pi" \
      --dataset.return_uint8=true \
      --policy.path="lerobot/pi0_base" \
      --policy.input_features=null \
      --policy.output_features=null \
      --policy.device=cuda \
      --policy.dtype=bfloat16 \
      --policy.chunk_size=$CHUNK_SIZE \
      --policy.use_relative_actions=true \
      --policy.relative_exclude_joints='["gripper"]' \
      --policy.freeze_vision_encoder=false \
      --policy.train_expert_only=false \
      --policy.gradient_checkpointing=false \
      --policy.scheduler_decay_steps=$STEPS \
      --policy.push_to_hub=false \
      --batch_size=$PER_DEVICE_BATCH_SIZE \
      --steps=$STEPS \
      --eval_freq=0 \
      --save_freq=$SAVE_FREQ \
      --log_freq=$LOG_FREQ \
      --output_dir="${STORAGE_ROOT}/runs/train/$TRAINING_ID" \
      --wandb.enable=$WANDB_ENABLE \
      --wandb.project="lerobot_pi0_rm65" \
      --wandb.disable_artifact=true \
      --job_name=$TRAINING_ID \
      2>&1 | tee ${STORAGE_ROOT}/logs/train_$TRAINING_ID.log

    echo "Finished training: $TRAINING_ID"
done


# ========================== SmolVLA ==========================

elif [ "$POLICY_TYPE" = "smolvla" ]; then
for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    OPTIMIZER_LR="${OPTIMIZER_LR:-1e-4}"
    TRAIN_MIXED_PRECISION="${TRAIN_MIXED_PRECISION:-no}"
    TRAINING_ID=rm65_smolvla_${TASK_ID#rm65_}_episodes_${TASK_EPISODES}_steps_${STEPS}_batch_${PER_DEVICE_BATCH_SIZE}_chunk_${CHUNK_SIZE}
    ACCELERATE_ARGS=(--num_processes=$NUM_GPUS --mixed_precision=$TRAIN_MIXED_PRECISION)
    if [ "$NUM_GPUS" -gt 1 ]; then
        ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
    fi

    echo "=========================================="
    echo "Start training: $TRAINING_ID"
    echo "chunk_size: $CHUNK_SIZE"
    echo "optimizer_lr: $OPTIMIZER_LR"
    echo "mixed_precision: $TRAIN_MIXED_PRECISION"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
      "${ACCELERATE_ARGS[@]}" \
      -m lerobot.scripts.lerobot_train \
      --dataset.repo_id="$TASK_ID" \
      --dataset.root="$TASK_ROOT" \
      --dataset.return_uint8=true \
      --policy.path="lerobot/smolvla_base" \
      --policy.vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct" \
      --policy.input_features=null \
      --policy.output_features=null \
      --policy.device=cuda \
      --policy.chunk_size=$CHUNK_SIZE \
      --policy.n_action_steps=1 \
      --policy.num_steps=10 \
      --policy.freeze_vision_encoder=false \
      --policy.train_expert_only=false \
      --policy.train_state_proj=true \
      --policy.load_vlm_weights=true \
      --policy.optimizer_lr=$OPTIMIZER_LR \
      --policy.scheduler_decay_steps=$STEPS \
      --policy.push_to_hub=false \
      --batch_size=$PER_DEVICE_BATCH_SIZE \
      --steps=$STEPS \
      --eval_freq=0 \
      --save_freq=$SAVE_FREQ \
      --log_freq=$LOG_FREQ \
      --output_dir="${STORAGE_ROOT}/runs/train/$TRAINING_ID" \
      --wandb.enable=$WANDB_ENABLE \
      --wandb.project="lerobot_smolvla_rm65" \
      --wandb.disable_artifact=true \
      --job_name="$TRAINING_ID" \
      2>&1 | tee "${STORAGE_ROOT}/logs/train_$TRAINING_ID.log"

    echo "Finished training: $TRAINING_ID"
done

# ========================== Action Flow V2 ==========================

elif [ "$POLICY_TYPE" = "action_flow_v2" ]; then
for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    RELEVANCE_LOSS_WEIGHT="${RELEVANCE_LOSS_WEIGHT:-0.1}"
    OPTIMIZER_LR="${OPTIMIZER_LR:-3e-5}"
    TRAIN_MIXED_PRECISION="${TRAIN_MIXED_PRECISION:-bf16}"
    AFV2_DATASET_ID="${TASK_ID}_afv2_segmentation"
    AFV2_DATASET_ROOT="${STORAGE_ROOT}/runs/dataset/${AFV2_DATASET_ID}"
    TRAINING_ID=rm65_action_flow_v2_${TASK_ID#rm65_}_episodes_${TASK_EPISODES}_steps_${STEPS}_batch_${PER_DEVICE_BATCH_SIZE}_chunk_${CHUNK_SIZE}_relevance_${RELEVANCE_LOSS_WEIGHT}
    ACCELERATE_ARGS=(--num_processes=$NUM_GPUS --mixed_precision=$TRAIN_MIXED_PRECISION)
    if [ "$NUM_GPUS" -gt 1 ]; then
        ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
    fi

    echo "=========================================="
    echo "Start training: $TRAINING_ID"
    echo "chunk_size: $CHUNK_SIZE"
    echo "relevance_loss_weight: $RELEVANCE_LOSS_WEIGHT"
    echo "optimizer_lr: $OPTIMIZER_LR"
    echo "mixed_precision: $TRAIN_MIXED_PRECISION"
    echo "dataset_root: $AFV2_DATASET_ROOT"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
      "${ACCELERATE_ARGS[@]}" \
      -m lerobot.scripts.lerobot_train \
      --dataset.repo_id="$AFV2_DATASET_ID" \
      --dataset.root="$AFV2_DATASET_ROOT" \
      --dataset.return_uint8=true \
      --policy.type=action_flow_v2 \
      --policy.pretrained_path="lerobot/smolvla_base" \
      --policy.vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct" \
      --policy.relevance_loss_weight="$RELEVANCE_LOSS_WEIGHT" \
      --policy.input_features=null \
      --policy.output_features=null \
      --policy.device=cuda \
      --policy.chunk_size="$CHUNK_SIZE" \
      --policy.n_action_steps=1 \
      --policy.freeze_vision_encoder=false \
      --policy.train_expert_only=false \
      --policy.train_state_proj=true \
      --policy.load_vlm_weights=true \
      --policy.optimizer_lr=$OPTIMIZER_LR \
      --policy.scheduler_decay_steps="$STEPS" \
      --policy.push_to_hub=false \
      --batch_size="$PER_DEVICE_BATCH_SIZE" \
      --steps="$STEPS" \
      --eval_freq=0 \
      --save_freq="$SAVE_FREQ" \
      --log_freq="$LOG_FREQ" \
      --output_dir="${STORAGE_ROOT}/runs/train/$TRAINING_ID" \
      --wandb.enable="$WANDB_ENABLE" \
      --wandb.project=lerobot_action_flow_v2_rm65 \
      --wandb.disable_artifact=true \
      --job_name="$TRAINING_ID" \
      2>&1 | tee "${STORAGE_ROOT}/logs/train_$TRAINING_ID.log"

    echo "Finished training: $TRAINING_ID"
done

# ========================== Action Flow V2.1 ==========================

elif [ "$POLICY_TYPE" = "action_flow_v2_1" ]; then
for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    RELEVANCE_LOSS_WEIGHT="${RELEVANCE_LOSS_WEIGHT:-0.1}"
    OPTIMIZER_LR="${OPTIMIZER_LR:-3e-5}"
    TRAIN_MIXED_PRECISION="${TRAIN_MIXED_PRECISION:-bf16}"
    TRAINING_ID=rm65_action_flow_v21_${TASK_ID#rm65_}_episodes_${TASK_EPISODES}_steps_${STEPS}_batch_${PER_DEVICE_BATCH_SIZE}_chunk_${CHUNK_SIZE}_relevance_${RELEVANCE_LOSS_WEIGHT}
    ACCELERATE_ARGS=(--num_processes=$NUM_GPUS --mixed_precision=$TRAIN_MIXED_PRECISION)
    if [ "$NUM_GPUS" -gt 1 ]; then
        ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
    fi

    echo "=========================================="
    echo "Start training: $TRAINING_ID"
    echo "chunk_size: $CHUNK_SIZE"
    echo "relevance_loss_weight: $RELEVANCE_LOSS_WEIGHT"
    echo "optimizer_lr: $OPTIMIZER_LR"
    echo "mixed_precision: $TRAIN_MIXED_PRECISION"
    echo "dataset_root: $TASK_ROOT"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
      "${ACCELERATE_ARGS[@]}" \
      -m lerobot.scripts.lerobot_train \
      --dataset.repo_id="$TASK_ID" \
      --dataset.root="$TASK_ROOT" \
      --dataset.return_uint8=true \
      --policy.type=action_flow_v2_1 \
      --policy.pretrained_path="lerobot/smolvla_base" \
      --policy.vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct" \
      --policy.relevance_loss_weight="$RELEVANCE_LOSS_WEIGHT" \
      --policy.input_features=null \
      --policy.output_features=null \
      --policy.device=cuda \
      --policy.chunk_size="$CHUNK_SIZE" \
      --policy.n_action_steps=1 \
      --policy.freeze_vision_encoder=false \
      --policy.train_expert_only=false \
      --policy.train_state_proj=true \
      --policy.load_vlm_weights=true \
      --policy.optimizer_lr=$OPTIMIZER_LR \
      --policy.relevance_bce_weight=1.0 \
      --policy.relevance_dice_weight=1.0 \
      --policy.scheduler_warmup_steps=2000 \
      --policy.scheduler_decay_steps="$STEPS" \
      --policy.scheduler_decay_lr=2.5e-6 \
      --policy.push_to_hub=false \
      --batch_size="$PER_DEVICE_BATCH_SIZE" \
      --steps="$STEPS" \
      --eval_freq=0 \
      --save_freq="$SAVE_FREQ" \
      --log_freq="$LOG_FREQ" \
      --output_dir="${STORAGE_ROOT}/runs/train/$TRAINING_ID" \
      --wandb.enable="$WANDB_ENABLE" \
      --wandb.project=lerobot_action_flow_v2_rm65 \
      --wandb.disable_artifact=true \
      --job_name="$TRAINING_ID" \
      2>&1 | tee "${STORAGE_ROOT}/logs/train_$TRAINING_ID.log"

    echo "Finished training: $TRAINING_ID"
done

# ========================== Action Flow V2.2 ==========================

elif [ "$POLICY_TYPE" = "action_flow_v2_2" ]; then
for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    BOX_LOSS_WEIGHT="${BOX_LOSS_WEIGHT:-1.0}"
    FAST_LOSS_WEIGHT="${FAST_LOSS_WEIGHT:-1.0}"
    FLOW_LOSS_WEIGHT="${FLOW_LOSS_WEIGHT:-1.0}"
    OPTIMIZER_LR="${OPTIMIZER_LR:-3e-5}"
    TRAIN_MIXED_PRECISION="${TRAIN_MIXED_PRECISION:-bf16}"
    AFV22_TOKENIZER_ARTIFACT="${AFV22_TOKENIZER_ARTIFACT:-${PROJECT_ROOT}/artifacts/tokenizers/action_flow_v2_2_smolvlm_fast}"
    SMOLVLA_PRETRAINED_PATH="${SMOLVLA_PRETRAINED_PATH:-}"
    AFV22_SMOLVLA_INIT_ARGS=()
    SMOLVLA_INIT_TAG="_smolvla_uninit"
    if [ -n "$SMOLVLA_PRETRAINED_PATH" ]; then
        AFV22_SMOLVLA_INIT_ARGS+=(--policy.smolvla_pretrained_path="$SMOLVLA_PRETRAINED_PATH")
        SMOLVLA_INIT_TAG="_smolvla_init"
    fi
    TRAINING_ID=rm65_action_flow_v22_${TASK_ID#rm65_}_episodes_${TASK_EPISODES}_steps_${STEPS}_batch_${PER_DEVICE_BATCH_SIZE}_chunk_${CHUNK_SIZE}_${SMOLVLA_INIT_TAG}_unresize
    ACCELERATE_ARGS=(--num_processes=$NUM_GPUS --mixed_precision=$TRAIN_MIXED_PRECISION)
    if [ "$NUM_GPUS" -gt 1 ]; then
        ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
    fi

    echo "=========================================="
    echo "Start training: $TRAINING_ID"
    echo "chunk_size: $CHUNK_SIZE"
    echo "box_loss_weight: $BOX_LOSS_WEIGHT"
    echo "fast_loss_weight: $FAST_LOSS_WEIGHT"
    echo "flow_loss_weight: $FLOW_LOSS_WEIGHT"
    echo "optimizer_lr: $OPTIMIZER_LR"
    echo "mixed_precision: $TRAIN_MIXED_PRECISION"
    echo "dataset_root: $TASK_ROOT"
    echo "tokenizer_artifact: $AFV22_TOKENIZER_ARTIFACT"
    echo "smolvla_pretrained_path: ${SMOLVLA_PRETRAINED_PATH:-disabled}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
      "${ACCELERATE_ARGS[@]}" \
      -m lerobot.scripts.lerobot_train \
      --dataset.repo_id="$TASK_ID" \
      --dataset.root="$TASK_ROOT" \
      --dataset.return_uint8=true \
      --policy.type=action_flow_v2_2 \
      --policy.vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct" \
      --policy.tokenizer_artifact_path="$AFV22_TOKENIZER_ARTIFACT" \
      "${AFV22_SMOLVLA_INIT_ARGS[@]}" \
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
      --policy.box_loss_weight="$BOX_LOSS_WEIGHT" \
      --policy.fast_loss_weight="$FAST_LOSS_WEIGHT" \
      --policy.flow_loss_weight="$FLOW_LOSS_WEIGHT" \
      --policy.optimizer_lr="$OPTIMIZER_LR" \
      --policy.scheduler_warmup_steps=2000 \
      --policy.scheduler_decay_steps="$STEPS" \
      --policy.scheduler_decay_lr=2.5e-6 \
      --policy.push_to_hub=false \
      --batch_size="$PER_DEVICE_BATCH_SIZE" \
      --steps="$STEPS" \
      --eval_freq=0 \
      --save_freq="$SAVE_FREQ" \
      --log_freq="$LOG_FREQ" \
      --output_dir="${STORAGE_ROOT}/runs/train/$TRAINING_ID" \
      --wandb.enable="$WANDB_ENABLE" \
      --wandb.project=lerobot_action_flow_v2_rm65 \
      --wandb.disable_artifact=true \
      --job_name="$TRAINING_ID" \
      2>&1 | tee "${STORAGE_ROOT}/logs/train_$TRAINING_ID.log"

    echo "Finished training: $TRAINING_ID"
done

# ========================== Action Flow V2.3 ==========================

elif [ "$POLICY_TYPE" = "action_flow_v2_3" ]; then
for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    PLAN_LOSS_WEIGHT="${PLAN_LOSS_WEIGHT:-1.0}"
    BOX_LOSS_WEIGHT="${BOX_LOSS_WEIGHT:-1.0}"
    FAST_LOSS_WEIGHT="${FAST_LOSS_WEIGHT:-1.0}"
    FLOW_LOSS_WEIGHT="${FLOW_LOSS_WEIGHT:-1.0}"
    KEYFRAME_SAMPLING_FRACTION="${KEYFRAME_SAMPLING_FRACTION:-0.05}"
    OPTIMIZER_LR="${OPTIMIZER_LR:-3e-5}"
    SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-2000}"
    SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-$OPTIMIZER_LR}"
    RESTRICT_FAST_VOCABULARY="${RESTRICT_FAST_VOCABULARY:-true}"
    AFV23_FAST_TOKEN_COUNT="${AFV23_FAST_TOKEN_COUNT:-2048}"
    TRAIN_MIXED_PRECISION="${TRAIN_MIXED_PRECISION:-bf16}"
    AFV23_DATASET_ID="${AFV23_DATASET_ID:-rm65_single_table_preparation_and_bussing_afv2_2_planning_boxes}"
    AFV23_DATASET_ROOT="${AFV23_DATASET_ROOT:-${STORAGE_ROOT}/runs/dataset/${AFV23_DATASET_ID}}"
    AFV23_TOKENIZER_ARTIFACT="${AFV23_TOKENIZER_ARTIFACT:-${PROJECT_ROOT}/artifacts/tokenizers/action_flow_v2_3_smolvlm_fast}"
    AFV23_VLM_MODEL_NAME="${AFV23_VLM_MODEL_NAME:-HuggingFaceTB/SmolVLM2-500M-Video-Instruct}"
    AFV23_NUM_VLM_LAYERS="${AFV23_NUM_VLM_LAYERS:-16}"
    AFV23_RUN_SUFFIX="${AFV23_RUN_SUFFIX:-}"
    AFV23_KNOWLEDGE_INSULATION="${AFV23_KNOWLEDGE_INSULATION:-true}"
    AFV23_DETACH_EXPERT_PREFIX_KV="${AFV23_DETACH_EXPERT_PREFIX_KV:-$AFV23_KNOWLEDGE_INSULATION}"
    if [ "$AFV23_KNOWLEDGE_INSULATION" = "true" ]; then
        KI_TAG=""
    elif [ "$AFV23_KNOWLEDGE_INSULATION" = "false" ]; then
        KI_TAG="_no_ki"
    else
        echo "AFV23_KNOWLEDGE_INSULATION must be true or false." >&2
        exit 1
    fi
    if [ "$AFV23_DETACH_EXPERT_PREFIX_KV" != "true" ] && [ "$AFV23_DETACH_EXPERT_PREFIX_KV" != "false" ]; then
        echo "AFV23_DETACH_EXPERT_PREFIX_KV must be true or false." >&2
        exit 1
    fi
    SMOLVLA_PRETRAINED_PATH="${SMOLVLA_PRETRAINED_PATH:-}"
    AFV23_SMOLVLA_INIT_ARGS=()
    SMOLVLA_INIT_TAG="smolvla_uninit"
    if [ -n "$SMOLVLA_PRETRAINED_PATH" ]; then
        AFV23_SMOLVLA_INIT_ARGS+=(--policy.smolvla_pretrained_path="$SMOLVLA_PRETRAINED_PATH")
        SMOLVLA_INIT_TAG="smolvla_init"
    fi
    TRAINING_ID="rm65_action_flow_v23_plan_conditioned_segment_aware_episodes_${TASK_EPISODES}_steps_${STEPS}_batch_${PER_DEVICE_BATCH_SIZE}_chunk_${CHUNK_SIZE}_${KEYFRAME_SAMPLING_FRACTION}${AFV23_RUN_SUFFIX:+_${AFV23_RUN_SUFFIX}}${KI_TAG}"
    ACCELERATE_ARGS=(--num_processes=$NUM_GPUS --mixed_precision=$TRAIN_MIXED_PRECISION)
    if [ "$NUM_GPUS" -gt 1 ]; then
        ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
    fi

    echo "=========================================="
    echo "Start training: $TRAINING_ID"
    echo "dataset_root: $AFV23_DATASET_ROOT"
    echo "chunk_size: $CHUNK_SIZE"
    echo "loss weights: plan=$PLAN_LOSS_WEIGHT box=$BOX_LOSS_WEIGHT fast=$FAST_LOSS_WEIGHT flow=$FLOW_LOSS_WEIGHT"
    echo "restrict_fast_vocabulary: $RESTRICT_FAST_VOCABULARY"
    echo "fast_token_count: $AFV23_FAST_TOKEN_COUNT"
    echo "keyframe_sampling_fraction: $KEYFRAME_SAMPLING_FRACTION"
    echo "learning rate: warmup_steps=$SCHEDULER_WARMUP_STEPS peak=$OPTIMIZER_LR constant=$SCHEDULER_DECAY_LR"
    echo "vlm_model_name: $AFV23_VLM_MODEL_NAME"
    echo "num_vlm_layers: $AFV23_NUM_VLM_LAYERS"
    echo "tokenizer_artifact: $AFV23_TOKENIZER_ARTIFACT"
    echo "smolvla_pretrained_path: ${SMOLVLA_PRETRAINED_PATH:-disabled}"
    echo "knowledge_insulation: $AFV23_KNOWLEDGE_INSULATION"
    echo "detach_expert_prefix_kv: $AFV23_DETACH_EXPERT_PREFIX_KV"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
      "${ACCELERATE_ARGS[@]}" \
      -m lerobot.scripts.lerobot_train \
      --dataset.repo_id="$AFV23_DATASET_ID" \
      --dataset.root="$AFV23_DATASET_ROOT" \
      --dataset.return_uint8=true \
      --dataset.use_imagenet_stats=false \
      --policy.type=action_flow_v2_3 \
      --policy.vlm_model_name="$AFV23_VLM_MODEL_NAME" \
      --policy.num_vlm_layers="$AFV23_NUM_VLM_LAYERS" \
      --policy.tokenizer_artifact_path="$AFV23_TOKENIZER_ARTIFACT" \
      --policy.fast_token_count="$AFV23_FAST_TOKEN_COUNT" \
      "${AFV23_SMOLVLA_INIT_ARGS[@]}" \
      --policy.input_features=null \
      --policy.output_features=null \
      --policy.device=cuda \
      --policy.chunk_size="$CHUNK_SIZE" \
      --policy.n_action_steps=1 \
      --policy.freeze_vision_encoder=false \
      --policy.train_expert_only=false \
      --policy.train_state_proj=true \
      --policy.load_vlm_weights=true \
      --policy.knowledge_insulation="$AFV23_KNOWLEDGE_INSULATION" \
      --policy.detach_expert_prefix_kv="$AFV23_DETACH_EXPERT_PREFIX_KV" \
      --policy.segment_aware_action_loss=true \
      --policy.keyframe_sampling_fraction="$KEYFRAME_SAMPLING_FRACTION" \
      --policy.plan_loss_weight="$PLAN_LOSS_WEIGHT" \
      --policy.box_loss_weight="$BOX_LOSS_WEIGHT" \
      --policy.fast_loss_weight="$FAST_LOSS_WEIGHT" \
      --policy.flow_loss_weight="$FLOW_LOSS_WEIGHT" \
      --policy.restrict_fast_vocabulary="$RESTRICT_FAST_VOCABULARY" \
      --policy.optimizer_lr="$OPTIMIZER_LR" \
      --policy.scheduler_warmup_steps="$SCHEDULER_WARMUP_STEPS" \
      --policy.scheduler_decay_steps="$STEPS" \
      --policy.scheduler_decay_lr="$SCHEDULER_DECAY_LR" \
      --policy.push_to_hub=false \
      --batch_size="$PER_DEVICE_BATCH_SIZE" \
      --steps="$STEPS" \
      --eval_freq=0 \
      --save_freq="$SAVE_FREQ" \
      --log_freq="$LOG_FREQ" \
      --output_dir="${STORAGE_ROOT}/runs/train/$TRAINING_ID" \
      --wandb.enable="$WANDB_ENABLE" \
      --wandb.project=lerobot_action_flow_v2_rm65 \
      --wandb.disable_artifact=true \
      --job_name="$TRAINING_ID" \
      2>&1 | tee "${STORAGE_ROOT}/logs/train_$TRAINING_ID.log"

    echo "Finished training: $TRAINING_ID"
done

# ========================== Action Flow V2.4 ==========================

elif [ "$POLICY_TYPE" = "action_flow_v2_4" ]; then
for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    AFV24_DATASET_ID="${AFV24_DATASET_ID:-rm65_single_table_preparation_and_bussing_afv2_2_planning_boxes}"
    AFV24_DATASET_ROOT="${AFV24_DATASET_ROOT:-${STORAGE_ROOT}/runs/dataset/${AFV24_DATASET_ID}}"
    PI05_PRETRAINED_PATH="${PI05_PRETRAINED_PATH:-lerobot/pi05_base}"
    AFV24_TARGET_TOKENIZER="${AFV24_TARGET_TOKENIZER:-google/paligemma-3b-pt-224}"
    AFV24_FAST_TOKENIZER="${AFV24_FAST_TOKENIZER:-artifacts/tokenizers/rm65_fast_h30}"
    AFV24_PLAN_LOSS_WEIGHT="${AFV24_PLAN_LOSS_WEIGHT:-1.0}"
    AFV24_BOX_LOSS_WEIGHT="${AFV24_BOX_LOSS_WEIGHT:-1.0}"
    AFV24_FAST_LOSS_WEIGHT="${AFV24_FAST_LOSS_WEIGHT:-1.0}"
    AFV24_FLOW_LOSS_WEIGHT="${AFV24_FLOW_LOSS_WEIGHT:-1.0}"
    AFV24_RESTRICT_FAST_VOCABULARY="${AFV24_RESTRICT_FAST_VOCABULARY:-true}"
    AFV24_KEYFRAME_SAMPLING_FRACTION="${AFV24_KEYFRAME_SAMPLING_FRACTION:-0.05}"
    AFV24_OPTIMIZER_LR="${AFV24_OPTIMIZER_LR:-2.5e-5}"
    AFV24_SCHEDULER_WARMUP_STEPS="${AFV24_SCHEDULER_WARMUP_STEPS:-1000}"
    AFV24_SCHEDULER_DECAY_LR="${AFV24_SCHEDULER_DECAY_LR:-2.5e-6}"
    AFV24_GRADIENT_CHECKPOINTING="${AFV24_GRADIENT_CHECKPOINTING:-true}"
    TRAIN_MIXED_PRECISION="${TRAIN_MIXED_PRECISION:-bf16}"
    GLOBAL_BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * NUM_GPUS))
    TRAINING_ID="rm65_action_flow_v24_pi05_plan_boxes_fast_episodes_${TASK_EPISODES}_steps_${STEPS}_global_batch_${GLOBAL_BATCH_SIZE}_chunk_${CHUNK_SIZE}"
    mkdir -p "${STORAGE_ROOT}/logs" "${STORAGE_ROOT}/runs/train"
    ACCELERATE_ARGS=(--num_processes="$NUM_GPUS" --mixed_precision="$TRAIN_MIXED_PRECISION")
    if [ "$NUM_GPUS" -gt 1 ]; then
        ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
    fi

    echo "=========================================="
    echo "Start training: $TRAINING_ID"
    echo "dataset_root: $AFV24_DATASET_ROOT"
    echo "pi05_pretrained_path: $PI05_PRETRAINED_PATH"
    echo "target_tokenizer: $AFV24_TARGET_TOKENIZER"
    echo "fast_tokenizer: $AFV24_FAST_TOKENIZER"
    echo "chunk_size: $CHUNK_SIZE"
    echo "batch_size: per_device=$PER_DEVICE_BATCH_SIZE global=$GLOBAL_BATCH_SIZE"
    echo "loss weights: plan=$AFV24_PLAN_LOSS_WEIGHT box=$AFV24_BOX_LOSS_WEIGHT fast=$AFV24_FAST_LOSS_WEIGHT flow=$AFV24_FLOW_LOSS_WEIGHT"
    echo "restrict_fast_vocabulary: $AFV24_RESTRICT_FAST_VOCABULARY"
    echo "keyframe_sampling_fraction: $AFV24_KEYFRAME_SAMPLING_FRACTION"
    echo "learning rate: warmup_steps=$AFV24_SCHEDULER_WARMUP_STEPS peak=$AFV24_OPTIMIZER_LR decay=$AFV24_SCHEDULER_DECAY_LR"
    echo "gradient_checkpointing: $AFV24_GRADIENT_CHECKPOINTING"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch \
      "${ACCELERATE_ARGS[@]}" \
      -m lerobot.scripts.lerobot_train \
      --dataset.repo_id="$AFV24_DATASET_ID" \
      --dataset.root="$AFV24_DATASET_ROOT" \
      --dataset.return_uint8=true \
      --dataset.use_imagenet_stats=false \
      --policy.type=action_flow_v2_4 \
      --policy.pi05_pretrained_path="$PI05_PRETRAINED_PATH" \
      --policy.target_tokenizer_name="$AFV24_TARGET_TOKENIZER" \
      --policy.fast_tokenizer_name="$AFV24_FAST_TOKENIZER" \
      --policy.input_features=null \
      --policy.output_features=null \
      --policy.device=cuda \
      --policy.dtype=bfloat16 \
      --policy.chunk_size="$CHUNK_SIZE" \
      --policy.n_action_steps=1 \
      --policy.use_relative_actions=true \
      --policy.relative_exclude_joints='["gripper"]' \
      --policy.freeze_vision_encoder=false \
      --policy.train_expert_only=false \
      --policy.gradient_checkpointing="$AFV24_GRADIENT_CHECKPOINTING" \
      --policy.compile_model=false \
      --policy.segment_aware_action_loss=true \
      --policy.keyframe_sampling_fraction="$AFV24_KEYFRAME_SAMPLING_FRACTION" \
      --policy.plan_loss_weight="$AFV24_PLAN_LOSS_WEIGHT" \
      --policy.box_loss_weight="$AFV24_BOX_LOSS_WEIGHT" \
      --policy.fast_loss_weight="$AFV24_FAST_LOSS_WEIGHT" \
      --policy.flow_loss_weight="$AFV24_FLOW_LOSS_WEIGHT" \
      --policy.restrict_fast_vocabulary="$AFV24_RESTRICT_FAST_VOCABULARY" \
      --policy.optimizer_lr="$AFV24_OPTIMIZER_LR" \
      --policy.scheduler_warmup_steps="$AFV24_SCHEDULER_WARMUP_STEPS" \
      --policy.scheduler_decay_steps="$STEPS" \
      --policy.scheduler_decay_lr="$AFV24_SCHEDULER_DECAY_LR" \
      --policy.push_to_hub=false \
      --batch_size="$PER_DEVICE_BATCH_SIZE" \
      --steps="$STEPS" \
      --eval_freq=0 \
      --save_freq="$SAVE_FREQ" \
      --log_freq="$LOG_FREQ" \
      --output_dir="${STORAGE_ROOT}/runs/train/$TRAINING_ID" \
      --wandb.enable="$WANDB_ENABLE" \
      --wandb.project=lerobot_action_flow_v2_rm65 \
      --wandb.disable_artifact=true \
      --job_name="$TRAINING_ID" \
      2>&1 | tee "${STORAGE_ROOT}/logs/train_$TRAINING_ID.log"

    echo "Finished training: $TRAINING_ID"
done
else
    echo "Unsupported POLICY_TYPE: $POLICY_TYPE" >&2
    echo "Supported values: act, action_flow_v1, action_flow_v2, action_flow_v2_1, action_flow_v2_2, action_flow_v2_3, action_flow_v2_4, pi0, smolvla" >&2
    exit 2
fi
