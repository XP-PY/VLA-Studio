#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

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
        --output_dir="./runs/train/$TRAINING_ID" \
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
    TRAIN_MIXED_PRECISION="${TRAIN_MIXED_PRECISION:-bf16}"
    AFV23_DATASET_ID="${AFV23_DATASET_ID:-rm65_single_table_preparation_and_bussing_afv2_2_planning_boxes}"
    AFV23_DATASET_ROOT="${AFV23_DATASET_ROOT:-${STORAGE_ROOT}/runs/dataset/${AFV23_DATASET_ID}}"
    AFV23_TOKENIZER_ARTIFACT="${AFV23_TOKENIZER_ARTIFACT:-${PROJECT_ROOT}/artifacts/tokenizers/action_flow_v2_3_smolvlm_fast}"
    SMOLVLA_PRETRAINED_PATH="${SMOLVLA_PRETRAINED_PATH:-}"
    AFV23_SMOLVLA_INIT_ARGS=()
    SMOLVLA_INIT_TAG="smolvla_uninit"
    if [ -n "$SMOLVLA_PRETRAINED_PATH" ]; then
        AFV23_SMOLVLA_INIT_ARGS+=(--policy.smolvla_pretrained_path="$SMOLVLA_PRETRAINED_PATH")
        SMOLVLA_INIT_TAG="smolvla_init"
    fi
    TRAINING_ID="rm65_action_flow_v23_independent_branches_planning_boxes_episodes_442_steps_${STEPS}_batch_${PER_DEVICE_BATCH_SIZE}_chunk_${CHUNK_SIZE}_${KEYFRAME_SAMPLING_FRACTION}"
    ACCELERATE_ARGS=(--num_processes=$NUM_GPUS --mixed_precision=$TRAIN_MIXED_PRECISION)
    if [ "$NUM_GPUS" -gt 1 ]; then
        ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
    fi

    echo "=========================================="
    echo "Start training: $TRAINING_ID"
    echo "dataset_root: $AFV23_DATASET_ROOT"
    echo "chunk_size: $CHUNK_SIZE"
    echo "loss weights: plan=$PLAN_LOSS_WEIGHT box=$BOX_LOSS_WEIGHT fast=$FAST_LOSS_WEIGHT flow=$FLOW_LOSS_WEIGHT"
    echo "keyframe_sampling_fraction: $KEYFRAME_SAMPLING_FRACTION"
    echo "tokenizer_artifact: $AFV23_TOKENIZER_ARTIFACT"
    echo "smolvla_pretrained_path: ${SMOLVLA_PRETRAINED_PATH:-disabled}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
      "${ACCELERATE_ARGS[@]}" \
      -m lerobot.scripts.lerobot_train \
      --dataset.repo_id="$AFV23_DATASET_ID" \
      --dataset.root="$AFV23_DATASET_ROOT" \
      --dataset.return_uint8=true \
      --dataset.use_imagenet_stats=false \
      --policy.type=action_flow_v2_3 \
      --policy.vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct" \
      --policy.tokenizer_artifact_path="$AFV23_TOKENIZER_ARTIFACT" \
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
      --policy.knowledge_insulation=true \
      --policy.detach_expert_prefix_kv=true \
      --policy.keyframe_sampling_fraction="$KEYFRAME_SAMPLING_FRACTION" \
      --policy.plan_loss_weight="$PLAN_LOSS_WEIGHT" \
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
else
    echo "Unsupported POLICY_TYPE: $POLICY_TYPE" >&2
    echo "Supported values: act, action_flow_v1, action_flow_v2, action_flow_v2_1, action_flow_v2_2, action_flow_v2_3, pi0, smolvla" >&2
    exit 2
fi
