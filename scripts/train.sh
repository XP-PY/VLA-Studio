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
      --policy.path="${STORAGE_ROOT}/models/pi0_base" \
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
    TRAINING_ID=rm65_smolvla_${TASK_ID#rm65_}_episodes_${TASK_EPISODES}_steps_${STEPS}_batch_${PER_DEVICE_BATCH_SIZE}_chunk_${CHUNK_SIZE}
    ACCELERATE_ARGS=(--num_processes=$NUM_GPUS --mixed_precision=no)
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
      --dataset.repo_id="$TASK_ID" \
      --dataset.root="$TASK_ROOT" \
      --dataset.return_uint8=true \
      --policy.path="${STORAGE_ROOT}/models/smolvla_base" \
      --policy.vlm_model_name="${STORAGE_ROOT}/models/SmolVLM2-500M-Video-Instruct" \
      --policy.input_features=null \
      --policy.output_features=null \
      --policy.device=cuda \
      --policy.chunk_size=$CHUNK_SIZE \
      --policy.n_action_steps=1 \
      --policy.num_steps=10 \
      --policy.freeze_vision_encoder=true \
      --policy.train_expert_only=true \
      --policy.train_state_proj=true \
      --policy.load_vlm_weights=true \
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
else
    echo "Unsupported POLICY_TYPE: $POLICY_TYPE" >&2
    echo "Supported values: act, action_flow_v1, pi0, smolvla" >&2
    exit 2
fi
