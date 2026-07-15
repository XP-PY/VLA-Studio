#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# ========================== ACT ==========================
# for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
#     TRAINING_ID=rm65_act_bimanual_assembly_parts_to_center_episodes_${TASK_EPISODES}_steps_${STEPS}_batch_${PER_DEVICE_BATCH_SIZE}_chunk_${CHUNK_SIZE}

#     echo "=========================================="
#     echo "Start training: $TRAINING_ID"
#     echo "chunk_size: $CHUNK_SIZE"
#     echo "=========================================="

#     lerobot-train \
#         --dataset.repo_id="$TASK_ID" \
#         --dataset.root="$TASK_ROOT" \
#         --policy.type="act" \
#         --policy.device=cuda \
#         --policy.push_to_hub=false \
#         --policy.chunk_size=$CHUNK_SIZE \
#         --policy.n_action_steps=1 \
#         --batch_size=$PER_DEVICE_BATCH_SIZE \
#         --steps=$STEPS \
#         --eval_freq=0 \
#         --save_freq=$SAVE_FREQ \
#         --log_freq=$LOG_FREQ \
#         --output_dir="$STORAGE_ROOT/runs/train/$TRAINING_ID" \
#         --wandb.enable=$WANDB_ENABLE \
#         --wandb.project="lerobot_act_rm65" \
#         --job_name="$TRAINING_ID" \
#         2>&1 | tee "$STORAGE_ROOT/logs/train_$TRAINING_ID.log"

#     echo "Finished training: $TRAINING_ID"
# done

# ========================== Action Flow ==========================

# for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
#     TRAINING_ID=rm65_action_flow_v1_bimanual_assembly_parts_to_center_episodes_${TASK_EPISODES}_steps_${STEPS}_batch_${PER_DEVICE_BATCH_SIZE}_chunk_${CHUNK_SIZE}

#     echo "=========================================="
#     echo "Start training: $TRAINING_ID"
#     echo "chunk_size: $CHUNK_SIZE"
#     echo "=========================================="

#     lerobot-train \
#         --dataset.repo_id="$TASK_ID" \
#         --dataset.root="$TASK_ROOT" \
#         --policy.type="action_flow_v1" \
#         --policy.device=cuda \
#         --policy.push_to_hub=false \
#         --policy.chunk_size=$CHUNK_SIZE \
#         --policy.n_action_steps=1 \
#         --batch_size=$PER_DEVICE_BATCH_SIZE \
#         --steps=$STEPS \
#         --eval_freq=0 \
#         --save_freq=$SAVE_FREQ \
#         --log_freq=$LOG_FREQ \
#         --output_dir="./runs/train/$TRAINING_ID" \
#         --wandb.enable=$WANDB_ENABLE \
#         --wandb.project="lerobot_action_flow_rm65" \
#         --job_name=$TRAINING_ID \
#         2>&1 | tee "$STORAGE_ROOT/logs/train_$TRAINING_ID.log"

#     echo "Finished training: $TRAINING_ID"
# done

# ========================== Pi0 ==========================

for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    TRAINING_ID=rm65_pi0_${TASK_ID#rm65_}_episodes_${TASK_EPISODES}_steps_${STEPS}_batch_${PER_DEVICE_BATCH_SIZE}_chunk_${CHUNK_SIZE}
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
      --policy.n_action_steps=1 \
      --policy.num_inference_steps=10 \
      --policy.use_relative_actions=true \
      --policy.relative_exclude_joints='["gripper"]' \
      --policy.freeze_vision_encoder=true \
      --policy.train_expert_only=true \
      --policy.gradient_checkpointing=true \
      --policy.scheduler_decay_steps=$STEPS \
      --policy.push_to_hub=false \
      --batch_size=$PER_DEVICE_BATCH_SIZE \
      --steps=$STEPS \
      --eval_freq=0 \
      --save_freq=$SAVE_FREQ \
      --log_freq=$LOG_FREQ \
      --output_dir="${STORAGE_ROOT}/runs/train/$TRAINING_ID" \
      --wandb.enable=$WANDB_ENABLE \
      --wandb.project="${WANDB_PROJECT}" \
      --job_name=$TRAINING_ID \
      2>&1 | tee ${STORAGE_ROOT}/logs/train_$TRAINING_ID.log

    echo "Finished training: $TRAINING_ID"
done
