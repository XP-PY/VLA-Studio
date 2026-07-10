#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# ========================== ACT ==========================
for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    TRAINING_ID=rm65_act_bimanual_assembly_parts_to_center_episodes_${TASK_EPISODES}_steps_${STEPS}_batch_${BATCH_SIZE}_chunk_${CHUNK_SIZE}

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
        --batch_size=$BATCH_SIZE \
        --steps=$STEPS \
        --eval_freq=0 \
        --save_freq=$SAVE_FREQ \
        --log_freq=$LOG_FREQ \
        --output_dir="./runs/train/$TRAINING_ID" \
        --wandb.enable=$WANDB_ENABLE \
        --wandb.project="lerobot_act_rm65" \
        --job_name="$TRAINING_ID" \
        2>&1 | tee "./logs/train_$TRAINING_ID.log"

    echo "Finished training: $TRAINING_ID"
done

# ========================== Action Flow ==========================

for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    TRAINING_ID=rm65_action_flow_v1_bimanual_assembly_parts_to_center_episodes_${TASK_EPISODES}_steps_${STEPS}_batch_${BATCH_SIZE}_chunk_${CHUNK_SIZE}

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
        --batch_size=$BATCH_SIZE \
        --steps=$STEPS \
        --eval_freq=0 \
        --save_freq=$SAVE_FREQ \
        --log_freq=$LOG_FREQ \
        --output_dir="./runs/train/$TRAINING_ID" \
        --wandb.enable=$WANDB_ENABLE \
        --wandb.project="lerobot_action_flow_rm65" \
        --job_name=$TRAINING_ID \
        2>&1 | tee "./logs/train_$TRAINING_ID.log"

    echo "Finished training: $TRAINING_ID"
done

# ========================== Power Off ==========================

systemctl poweroff