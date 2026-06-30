export HF_HOME=./.cache/huggingface
export HF_LEROBOT_HOME=./.cache/lerobot

DATASET_ID=rm65_aloha_pick_screwdriver_to_fixed_box_closer_front
TRAINING_ID=rm65_act_pick_screwdriver_to_fixed_box_closer_front_episodes_250_steps_40000_batch_32_chunck_50

lerobot-train \
    --dataset.repo_id $DATASET_ID \
    --dataset.root ./runs/dataset/$DATASET_ID \
    --policy.type act \
    --policy.device cuda \
    --policy.push_to_hub false \
    --policy.chunk_size 50 \
    --policy.n_action_steps 1 \
    --batch_size 32 \
    --steps 40000 \
    --eval_freq 0 \
    --save_freq 10000 \
    --log_freq 100 \
    --output_dir ./runs/train/$TRAINING_ID \
    --wandb.enable true \
    --wandb.project rm65_lerobot \
    --job_name $TRAINING_ID \
    2>&1 | tee ./logs/train_$TRAINING_ID.log