export HF_HOME=/home/ubuntu/py/vla_studio/.cache/huggingface
export HF_LEROBOT_HOME=/home/ubuntu/py/vla_studio/.cache/lerobot

DATASET_ID=rm65_aloha_pick_screwdriver_to_fixed_box_closer_front
TRAINING_ID=rm65_act_pick_screwdriver_to_fixed_box_closer_front_episodes_200_steps_40000_batch_32_chunck_100

lerobot-train \
    --dataset.repo_id $DATASET_ID \
    --dataset.root /home/ubuntu/py/vla_studio/runs/dataset/$DATASET_ID \
    --policy.type act \
    --policy.device cuda \
    --policy.push_to_hub false \
    --policy.chunk_size 100 \
    --policy.n_action_steps 1 \
    --batch_size 32 \
    --steps 40000 \
    --eval_freq 0 \
    --save_freq 10000 \
    --log_freq 100 \
    --output_dir /home/ubuntu/py/vla_studio/runs/train/$TRAINING_ID \
    --wandb.enable true \
    --wandb.project rm65_lerobot \
    --job_name $TRAINING_ID \
    2>&1 | tee /home/ubuntu/py/vla_studio/logs/train_$TRAINING_ID.log