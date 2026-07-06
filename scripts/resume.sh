export HF_HOME=./.cache/huggingface
export HF_LEROBOT_HOME=./.cache/lerobot
export HF_ENDPOINT=https://hf-mirror.com

DATASET_ID=rm65_aloha_pick_screwdriver_to_fixed_box_closer_front
TRAINING_ID=rm65_smolvla_pick_screwdriver_to_fixed_box_episodes_250_steps_20000_batch_64_chunk_50
RESUME_ID=rm65_smolvla_pick_screwdriver_to_fixed_box_episodes_250_steps_40000_batch_64_chunk_50

CKPT_CONFIG=./runs/train/$TRAINING_ID/checkpoints/last/pretrained_model/train_config.json

lerobot-train \
    --resume=true \
    --config_path=$CKPT_CONFIG \
    --dataset.repo_id=$DATASET_ID \
    --dataset.root=./runs/dataset/$DATASET_ID \
    --dataset.return_uint8=true \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --batch_size=64 \
    --steps=40000 \
    --eval_freq=0 \
    --save_freq=5000 \
    --log_freq=100 \
    --wandb.enable=true \
    --wandb.project=lerobot_rm65_smolvla \
    --output_dir=./runs/train/$RESUME_ID \
    --job_name=$RESUME_ID \
    2>&1 | tee -a ./logs/train_$RESUME_ID.log