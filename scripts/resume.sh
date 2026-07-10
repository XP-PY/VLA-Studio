export HF_HOME=./.cache/huggingface
export HF_LEROBOT_HOME=./.cache/lerobot
export HF_ENDPOINT=https://hf-mirror.com

DATASET_ID=rm65_bimanual_farthest_to_closest_assembly_parts_to_center
TRAINING_ID=rm65_action_flow_v1_bimanual_assembly_parts_to_center_episodes_240_steps_60000_batch_8_chunk_100
RESUME_ID=rm65_action_flow_v1_bimanual_assembly_parts_to_center_episodes_240_steps_60000_batch_8_chunk_100_resume

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