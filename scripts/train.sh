export HF_HOME=./.cache/huggingface
export HF_LEROBOT_HOME=./.cache/lerobot
export HF_ENDPOINT=https://hf-mirror.com

DATASET_ID=rm65_bimanual_grasp_and_place

# ==== ACT ====

TRAINING_ID=rm65_act_bimanual_grasp_and_place_episodes_20_steps_20000_batch_16_chunk_50

lerobot-train \
    --dataset.repo_id $DATASET_ID \
    --dataset.root ./runs/dataset/$DATASET_ID \
    --policy.type act \
    --policy.device cuda \
    --policy.push_to_hub false \
    --policy.chunk_size 50 \
    --policy.n_action_steps 1 \
    --batch_size 16 \
    --steps 20000 \
    --eval_freq 0 \
    --save_freq 5000 \
    --log_freq 100 \
    --output_dir ./runs/train/$TRAINING_ID \
    --wandb.enable true \
    --wandb.project lerobot_rm65_act \
    --job_name $TRAINING_ID \
    2>&1 | tee ./logs/train_$TRAINING_ID.log


# ==== Action Flow V1 ====

TRAINING_ID=rm65_action_flow_v1_bimanual_grasp_and_place_episodes_20_steps_20000_batch_16_chunk_50

lerobot-train \
    --dataset.repo_id $DATASET_ID \
    --dataset.root ./runs/dataset/$DATASET_ID \
    --policy.type action_flow_v1 \
    --policy.device cuda \
    --policy.push_to_hub false \
    --policy.chunk_size 50 \
    --policy.n_action_steps 1 \
    --batch_size 16 \
    --steps 20000 \
    --eval_freq 0 \
    --save_freq 5000 \
    --log_freq 100 \
    --output_dir ./runs/train/$TRAINING_ID \
    --wandb.enable true \
    --wandb.project lerobot_rm65_act \
    --job_name $TRAINING_ID \
    2>&1 | tee ./logs/train_$TRAINING_ID.log

systemctl poweroff

# lerobot-train \
#     --dataset.repo_id=$DATASET_ID \
#     --dataset.root=./runs/dataset/$DATASET_ID \
#     --dataset.return_uint8=true \
#     --policy.path=./models/smolvla_base \
#     --policy.vlm_model_name=models/SmolVLM2-500M-Video-Instruct \
#     --policy.input_features=null \
#     --policy.output_features=null \
#     --policy.device=cuda \
#     --policy.push_to_hub=false \
#     --policy.chunk_size=50 \
#     --policy.n_action_steps=50 \
#     --batch_size=64 \
#     --steps=20000 \
#     --eval_freq=0 \
#     --save_freq=5000 \
#     --log_freq=100 \
#     --wandb.enable=true \
#     --wandb.project=lerobot_rm65_smolvla \
#     --output_dir=./runs/train/$TRAINING_ID \
#     --job_name=$TRAINING_ID \
#     2>&1 | tee ./logs/train_$TRAINING_ID.log