export HF_HOME=./.cache/huggingface
export HF_LEROBOT_HOME=./.cache/lerobot
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=./repos/lerobot/src:${PYTHONPATH}
export WANDB_API_KEY=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

GPU_IDS=2,3
NUM_GPUS=2
PER_DEVICE_BATCH_SIZE=32

DATASET_ID=rm65_bimanual_bottle_multitask
TRAINING_ID=smolvla_rm65_bimanual_bottle_multitask_episodes_160_steps_40000_batch_32x2_chunk_50

echo $TRAINING_ID
mkdir -p ./logs

CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --mixed_precision=no \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=$DATASET_ID \
    --dataset.root=./runs/dataset/$DATASET_ID \
    --dataset.return_uint8=true \
    --dataset.video_backend=pyav \
    --policy.path=./models/smolvla_base \
    --policy.vlm_model_name=models/SmolVLM2-500M-Video-Instruct \
    --policy.input_features=null \
    --policy.output_features=null \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.chunk_size=50 \
    --policy.n_action_steps=1 \
    --policy.num_steps=10 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.load_vlm_weights=true \
    --policy.scheduler_decay_steps=40000 \
    --batch_size=$PER_DEVICE_BATCH_SIZE \
    --steps=40000 \
    --eval_freq=0 \
    --save_freq=10000 \
    --log_freq=500 \
    --wandb.enable=true \
    --wandb.project=lerobot_smolvla_rm65 \
    --output_dir=./runs/train/$TRAINING_ID \
    --job_name=$TRAINING_ID \
    2>&1 | tee ./logs/train_$TRAINING_ID.log
