# [SmolVLA](http://arxiv.org/abs/2506.01844) Training Log

*Last updated: July 5, 2026*

---

## Model Assets

Before training, download the base SmolVLA policy and the SmolVLM model used by the policy:

```bash
./scripts/model_download.sh lerobot/smolvla_base
./scripts/model_download.sh HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

The local paths used in the training command are:

- SmolVLA base policy: `./models/smolvla_base`
- SmolVLM model: `./models/SmolVLM2-500M-Video-Instruct`

---

## Training Objective 1: RM65 Screwdriver-to-Box Task

The first SmolVLA experiment uses the same task as the ACT experiments in [ACT Training Objective](./act.md#training-objectives): train the RM65 robotic arm with a two-finger gripper to grasp a screwdriver and place it into a white box.

The scene setup uses two cameras:

- Wrist camera: mounted on the robotic arm.
- Front camera: placed in front of the robot, with a closer viewpoint for the screwdriver and box.

![Scene](assets/act/scene.jpg)

Both cameras are Intel RealSense D435 cameras:

- Name: Intel RealSense D435
- Type: RealSense
- Default stream profile:
  - Stream type: Color
  - Format: `rgb8`
  - Width: 640
  - Height: 480
  - FPS: 30

The robotic arm is a [Realman RM65](https://github.com/RealManRobot/rm_models/tree/main/RM65), a six-axis robotic arm.

The purpose of this experiment is to test whether SmolVLA can improve the RM65 grasping task compared with the ACT baselines, while keeping the same dataset and action interface. Unlike ACT, SmolVLA uses a vision-language-action model with a flow-matching action head, so this experiment is treated as fine-tuning from `lerobot/smolvla_base` rather than training a policy from scratch.

---

## Training Logs

### Training Log 1

#### Dataset

This run reuses the dataset from [ACT Training Log 6](./act.md#training-log-6). That dataset was recorded with the closer front-camera viewpoint and contains the strongest ACT baseline setup so far.

Dataset ID:

```text
rm65_aloha_pick_screwdriver_to_fixed_box_closer_front
```

#### Training Plan

The initial SmolVLA run keeps the ACT Log 6 action horizon choice:

- `chunk_size=50`
- `n_action_steps=50`
- `batch_size=64`
- `steps=20000`

The policy starts from `./models/smolvla_base`, and the VLM backbone is loaded from `./models/SmolVLM2-500M-Video-Instruct`.

#### Training Script

```bash
export HF_HOME=./.cache/huggingface
export HF_LEROBOT_HOME=./.cache/lerobot
export HF_ENDPOINT=https://hf-mirror.com

DATASET_ID=rm65_aloha_pick_screwdriver_to_fixed_box_closer_front
TRAINING_ID=rm65_smolvla_pick_screwdriver_to_fixed_box_episodes_250_steps_20000_batch_64_chunk_50

lerobot-train \
    --dataset.repo_id=$DATASET_ID \
    --dataset.root=./runs/dataset/$DATASET_ID \
    --dataset.return_uint8=true \
    --policy.path=./models/smolvla_base \
    --policy.vlm_model_name=models/SmolVLM2-500M-Video-Instruct \
    --policy.input_features=null \
    --policy.output_features=null \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --batch_size=64 \
    --steps=20000 \
    --eval_freq=0 \
    --save_freq=5000 \
    --log_freq=100 \
    --wandb.enable=true \
    --wandb.project=lerobot_rm65_smolvla \
    --output_dir=./runs/train/$TRAINING_ID \
    --job_name=$TRAINING_ID \
    2>&1 | tee ./logs/train_$TRAINING_ID.log
```

#### Evaluation
