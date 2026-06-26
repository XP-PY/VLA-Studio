# ACT Training Log
---

## Training objectives
Using two camera, one is on the wrist of robotic arm and another one is in front of robotic arm, and a robotic arm with two-finger gripper to pick a screwdriver and put it in a white box:
![Scene](assets/act/scene.jpg)

Two camera are all Intel RealSense D435:
- Name: Intel RealSense D435
- Type: RealSense
- Default stream profile:
    - Stream_type: Color
    - Format: rgb8
    - Width: 640
    - Height: 480
    - Fps: 30

Robotic arm is [Realman RM65](https://github.com/RealManRobot/rm_models/tree/main/RM65) six-axis robotic arm.

---

## Training Logs
### Training Log 1
#### Dataset Recording
At the begining, the camera positioned in front of the robotic arm uses a relatively far viewing angle, allowing for a complete view of the robotic arm:
![Front camera view](assets/act/camera_front_0.png)

But I found even though I record more than 150 episodes and modify various hyperparameters under this setting, robotic arm always fails to pick screwdriver, e.g., gripper is not aligned with the screwdriver.

So I change the front camera view to see screwdriver and box closer and clearer:
![Changed front camera view](assets/act/camera_front_1.png)

Then I record datset with 50 episodes, and white box is fixed:
```bash
TASK_NAME=rm65_aloha_pick_screwdriver_to_fixed_box_close_front
TASK_PROMPT="pick screwdriver and put it in box"

cd /home/ubuntu/py/vla_studio/repos/lerobot
python examples/rm65/passive_record.py \
    --config-path configs/rm65_realsense.yaml \
    --repo-id local/$TASK_NAME \
    --root /home/ubuntu/py/vla_studio/runs/dataset/$TASK_NAME \
    --task "$TASK_PROMPT" \
    --fps 30 \
    --episode-time-s 30 \
    --num-episodes 50 \
    --teleop-active-io 1 \
    --record-button-io 2 \
    --record-button-mode toggle \
    --io-read-source aloha \
    --aloha-target-ip 192.168.1.100 \
    --log-path /home/ubuntu/py/vla_studio/logs/record_$TASK_NAME.log
```

#### Training script
Training ACT with 8 batch size, 20000 steps and 100 chunck size:
```bash
DATASET_ID=rm65_aloha_pick_screwdriver_to_fixed_box_close_front
TRAINING_ID=rm65_act_pick_screwdriver_to_fixed_box_close_front_episodes_50_steps_20000_batch_8_chunck_100

lerobot-train \
    --dataset.repo_id $DATASET_ID \
    --dataset.root /home/ubuntu/py/vla_studio/runs/dataset/$DATASET_ID \
    --policy.type act \
    --policy.device cuda \
    --policy.push_to_hub false \
    --policy.chunk_size 100 \
    --policy.n_action_steps 20 \
    --batch_size 8 \
    --steps 20000 \
    --eval_freq 0 \
    --save_freq 5000 \
    --log_freq 100 \
    --output_dir /home/ubuntu/py/vla_studio/runs/train/$TRAINING_ID \
    --wandb.enable true \
    --wandb.project rm65_lerobot \
    --job_name $TRAINING_ID \
    2>&1 | tee /home/ubuntu/py/vla_studio/logs/train_$TRAINING_ID.log
```

#### Evalution
- Success rate: 5 / 20

- Successfully pick screwdriver and put in box:
<video controls width="300">
  <source src="assets/act/pick_screwdriver_success_0.mp4" type="video/mp4">
</video>

- Failed to pick screwdriver:
<video controls width="300">
  <source src="assets/act/pick_screwdriver_fail_0.mp4" type="video/mp4">
</video>
    
It refelects:
    - Success rate of picking is very low, it may needs more data to train or modify hyperparameters. And most success pick are similar to the success videos mentioned earlier which may comes uneven distribution of collected data.
    - When screwdriver missed from gripper, robotic arm cann't reacted rapidly, it keep going for a while and then go back to try again. I think it may comes from ACT temporal ensemble method, previous predicted action influents the actual movement action at time t , e.g., ACT model at time t-10 though it should have picked screwdriver successfully at time t but it actually don't.

Questions:
    - This training set chunck size 100, under the setting fps equals 30, it means every time, ACT predicts more than two seconds of future moments, is it really necessary? Especially meeting picked object moves or failed to pick and reacted quickly.
    - How many dataset is enough?

### Training Log 2
#### Dataset Recording
Using the same dataset as [log 1](#training-log-1)

#### Training script
Using the same training setting as [log 1](#training-log-1) except setting chunck size equals 20 to test whether ACT need 100 chunck size (more than two seconds of future moments at 30 fps)

#### Evalution
- Success rate: 0 / 0