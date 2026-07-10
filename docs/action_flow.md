# Action Flow Training Log

*Last updated: July 7, 2026*

---

## Action Flow V1

`action_flow_v1` is a first experimental LeRobot policy for combining an ACT-style robot control interface with a SmolVLA-style flow matching training objective.

### Design Starting Point

The starting point is the RM65 arm imitation-learning setup where the desired policy input/output contract is close to ACT:

- Input: current camera image observations and the current robot state.
- Output: a chunk of continuous robot actions, normally joint angles or action targets in the same format as the dataset `action` feature.
- Runtime behavior: return one action at a time through `select_action`, while internally predicting a chunk and consuming it from an action queue.

ACT is a good reference for this boundary because it already fits common robot datasets in LeRobot: it encodes images with a ResNet backbone, adds robot state conditioning, uses a transformer encoder/decoder, and predicts a fixed action chunk. This makes it easy to plug into `lerobot-train`, existing dataset metadata, normalization, and evaluation scripts.

The weak point of plain ACT for this experiment is the direct regression target. ACT predicts the action chunk directly and is usually trained with an L1 loss against the demonstration action. `action_flow_v1` keeps the same continuous action interface, but changes the learning target: it learns a vector field over action chunks. During training, the model sees an interpolated noisy action chunk and predicts the direction from the clean demonstration action toward Gaussian noise. During inference, it starts from Gaussian noise and integrates the learned velocity field back toward a clean action chunk.

This design borrows the core idea from SmolVLA flow matching while avoiding the full vision-language model stack. For RM65 experiments, this gives a practical middle ground:

- Keep ACT's simple image/state/action data contract.
- Keep continuous angle/action outputs, so no action tokenization is required.
- Use flow matching for action generation, which may model multimodal action chunks better than one-shot regression.
- Leave clear replacement points for future competition changes.

### Policy Structure

The implementation lives under:

```text
repos/lerobot/src/lerobot/policies/action_flow_v1/
├── configuration_action_flow_v1.py
├── modeling_action_flow_v1.py
├── processor_action_flow_v1.py
└── __init__.py
```

The policy is registered as:

```text
policy.type=action_flow_v1
```

It is also wired into `policies/factory.py`, so the normal LeRobot factory path
can create the config, model class, and pre/post processors.

#### Configuration

`ActionFlowV1Config` defines the model and training defaults:

- `chunk_size`: number of future action steps generated at once.
- `n_action_steps`: number of generated actions executed before sampling a new chunk.
- `vision_backbone`: torchvision ResNet backbone, default `resnet18`.
- `dim_model`, `n_heads`, `n_encoder_layers`, `n_decoder_layers`: transformer architecture parameters.
- `num_inference_steps`: Euler integration steps used to transform noise into actions during inference.
- `time_sampling_alpha`, `time_sampling_beta`: Beta distribution parameters for training timestep sampling.
- `optimizer_lr`, `optimizer_lr_backbone`: separate learning rates for the main policy and image backbone.
- `scheduler_warmup_steps`, `scheduler_decay_steps`, `scheduler_decay_lr`: the default cosine decay schedule.

The config requires at least one conditioning input: image, robot state, or environment state. For the intended RM65 setup, the normal input is two camera features plus `observation.state`; the output is the dataset `action` feature.

#### Pre/Post Processing

`processor_action_flow_v1.py` mirrors the ACT processor:

1. Rename observations using the configured rename map.
2. Add a batch dimension when needed.
3. Move tensors to the policy device.
4. Normalize input observations and training actions using dataset statistics.
5. After inference, unnormalize the predicted action and move it back to CPU.

This is important because the flow model is trained in normalized action space, not raw joint-angle space. At runtime, LeRobot's postprocessor converts the normalized sampled action back to the dataset/robot action scale.

#### Condition Encoder

The condition encoder builds one transformer encoder sequence from all available observations:

- A learned condition token is always added.
- If `observation.state` exists, it is projected by a linear layer into `dim_model`.
- If an environment state exists, it is projected in the same way.
- Each camera image is passed through the ResNet backbone.
- The final ResNet feature map is projected to `dim_model` with a `1x1` convolution.
- A 2D sinusoidal position embedding is added for image feature-map tokens.
- Learned 1D position embeddings are used for condition/state tokens.

The resulting token sequence is passed through the reused ACT transformer encoder.

#### Flow Action Decoder

The action side of the model operates on full action chunks:

```text
noisy action chunk: (batch, chunk_size, action_dim)
timestep:           (batch,)
condition tokens:   encoder output from images/state
```

The noisy action chunk is projected into the transformer dimension. The scalar flow timestep is converted to a sinusoidal embedding, passed through a small MLP, and added to every action token. A learned positional embedding identifies each step in the action chunk. The ACT decoder attends from action tokens to the encoded observation tokens, and the final linear head predicts velocity with the same shape as the action chunk.

### Training Objective

For each training batch:

```python
actions = batch["action"]
noise = torch.randn_like(actions)
t = sample_time(batch_size)
x_t = t * noise + (1 - t) * actions
target_velocity = noise - actions
pred_velocity = model(x_t, t, observations)
loss = mse(pred_velocity, target_velocity)
```

If `action_is_pad` is present, padded action steps are excluded from the MSE.

The important difference from ACT is that the model does not directly predict `actions`. It predicts the velocity field at an interpolated point `x_t`. In other words, it learns "which direction should this noisy action chunk move" for the current visual and robot-state condition.

### Inference Path

Inference starts from random Gaussian action noise:

```python
x_1 = torch.randn(batch, chunk_size, action_dim)
```

Then the model integrates from `t=1` to `t=0` using a simple Euler solver:

```python
dt = -1.0 / num_inference_steps
for step in range(num_inference_steps):
    t = 1.0 + step * dt
    velocity = model(x_t, t, observations)
    x_t = x_t + dt * velocity
```

The final `x_t` is treated as the normalized action chunk. `select_action` stores the first `n_action_steps` actions in a queue and returns one action per control step, matching ACT's deployment behavior.

---

## Training Objective 1: Bimanual Assembly-Part Grasping

This experiment uses the same task and hardware setup as [ACT Training Objective 2](./act.md#training-objective-2-bimanual-assembly-part-grasping), so the result can be compared directly with the ACT baseline.

### Training Log 1 of Objective 1

#### Dataset Recording

I used the dataset recorded for [ACT Training Objective 2, Log 1](./act.md#training-log-1-of-objective-2).

#### Training script

I used the same setting as [ACT Training Objective 2, Log 1](./act.md#training-log-1-of-objective-2):

```bash
export HF_HOME=./.cache/huggingface
export HF_LEROBOT_HOME=./.cache/lerobot
export HF_ENDPOINT=https://hf-mirror.com

DATASET_ID=rm65_bimanual_grasp_and_place
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
```

#### Evaluation

In this evaluation, `canfd_follow` was disabled. The actual rollout speed was about 10 FPS; when `canfd_follow` was enabled, the arms showed noticeable lag. Therefore, rollout `fps` was set to 10 to better match the real execution speed.

```bash
PRETRAINED=rm65_action_flow_v1_bimanual_grasp_and_place_episodes_20_steps_20000_batch_16_chunk_50
TASK_PROMPT="grasp two assembly parts and place them in the center"

TASK_NAME=rollout_$PRETRAINED
FOLDER_PATH="./runs/rollout_dataset/$TASK_NAME"

if [ -d "$FOLDER_PATH" ]; then
    RESUME=true
else
    RESUME=false
fi

lerobot-rollout \
    --config_path=configs/rm65_left_right_wrist_front_top.yaml \
    --strategy.type=highlight \
    --inference.type=sync \
    --robot.left_arm.canfd_follow=false \
    --robot.right_arm.canfd_follow=false \
    --robot.left_arm.max_relative_target=1000.0 \
    --robot.right_arm.max_relative_target=1000.0 \
    --policy.path=./runs/train/$PRETRAINED/checkpoints/last/pretrained_model \
    --policy.temporal_ensemble_coeff=0.01 \
    --policy.n_action_steps=1 \
    --dataset.repo_id=local/$TASK_NAME \
    --dataset.root=$FOLDER_PATH \
    --dataset.single_task="$TASK_PROMPT" \
    --dataset.push_to_hub=false \
    --resume=$RESUME \
    --device=cuda \
    --fps=10 \
    --duration=120 \
    --task="$TASK_PROMPT" \
    --return_to_initial_position=false \
    2>&1 | tee ./logs/rollout_$PRETRAINED.log
```

- Success rate: 8 / 8

- Thinking
  1. Its whole performance is similar to [ACT Training Objective 2, Log 1](./act.md#training-log-1-of-objective-2): the grippers do not always grasp the small assembly part along its length, and the gripper sometimes lightly contacts the table while grasping. Unlike ACT, this table contact mainly appears when grasping the large assembly part rather than the small assembly part.
  2. I used ACT-style temporal ensembling during rollout. Without it, `policy.n_action_steps=10` caused unstable motion: before moving toward the assembly part, the arms often moved a short distance in the opposite direction, producing visible twitching. During placement, this instability sometimes caused one part to be released while the other remained gripped and stuck. If increase `policy.n_action_steps` like 20 or more, It will reduce the frequency of twitching and move more smoothly, but it also increases risk of contacting the table. With `policy.n_action_steps=1`, both arms tended to get stuck near the beginning of the rollout.
  3. `policy.n_action_steps=10` rollout video:
    
        <video controls width="300">
        <source src="assets/action_flow/ob_1_grasp_and_place_fail_0.mp4" type="video/mp4">
        </video>
  4. This behavior is likely caused by inconsistency between consecutive action chunks. ActionFlow samples each chunk from fresh random noise, so two neighboring predictions can disagree even when the observation changes only slightly. With `n_action_steps=10`, the policy executes the first 10 actions from one sampled chunk, then replans from a newly sampled chunk; if the new chunk starts with a different local direction, the arm can briefly move backward before continuing forward. With `n_action_steps=1`, the policy replans every step, which makes it more reactive but also more sensitive to sampling noise and small observation changes. Temporal ensembling reduces this problem by averaging overlapping predictions across successive chunks.
  5. I think add more better demonstrations will improve `Action Flow V1` performence, 20 episodes is very small for bimanual control.

---

### Training Log 2 of Objective 1

#### Dataset Recording

Use same dataste of [ACT log 3 of objective 2](act.md#training-log-3-of-objective-2).

Dataset information:

  | Situation ID | Situation Describtion | Record FPS | Total episodes | Total frames |
  | :---: | :---: | :---: | :---: | :---: |
  | 1 | 1 visible part on each side | 30 | 40 | 13095 |
  | 2 | 2 visible parts on each side | 30 | 60 | 34977 |
  | 3 | 3 visible parts on each side | 30 | 80 | 67884 |
  | 4 | Mixed visibility between left and right sides | 30 | 60 | 32852 |

The final combined dataset is named:

```text
rm65_bimanual_farthest_to_closest_assembly_parts_to_center
```

#### Training

```bash
export HF_HOME=./.cache/huggingface
export HF_LEROBOT_HOME=./.cache/lerobot
export HF_ENDPOINT=https://hf-mirror.com

DATASET_ID=rm65_bimanual_farthest_to_closest_assembly_parts_to_center
TRAINING_ID=rm65_action_flow_v1_bimanual_assembly_parts_to_center_episodes_240_steps_60000_batch_8_chunk_100

lerobot-train \
    --dataset.repo_id $DATASET_ID \
    --dataset.root ./runs/dataset/$DATASET_ID \
    --policy.type action_flow_v1 \
    --policy.device cuda \
    --policy.push_to_hub false \
    --policy.chunk_size 100 \
    --policy.n_action_steps 1 \
    --batch_size 8 \
    --steps 60000 \
    --eval_freq 0 \
    --save_freq 5000 \
    --log_freq 100 \
    --output_dir ./runs/train/$TRAINING_ID \
    --wandb.enable true \
    --wandb.project lerobot_action_flow_rm65 \
    --job_name $TRAINING_ID \
    2>&1 | tee ./logs/train_$TRAINING_ID.log
```