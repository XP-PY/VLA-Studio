# Action Flow Training Log

*Last updated: July 23, 2026*

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

#### Training Configuration

I used the same setting as [ACT Training Objective 2, Log 1](./act.md#training-log-1-of-objective-2):

Dataset: `rm65_bimanual_grasp_and_place`

Output policy: `rm65_action_flow_v1_bimanual_grasp_and_place_episodes_20_steps_20000_batch_16_chunk_50`

| Parameter | Configured value | Default value |
|---|---:|---:|
| `policy.push_to_hub` | `false` | `true` |
| `policy.n_action_steps` | `1` | `10` |
| `batch_size` | `16` | `8` |
| `steps` | `20,000` | `100,000` |
| `eval_freq` | `0` (disabled) | `20,000` |
| `save_freq` | `5,000` | `20,000` |
| `log_freq` | `100` | `200` |
| `wandb.enable` | `true` | `false` |
| `wandb.project` | `lerobot_rm65_act` | `lerobot` |

> **Note:** `policy.chunk_size=50` uses the Action Flow V1 default. All other unlisted parameters also use their default values.

#### Evaluation

In this evaluation, `canfd_follow` was disabled. The actual rollout speed was about 10 FPS; when `canfd_follow` was enabled, the arms showed noticeable lag. Therefore, rollout `fps` was set to 10 to better match the real execution speed. The rollout dataset was resumed when its output directory already existed.

Robot config: `configs/rm65_left_right_wrist_front_top.yaml`

Checkpoint: `rm65_action_flow_v1_bimanual_grasp_and_place_episodes_20_steps_20000_batch_16_chunk_50/checkpoints/last/pretrained_model`

Task: `grasp two assembly parts and place them in the center`

| Parameter | Configured value | Default value |
|---|---:|---:|
| `strategy.type` | `highlight` | `base` |
| `robot.left_arm.max_relative_target` | `1000.0` | `10.0` |
| `robot.right_arm.max_relative_target` | `1000.0` | `10.0` |
| `policy.temporal_ensemble_coeff` | `0.01` | `None` |
| `policy.n_action_steps` | `1` | `10` |
| `dataset.push_to_hub` | `false` | `true` |
| `fps` | `10` | `30` |
| `duration` | `120` | `0` (unlimited) |
| `return_to_initial_position` | `false` | `true` |

> **Note:** `inference.type=sync` and both arms' `canfd_follow=false` use their defaults. All other unlisted parameters also use their default values.

- Success rate: 8 / 8

- Phenomenon and Analysis
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

This experiment uses the same dataset as [ACT Training Log 4 of Objective 2](act.md#training-log-4-of-objective-2).

Dataset information:

  | Situation ID | Situation Description | Record FPS | Total episodes | Total frames |
  | :---: | :---: | :---: | :---: | :---: |
  | 1 | 1 visible part on each side | 30 | 40 | 13095 |
  | 2 | 2 visible parts on each side | 30 | 60 | 34977 |
  | 3 | 3 visible parts on each side | 30 | 80 | 67884 |
  | 4 | Mixed visibility between left and right sides | 30 | 60 | 32852 |

The final combined dataset is named:

```text
rm65_bimanual_farthest_to_closest_assembly_parts_to_center
```

#### Training Configuration

Similar to [ACT Training Log 4 of Objective 2](act.md#training-log-4-of-objective-2), I trained Action Flow V1 with `policy.chunk_size` values of 20, 30, and 40. The saved training configurations show that each policy was trained for 40,000 steps with batch size 8.

Output policies:

- `rm65_action_flow_v1_bimanual_assembly_parts_to_center_episodes_240_steps_40000_batch_8_chunk_20`
- `rm65_action_flow_v1_bimanual_assembly_parts_to_center_episodes_240_steps_40000_batch_8_chunk_30`
- `rm65_action_flow_v1_bimanual_assembly_parts_to_center_episodes_240_steps_40000_batch_8_chunk_40`

| Parameter | Policy 1 | Policy 2 | Policy 3 | Default value |
|---|---:|---:|---:|---:|
| `policy.push_to_hub` | `false` | `false` | `false` | `true` |
| `policy.chunk_size` | `20` | `30` | `40` | `50` |
| `policy.n_action_steps` | `1` | `1` | `1` | `10` |
| `steps` | `40,000` | `40,000` | `40,000` | `100,000` |
| `eval_freq` | `0` | `0` | `0` | `20,000` |
| `log_freq` | `100` | `100` | `100` | `200` |
| `wandb.enable` | `true` | `true` | `true` | `false` |
| `wandb.project` | `lerobot_action_flow_rm65` | `lerobot_action_flow_rm65` | `lerobot_action_flow_rm65` | `lerobot` |

> **Note:** `batch_size=8` and `save_freq=20,000` use the training-pipeline defaults. All other unlisted parameters also use their default values.

#### Evaluation

In this rollout evaluation, I enabled Real-Time Chunking (RTC). RTC is more suitable than ACT-style temporal ensembling for this flow-matching policy because it explicitly reuses the unexecuted part of the previous chunk during the next denoising process.

| Parameter | Configured value | Default value |
|---|---:|---:|
| `inference.type` | `rtc` | `sync` |
| `inference.queue_threshold` | `10` | `30` |

> **Note:** `inference.rtc.execution_horizon=10`, `inference.rtc.max_guidance_weight=10`, and `inference.rtc.prefix_attention_schedule=LINEAR` use their defaults. All other unlisted parameters also use their default values.

Some individual rollouts changed the flow integration step count:

| Parameter | Tested values | Default value |
|---|---:|---:|
| `policy.num_inference_steps` | `5`, `7` | `10` |

- Phenomenon
    1. All three policies show different degrees of discontinuity between the current executed action and the next predicted chunk. Both arms often fail to align with the assembly parts. In some cases, the robot successfully grasps a part, but the gripper command is not stable: it slightly opens and closes during transfer, causing the part to drop.

        Rollout video with `chunk_size=20`:

        <video controls width="600">
            <source src="assets/action_flow/ob_1_log_2_rollout_0.mp4" type="video/mp4">
        </video>

        Rollout video with `chunk_size=30`:

        <video controls width="600">
            <source src="assets/action_flow/ob_1_log_2_rollout_1.mp4" type="video/mp4">
        </video>

        Rollout video with `chunk_size=40`:

        <video controls width="600">
            <source src="assets/action_flow/ob_1_log_2_rollout_2.mp4" type="video/mp4">
        </video>

    2. Compared with [ACT](./act.md#training-log-4-of-objective-2), `Action Flow V1` shows one encouraging behavior: it does not follow the fixed sequence and fixed position prior as strongly as ACT. This suggests that `Action Flow V1` may rely more on visual information. However, its grasp success rate is still almost zero because the grasp point is not accurate and the executed actions are not stable enough.
    3. Although the default `num_inference_steps` is 10, reducing it significantly improves grasp-point accuracy and gripper-command stability. This suggests that more denoising steps do not necessarily produce better real-robot rollout behavior. In this task, 10 inference steps may over-refine the action chunk or increase chunk-to-chunk inconsistency, while 7 or 5 steps produce smoother and more stable grasp targets. However, the grasp point is still not accurate enough for consistently successful pickup:

        Successful rollout video with `chunk_size=40` and `num_inference_step=7`:

        <video controls width="600">
            <source src="assets/action_flow/ob_1_log_2_rollout_3.mp4" type="video/mp4">
        </video>

        Fail rollout video with `chunk_size=40` and `num_inference_step=7`:

        <video controls width="600">
            <source src="assets/action_flow/ob_1_log_2_rollout_4.mp4" type="video/mp4">
        </video>

        Nearly successful rollout video with `chunk_size=40` and `num_inference_step=5`:

        <video controls width="600">
            <source src="assets/action_flow/ob_1_log_2_rollout_5.mp4" type="video/mp4">
        </video>

    4. Similar to ACT, `Action Flow V1` still has difficulty handling empty target positions. In the demonstrations, when one side has no visible part, the corresponding arm remains still. However, during rollout, if the right-side target position is empty while the left side still has visible assembly parts, the right arm may still try to grasp the empty position and then execute the place motion toward the center. This motion is noticeably slower than normal, but it still shows that the policy has not fully learned the conditional behavior of skipping an empty side.

- Analysis:
    1. `Action Flow V1` improves some low-level rollout behavior compared with ACT, especially after tuning `num_inference_steps`, but it does not fully solve the high-level conditional behavior problem. Like ACT, it can still execute a memorized empty-position grasp when one side has no visible part.
    2. The best low-level improvement comes from reducing `num_inference_steps`. This indicates that rollout quality is sensitive to the inference-time flow integration setting. In this task, the default value of 10 may generate action chunks that are too sharp or less consistent across consecutive RTC updates. Fewer inference steps make the action sequence smoother and make the gripper command more stable.
    3. Even after this improvement, grasp-point precision remains a bottleneck. The policy can often move toward the correct part, but the final grasp point is still slightly offset from the part center or length direction. For small assembly parts, this small error is enough to cause missed grasps, unstable grasps, or dropped parts during transfer.
    4. The empty-side failure suggests that the policy has not learned a reliable conditional rule: if no part is visible on one side, that arm should stay still. This is not mainly an RTC or `num_inference_steps` issue, because the same type of behavior also appears in ACT. It is more likely a limitation of visual grounding and action-condition learning in the current dataset/model setup.
    5. The next improvement should focus on object localization, centered grasp prediction, and explicit empty-side behavior. Possible directions include testing `num_inference_steps` values around 5 and 7 more systematically, adding gripper-specific smoothing or loss weighting, improving the visual encoder, adding more demonstrations that emphasize centered grasps, and adding more clear examples where one arm must remain still because its side is empty.

---

## Action Flow V2

### Design Motivation

Action Flow V1 uses a visual encoder without language conditioning. Its evaluation showed that a policy can reproduce smooth demonstrated trajectories while still relying on fixed object locations and action-order priors. Action Flow V2 is designed to make target selection depend more directly on the task prompt and current camera observations.

The policy therefore combines a vision-language model (VLM), a continuous flow-matching action expert, and an auxiliary spatial-grounding head. The grounding objective is intended to improve the shared visual-language representation. It does not replace the VLM representation with a detector output.

### Policy Structure

~~~text
camera images + task prompt
            |
            v
       shared VLM
            |
            +----------------------------+
            |                            |
            v                            v
prompt-conditioned tokens        spatial-grounding head
            |                            |
robot state + noisy actions              v
            |                    relevance prediction
            v
 flow-matching action expert
            |
            v
 continuous action chunk
~~~

The VLM processes all camera images and the task prompt while retaining camera identity and image token locations. The action expert combines these prompt-conditioned tokens with the robot state and noisy actions to predict a continuous action flow. The grounding head reads the same prompt-conditioned image tokens and predicts which image regions are relevant to the instruction.

The relevance prediction is auxiliary: it can be visualized during inference, but it is not directly fed back into the action expert. This makes it possible to test whether spatial supervision improves the shared representation without changing the policy's action interface. Future experiments may use other grounding targets or explicitly condition actions on the predicted spatial output.

### Auxiliary Data Record Pipeline

Action Flow V2 starts from a normal LeRobot demonstration dataset containing camera observations, robot state, actions, and task prompts. Spatial labels are generated as a separate, frame-aligned annotation source and then merged with the demonstrations before training.

~~~text
task prompts + camera videos
            |
            v
 identify prompt-relevant entities
            |
            v
 propose and review target locations
            |
            v
 track entities through each video
            |
            v
 build per-camera spatial targets
            |
            v
merge with robot observations and actions
            |
            v
 train action loss + grounding loss
~~~

Automatic detectors or grounding models are used only to initialize annotations. Their proposals must be reviewed, after which a video segmentation or tracking model propagates each accepted entity through the episode. Separate entity tracks are retained so later experiments can derive different targets, such as binary relevance, source and destination regions, or phase-dependent labels.

Annotation validity is stored separately from target visibility. A camera can have a valid empty target when no relevant entity is visible, while uncertain or incorrect annotations must be excluded from the grounding loss. Exact detectors, tracking settings, target formats, loss functions, resolutions, and model-freezing choices are experiment-specific and are documented in the corresponding training log.

### Training Objective 1: Single-Arm Table Preparation and Bussing

The task design and demonstration-recording plan follow [PI0 Training Objective 1](./pi_0.md#training-objective-1-single-arm-table-preparation-and-bussing).

#### Objective

The first Action Flow V2 experiment uses the single-arm table-preparation and table-bussing task.
The policy must use the language prompt to select the relevant vegetables, plates, or collection box instead of reproducing a fixed object-position trajectory.

The experiment has two connected objectives:

1. Predict continuous RM65 joint and gripper action chunks with the SmolVLA-style flow action expert.
2. Predict a task-relevance heatmap for the top and wrist cameras from the shared prompt-conditioned VLM tokens.

The main question is whether auxiliary spatial supervision improves target recognition and action accuracy without directly passing the predicted heatmap into the action expert.

### Training Log 1 of Objective 1

#### Dataset

Base dataset:

```text
runs/dataset/rm65_single_table_preparation_and_bussing
```

The cleaned dataset contains 203 successful episodes, 92,545 frames, and 52 prompt variants recorded at 30 FPS. Each frame contains:

- `observation.images.top`: 640 x 480 overhead workspace image.
- `observation.images.wrist`: 640 x 480 end-effector image.
- `observation.state`: six RM65 joint angles and one continuous gripper position.
- `action`: six recorded future joint-angle labels and one continuous gripper label.

| Task | Prompt template | Episodes | Total frames |
|---|---|---:|---:|
| Place one vegetable | `Place the {vegetable} on the {plate_color} plate.` | 48 | 16,212 |
| Arrange two vegetables | `Place the {vegetable_1} on the {plate_color_1} plate, then place the {vegetable_2} on the {plate_color_2} plate.` | 36 | 20,196 |
| Clear a specified plate | `Put the vegetable from the {plate_color} plate into the collection box.` | 72 | 22,149 |
| Bus the table | `Put all vegetables into the collection box.` | 47 | 33,988 |
| **Total** | **52 prompt variants** | **203** | **92,545** |

#### Relevance Annotation

This experiment uses one binary relevance channel per camera. The target is the union of every entity needed to interpret the complete prompt:

- A place-one prompt includes the manipulated vegetable and destination plate.
- A clear-plate prompt includes the source vegetable, source plate, and collection box.
- A table-bussing prompt includes all vegetables and the collection box.
- A multi-step prompt includes every named vegetable and destination plate for the full episode.

The target does not distinguish source from destination and does not change between task phases.
It is not used as a grasp-point or trajectory label.

Grounding DINO receives detector-oriented text instead of the literal dataset entity name when a
more visually specific description is needed:

| Dataset entity | Grounding DINO query | Usage |
|---|---|---|
| `tomato` | `tomato` | Source vegetable |
| `garlic` | `white garlic` | Source vegetable; the color reduces confusion with the table and paper |
| `corn` | `corn` | Source vegetable |
| `cucumber` | `green cucumber` | Source vegetable |
| `pink plate` | `pink square bowl` | Source or destination container |
| `yellow plate` | `cream square bowl` | Source or destination container; `cream` better matches its observed color |
| `blue plate` | `blue square bowl` | Source or destination container |
| `collection box` | `brown cardboard box` | Destination container |
| `vegetable on the {color} plate` | Plate query plus all four vegetable queries | Detect the specified plate and the vegetables separately, then select the vegetable whose box lies inside the plate box |

The annotations were generated as follows:

1. Parse each task prompt into deterministic entity IDs, names, roles, and detector queries.
2. Run [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) on sampled top- and wrist-camera frames to propose one initialization box for each entity.
3. Review the selected boxes and replace incorrect or missing proposals manually.
4. Use each accepted box to initialize a separate [facebook/sam2.1-hiera-small](https://huggingface.co/facebook/sam2.1-hiera-small) object track.
5. Run SAM2 on every third frame and assign the nearest tracked mask to intermediate frames.
6. Merge the requested per-object tracks into one union relevance mask for each camera frame.
7. Join the frame-aligned masks to the LeRobot data by `episode_index`, `frame_index`, and camera.

The reviewed Grounding DINO proposals and separate SAM2 object tracks are stored under:

```text
runs/sam/rm65_single_table_preparation_and_bussing
```

Keeping each object as a separate track preserves entity identity for inspection and future source/destination or phase-dependent experiments.

The full-resolution union masks are stored separately under:

```text
runs/dataset/rm65_single_table_preparation_and_bussing_afv2
```

This annotation dataset contains one Parquet file per episode and camera. Each row stores the episode and frame indices, camera name, image dimensions, compressed union mask, and annotation validity weight.

For training, the union masks are merged with the robot dataset and resized to the VLM image-token
grid:

```text
runs/dataset/rm65_single_table_preparation_and_bussing_afv2_segmentation
```

The merged dataset adds:

- `observation.relevance_mask` with shape `(2, 8, 8)`, ordered as wrist then top.
- `observation.relevance_loss_weight` with shape `(2,)`.

An all-zero mask is a valid label when no relevant entity is visible. A zero validity weight is used only when the annotation is uncertain or intentionally excluded.

#### Policy and Loss Design

The policy is initialized from `smolvla_base` and uses the local `SmolVLM2-500M-Video-Instruct` backbone. A learned camera embedding is added to each camera's image tokens before the shared multimodal transformer.

For each camera, the relevance head applies:

```text
LayerNorm -> Linear(hidden_size, 256) -> GELU -> Linear(256, 1)
```

The output token logits are reshaped to an 8 x 8 grid. The experiment trains:

```text
L_total = L_flow + 0.1 * (L_BCE + L_Dice)
```

The relevance loss is computed independently for each camera, multiplied by `observation.relevance_loss_weight`, and normalized over valid camera frames. BCE provides per-patch classification supervision, while Dice reduces the dominance of background patches when relevant regions are small.

The complete VLM and action expert remain trainable so the relevance loss can modify features shared with action prediction. Two otherwise identical policies are required for the intended ablation:

- Grounding policy: `policy.relevance_loss_weight=0.1`.
- Action-only ablation: `policy.relevance_loss_weight=0.0`.

#### Training Configuration

Output policies:

```text
rm65_action_flow_v2_single_table_preparation_and_bussing_episodes_203_steps_40000_batch_8_chunk_50_relevance_0.1
rm65_action_flow_v2_single_table_preparation_and_bussing_episodes_203_steps_40000_batch_8_chunk_50_relevance_0.0
```

| Parameter | Configured value | Action Flow V2 default |
|---|---:|---:|
| `dataset.repo_id` | `rm65_single_table_preparation_and_bussing_afv2_segmentation` | Not set |
| `dataset.root` | `runs/dataset/rm65_single_table_preparation_and_bussing_afv2_segmentation` | Not set |
| `dataset.return_uint8` | `true` | `false` |
| `policy.type` | `action_flow_v2` | Not set |
| `policy.device` | `cuda` | Automatically selected |
| `policy.pretrained_path` | `models/smolvla_base` | `None` |
| `policy.vlm_model_name` | `models/SmolVLM2-500M-Video-Instruct` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| `policy.input_features` | `null` (infer from dataset) | `{}` |
| `policy.output_features` | `null` (infer from dataset) | `{}` |
| `policy.n_action_steps` | `1` | `10` |
| `policy.relevance_loss_weight` | `0.1` (grounding) / `0.0` (ablation) | `0.1` |
| `policy.optimizer_lr` | `3e-5` | `1e-4` |
| `policy.scheduler_decay_steps` | `40,000` | `30,000` |
| `policy.push_to_hub` | `false` | `true` |
| `steps` | `40,000` | `100,000` |
| `eval_freq` | `0` (disabled) | `20,000` |
| `save_freq` | `10,000` | `20,000` |
| `log_freq` | `100` | `200` |
| `wandb.enable` | `true` | `false` |
| `wandb.project` | `lerobot_action_flow_v2_rm65` | `lerobot` |
| `wandb.disable_artifact` | `true` | `false` |

> **Note:** Parameters not listed in the table use their defaults. Important retained defaults are
> `policy.chunk_size=50`, `batch_size=8` per process, `policy.freeze_vision_encoder=false`,
> `policy.train_expert_only=false`, `policy.load_vlm_weights=true`,
> `policy.relevance_bce_weight=1.0`, `policy.relevance_dice_weight=1.0`, and
> `policy.relevance_target_size=[8, 8]`. Both policies used BF16 mixed precision on one GPU process,
> giving an effective batch size of 8.

The lower learning rate of `3e-5` reduces the risk of damaging the pretrained VLM while the newly initialized camera embeddings and relevance head learn the RM65 workspace.

During training, log at least:

- `flow_loss`
- `relevance_loss`
- `relevance_bce_loss`
- `relevance_dice_loss`
- `relevance_valid_camera_fraction`

Training should be stopped and inspected if the weighted relevance loss dominates the flow loss, or if heatmap quality improves while rollout action quality degrades.

#### Evaluation

The grounding policy (`relevance_loss_weight=0.1`) and action-only ablation
(`relevance_loss_weight=0.0`) were evaluated with the same five prompts and object layouts. Each
prompt was tested twice per policy, producing 20 paired rollouts. The policies used identical
training and rollout settings apart from the relevance-loss weight.

##### Rollout Configuration

| Parameter | Configured value | Default value |
|---|---:|---:|
| `strategy.type` | `episodic` | `base` |
| `inference.type` | `rtc` | `sync` |
| `inference.rtc.execution_horizon` | `15` | `10` |
| `inference.rtc.max_guidance_weight` | `15` | `10` |
| `inference.queue_threshold` | `25` | `30` |
| `policy.n_action_steps` | `1` | `10` |
| `policy.num_steps` | `5` | `10` |
| `robot.single_arm.max_relative_target` | `1000.0` | `100.0` in the robot YAML |
| `fps` | `20` | `30` |

> **Note:** `inference.rtc.prefix_attention_schedule=LINEAR`,
> `robot.single_arm.canfd_follow=false`, and `return_to_initial_position=true` use their
> defaults. All other unlisted parameters use their defaults. Although
> `policy.n_action_steps=1` is configured, RTC
> calls `predict_action_chunk` and manages its own action queue; continuity is controlled by
> `inference.rtc.execution_horizon`, not by `policy.n_action_steps`.

##### Results

The grounding policy's heatmaps are visualizations of an upsampled 8 x 8 relevance grid. They
should be interpreted as coarse relevant regions rather than precise segmentation boundaries.

| Prompt | Trials per policy | `0.1` successes | `0.0` successes | Paired observation |
|---|---:|---:|---:|---|
| `Place the tomato on the pink plate.` | 2 | 0 | 0 | Both policies approached the target but failed to complete the grasp and transfer. The ablation did not approach unrelated objects in these two trials. |
| `Place the cucumber on the blue plate.` | 2 | 1 | 0 | The grounding policy reacted more quickly after the cucumber slipped and attempted a more accurate re-grasp. The ablation continued the original trajectory longer and recovered less precisely. |
| `Put the vegetable from the blue plate into the collection box.` | 2 | 0 | 0 | Both policies failed, but the grounding policy showed slightly better target localization and corrective behavior. Gripper closure and grasp alignment remained unreliable. |
| `Put the vegetable from the yellow plate into the collection box.` | 2 | 0 | 0 | Both policies selected the same incorrect target, showing that the dominant grounding error was shared by both models. |
| `Place the tomato on the pink plate, then place the garlic on the blue plate.` | 2 | 0 | 0 | Both policies attempted incorrect targets. The behavior was slightly less severe with the grounding policy, but neither completed the first step reliably. |
| **Total** | **10 each** | **1 (10%)** | **0 (0%)** | **The grounding policy was slightly quicker and more precise, but the main failure modes were similar.** |

This remains a small qualitative comparison. The auxiliary relevance objective provides a modest
behavioral improvement in these paired trials, but it does not produce a reliable task policy.

##### Rollout Videos

**Place the tomato on the pink plate**

<table>
<tr><th colspan="2"><code>relevance_loss_weight=0.1</code></th></tr>
<tr>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_0.mp4" type="video/mp4">
    </video>
    <p><em>Trial 1: broad relevance and failed grasp.</em></p>
    </td>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_1.mp4" type="video/mp4">
    </video>
    <p><em>Trial 2: unstable relevance and failed transfer.</em></p>
    </td>
</tr>
<tr><th colspan="2"><code>relevance_loss_weight=0.0</code></th></tr>
<tr>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_10.mp4" type="video/mp4">
    </video>
    <p><em>Trial 1: target approach followed by grasp failure.</em></p>
    </td>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_11.mp4" type="video/mp4">
    </video>
    <p><em>Trial 2: target approach followed by transfer failure.</em></p>
    </td>
</tr>
</table>

**Place the cucumber on the blue plate**

<table>
<tr><th colspan="2"><code>relevance_loss_weight=0.1</code></th></tr>
<tr>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_2.mp4" type="video/mp4">
    </video>
    <p><em>Trial 1: successful with the pink plate removed.</em></p>
    </td>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_3.mp4" type="video/mp4">
    </video>
    <p><em>Trial 2: failed with the pink plate present.</em></p>
    </td>
</tr>
<tr><th colspan="2"><code>relevance_loss_weight=0.0</code></th></tr>
<tr>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_12.mp4" type="video/mp4">
    </video>
    <p><em>Trial 1: slow response after the cucumber slipped.</em></p>
    </td>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_13.mp4" type="video/mp4">
    </video>
    <p><em>Trial 2: inaccurate recovery and failed re-grasp.</em></p>
    </td>
</tr>
</table>

**Put the vegetable from the blue plate into the collection box**

<table>
<tr><th colspan="2"><code>relevance_loss_weight=0.1</code></th></tr>
<tr>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_4.mp4" type="video/mp4">
    </video>
    <p><em>Trial 1: insufficient gripper closure.</em></p>
    </td>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_5.mp4" type="video/mp4">
    </video>
    <p><em>Trial 2: misaligned grasp.</em></p>
    </td>
</tr>
<tr><th colspan="2"><code>relevance_loss_weight=0.0</code></th></tr>
<tr>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_14.mp4" type="video/mp4">
    </video>
    <p><em>Trial 1: failed target approach and grasp.</em></p>
    </td>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_15.mp4" type="video/mp4">
    </video>
    <p><em>Trial 2: weak correction and failed grasp.</em></p>
    </td>
</tr>
</table>

**Put the vegetable from the yellow plate into the collection box**

<table>
<tr><th colspan="2"><code>relevance_loss_weight=0.1</code></th></tr>
<tr>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_6.mp4" type="video/mp4">
    </video>
    <p><em>Trial 1: selected an incorrect target.</em></p>
    </td>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_7.mp4" type="video/mp4">
    </video>
    <p><em>Trial 2: selected the same incorrect target.</em></p>
    </td>
</tr>
<tr><th colspan="2"><code>relevance_loss_weight=0.0</code></th></tr>
<tr>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_16.mp4" type="video/mp4">
    </video>
    <p><em>Trial 1: repeated the grounding policy's target error.</em></p>
    </td>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_17.mp4" type="video/mp4">
    </video>
    <p><em>Trial 2: repeated the same incorrect grasp behavior.</em></p>
    </td>
</tr>
</table>

**Place the tomato on the pink plate, then place the garlic on the blue plate**

<table>
<tr><th colspan="2"><code>relevance_loss_weight=0.1</code></th></tr>
<tr>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_8.mp4" type="video/mp4">
    </video>
    <p><em>Trial 1: the initial grasp failed.</em></p>
    </td>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_9.mp4" type="video/mp4">
    </video>
    <p><em>Trial 2: attempted an incorrect target.</em></p>
    </td>
</tr>
<tr><th colspan="2"><code>relevance_loss_weight=0.0</code></th></tr>
<tr>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_18.mp4" type="video/mp4">
    </video>
    <p><em>Trial 1: attempted an incorrect target.</em></p>
    </td>
    <td align="center">
    <video controls width="100%">
        <source src="assets/action_flow/v2_ob_1_log_1_rollout_19.mp4" type="video/mp4">
    </video>
    <p><em>Trial 2: stronger incorrect-target behavior than the grounding policy.</em></p>
    </td>
</tr>
</table>

##### Relevance Annotation Quality

The top-camera relevance annotations are generally usable, but many wrist-camera annotations are
not reliable. In placement tasks, the initialization box was sometimes selected from a late frame
after the object had already been placed on the destination plate. At that point, the object and
plate overlap in the wrist view, so the initialization proposal can cover them as one combined
region.

This is problematic even though the final training target is a union mask. The object and plate
move independently during most of the episode, but SAM2 receives a single fused initialization
region from the contact phase. When that region is propagated through the episode, the track may
follow only one entity, include surrounding plate pixels, or fail when the object separates from
the plate. The resulting wrist labels can therefore be broad, inconsistent, or attached to the
wrong entity. Repeated errors of this type teach the relevance head incorrect spatial
correspondence.

##### Analysis

1. **The relevance objective provides only a modest benefit.** The grounding policy completed one
   trial, reacted more quickly to a dropped object, and produced slightly more accurate approaches
   and re-grasps. The ablation completed no trials. This direction is favorable, but the difference
   between one success and zero successes is too small to establish a reliable improvement.

2. **Both policies retain the same dominant failure modes.** They often approach the correct
   region but fail to align or close the gripper, continue a demonstrated trajectory after losing
   the object, or select the same incorrect target. The identical error in the yellow-plate task
   is especially important: the auxiliary loss did not overcome the shared scene-layout or
   object-co-occurrence prior.

3. **The action expert does not directly follow the predicted heatmap.** The heatmap is an
   auxiliary output rather than an action input. Consequently, an incorrect heatmap and incorrect
   motion indicate weak shared visual-language features; the heatmap is not directly commanding
   the robot to grasp an unrelated object.

4. **Wrist-camera label noise limits the grounding policy.** Many initialization boxes contain an
   already combined object-and-plate region, producing inconsistent wrist targets. The similar
   failures of the action-only ablation show that this noise is not the only problem, but it can
   weaken or misdirect the additional supervision and hide its potential benefit.

5. **Distractors expose weak language-to-entity binding.** The only successful rollout occurred
   when the pink plate was removed from the cucumber-to-blue-plate scene. With distractors present,
   both policies relied more strongly on learned scene priors than on the requested object and
   destination.

6. **The current relevance target cannot represent execution order.** A single union mask
   highlights every source and destination throughout a multi-step episode. It teaches general
   task relevance but does not distinguish entity roles, the active subtask, or whether the policy
   should execute the first or second placement.

7. **Spatial grounding and manipulation remain separate bottlenecks.** The 8 x 8 grid is coarse
   for small vegetables and precise grasp points, while correct target selection can still fail
   through grasp misalignment, insufficient gripper closure, or poor recovery. Improving the
   relevance target alone will not guarantee successful manipulation.

Before retraining, wrist initialization frames should be chosen while the source object and
destination plate are visibly separated. They should be initialized as separate object tracks,
with negative clicks used to exclude overlapping plate or background pixels. Uncertain fused
tracks should be corrected manually or assigned `loss_weight=0`; the wrist image can remain a
policy input even when its relevance loss is disabled for those frames.

The next evaluation should report task success, grasp success, and held-out heatmap Dice/IoU
separately. It should use more repeated paired scenes with and without distractors and log
predicted gripper commands to determine whether closure failures originate in the policy output or
the physical gripper execution.

### Training Log 2 of Objective 1

#### Relevance Annotation

The second annotation pass replaces open-vocabulary text grounding with
[YOLOE](https://docs.ultralytics.com/models/yoloe/) visual prompting. This change addresses two
problems observed in Training Log 1: text queries sometimes select semantically related but
incorrect objects, and a box around an already combined object-and-plate region can initialize a
fused SAM2 track. YOLOE is used only to obtain reviewed positive initialization points; SAM2 still
generates the frame-level segmentation masks.

##### Visual Reference

YOLOE is initialized once from a 640 x 480 top-camera reference image containing clear examples of
the task objects. Each bounding box is assigned a stable visual-prompt class:

| Class ID | Reference class | Dataset entity |
|---:|---|---|
| 0 | `blue_plate` | `blue plate` |
| 1 | `yellow_plate` | `yellow plate` |
| 2 | `pink_plate` | `pink plate` |
| 3 | `garlic` | `garlic` |
| 4 | `tomato` | `tomato` |
| 5 | `cucumber` | `cucumber` |
| 6 | `corn` | `corn` |
| 7 | `collection_box` | `collection box` |

The reference definition is stored in:

```text
examples/action_flow_v2/ref/yoloe_reference_realsense_viewer_with_collection_box.yaml
```

This is a visual class mapping, not a text-query mapping. The task parser still determines which
entities are relevant to each instruction, but only those corresponding YOLOE classes are retained
for review and SAM2 initialization.

##### YOLOE Proposal Generation

`examples/action_flow_v2/annotate_yoloe.py` scans the top and wrist videos and writes detection
artifacts under:

```text
runs/yoloe/rm65_single_table_preparation_and_bussing
```

| Parameter | Value |
|---|---|
| Model | `yoloe-26x-seg.pt` |
| Cameras | `top`, `wrist` |
| Frame selection | Every episode frame |
| Input size | `640` |
| Confidence threshold | `0.15` |
| IoU threshold | `0.7` |
| Candidates per entity | `3` |
| Candidate quality threshold | `0.3` |
| YOLOE mask threshold | `0.5` |
| Inference batch size | `4` |

For every prompt-relevant entity, YOLOE predicts a box and segmentation mask. A positive point is
chosen at the mask pixel farthest from its boundary, which places it inside the object rather than
on an uncertain edge. The box center is used only when YOLOE provides no usable mask.

Candidates are ranked using detection confidence, visible area, border clearance, and image
sharpness. Source objects receive a penalty for late frames so that their initialization is
preferably captured before manipulation. Up to three temporally separated candidates are retained
for each `(episode, camera, object_id)` item.

##### Point Review

The YOLOE result is reviewed with `examples/action_flow_v2/review_yoloe.py`. Unlike the first
annotation workflow, review corrections are not transcribed into a Markdown table and then repeated
manually in the SAM2 interface. The reviewer operates directly on the original video frames:

| Review action | Saved result | SAM2 behavior |
|---|---|---|
| Leave unchanged | No explicit override | Use the selected YOLOE point |
| Select a candidate and use its point | `candidate_point` | Use that candidate's frame and positive point |
| Click the correct target in any frame | `manual_point` | Use the clicked frame and coordinates as a positive point |
| Mark `Not visible` | `not_visible` | Produce an all-zero track for that entity and camera |

Each action is saved immediately and can be resumed from:

```text
runs/yoloe/rm65_single_table_preparation_and_bussing/review/decisions.json
```

If an absent target is not marked `not_visible`, a missing YOLOE proposal leaves the
episode-camera pair unresolved and batch SAM2 skips it. More seriously, an unreviewed false
positive is treated as the target and produces an incorrect track. Therefore, every absent target
and every false positive must be corrected during review.

##### SAM2 Tracking

`examples/action_flow_v2/annotate_sam2.py` loads the review file with
`prompt_source=yoloe`. Untouched targets use the selected YOLOE points, while explicit decisions
override them before the SAM2 session is constructed.

| Parameter | Value |
|---|---|
| Model | `sam2.1-hiera-small` |
| Prompt type | One positive point per visible entity |
| Frame stride | `2` |
| Intermediate frames | Mask from the nearest directly tracked frame |
| Top-camera range | Propagate forward and backward to cover the full episode |
| Wrist-camera range | Begin at the reviewed reference-point frame |
| Missing reviewed target | Skip the unresolved episode-camera pair |
| `not_visible` target | Save an all-zero per-object track |

Reference-point frames are always processed even when they are not divisible by the frame stride.
SAM2 tracks each entity independently, after which the per-object masks are unioned into one binary
task-relevance mask per frame.

The separate object tracks, resumable workspaces, batch report, and preview videos are stored under:

```text
runs/sam/rm65_single_table_preparation_and_bussing_yoloe
```

The full-resolution union masks and frame validity weights are stored under:

```text
runs/dataset/rm65_single_table_preparation_and_bussing_afv2_yoloe
```

Each annotation row is aligned by `episode_index`, `frame_index`, and camera, and contains the image
size, compressed binary mask, and `loss_weight`. The source robot dataset remains unchanged.

##### Dense Training Dataset

For policy training, `examples/action_flow_v2/merge_segmentation_dataset.py` joins the YOLOE/SAM2
sidecar with the original LeRobot dataset. The native-resolution SAM2 masks remain unchanged in
the sidecar dataset. The merged copies are resized to 512 x 512 using the same
aspect-ratio-preserving resize and left/top padding convention as the policy images. This retains
dense spatial supervision instead of reducing each mask to the VLM's 8 x 8 image-token grid.

The camera order is wrist then top. The merged dataset adds:

- `observation.relevance_mask`: `uint8[2, 512, 512]`.
- `observation.relevance_loss_weight`: `float32[2]`.

The resulting training dataset contains 203 episodes and 92,545 frames at 30 FPS:

```text
runs/dataset/rm65_single_table_preparation_and_bussing_afv2_yoloe_dense_segmentation
```

#### Policy and Loss Design

This experiment uses Action Flow V2.1. It keeps the prompt-conditioned SmolVLA action-flow
pipeline from Action Flow V2 but replaces the 8 x 8 relevance classifier with a learned dense
decoder. For each camera, the decoder reshapes the corresponding image tokens to their spatial
grid and applies six learned 2x upsampling stages:

```text
8 x 8 -> 16 x 16 -> 32 x 32 -> 64 x 64 -> 128 x 128 -> 256 x 256 -> 512 x 512
```

The decoder channels are `(256, 128, 64, 32, 16, 8)`, followed by a one-channel output projection.
The relevance target size matches the 512 x 512 policy image size, so the prediction and target
remain spatially aligned.

The training objective is:

```text
L_total = L_flow + 0.1 * (L_BCE + L_Dice)
```

BCE supervises each output pixel, while Dice reduces the dominance of background pixels when the
relevant objects occupy a small image region. `observation.relevance_loss_weight` excludes invalid
camera annotations without treating a valid empty mask as missing data.

The policy is initialized from the local `smolvla_base` checkpoint and local
`SmolVLM2-500M-Video-Instruct` backbone. The VLM, action expert, state projection, camera
embeddings, and dense relevance decoder are all trainable. The successful startup reports
405 million learnable parameters out of 452 million total parameters.

#### Training Configuration

Output policy:

```text
rm65_action_flow_v21_single_table_preparation_and_bussing_afv2_yoloe_dense_segmentation_episodes_203_steps_40000_batch_8_chunk_50_relevance_0.1
```

| Parameter | Configured value | Action Flow V2.1 default |
|---|---:|---:|
| `dataset.repo_id` | `rm65_single_table_preparation_and_bussing_afv2_yoloe_dense_segmentation` | Not set |
| `dataset.root` | `runs/dataset/rm65_single_table_preparation_and_bussing_afv2_yoloe_dense_segmentation` | Not set |
| `dataset.return_uint8` | `true` | `false` |
| `policy.pretrained_path` | `models/smolvla_base` | `None` |
| `policy.vlm_model_name` | `models/SmolVLM2-500M-Video-Instruct` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| `policy.input_features` | `null` (infer from dataset) | `{}` |
| `policy.output_features` | `null` (infer from dataset) | `{}` |
| `policy.n_action_steps` | `1` | `10` |
| `policy.optimizer_lr` | `3e-5` | `1e-4` |
| `policy.scheduler_warmup_steps` | `2,000` | `1,000` |
| `policy.scheduler_decay_steps` | `40,000` | `30,000` |
| `policy.push_to_hub` | `false` | `true` |
| `steps` | `40,000` | `100,000` |
| `eval_freq` | `0` (disabled) | `20,000` |
| `save_freq` | `10,000` | `20,000` |
| `log_freq` | `100` | `200` |
| `wandb.enable` | `true` | `false` |
| `wandb.project` | `lerobot_action_flow_v2_rm65` | `lerobot` |
| `wandb.disable_artifact` | `true` | `false` |
| `accelerate --mixed_precision` | `bf16` | `no` |

> **Note:** Parameters not listed in the table use their Action Flow V2.1 defaults. Important
> retained values are `policy.chunk_size=50`, `batch_size=8`,
> `policy.relevance_loss_weight=0.1`, `policy.relevance_bce_weight=1.0`,
> `policy.relevance_dice_weight=1.0`, `policy.relevance_target_size=[512, 512]`,
> `policy.freeze_vision_encoder=false`, `policy.train_expert_only=false`,
> `policy.train_state_proj=true`, and `policy.load_vlm_weights=true`. Training uses one GPU
> process, giving an effective batch size of 8.

---

### Training Log 3 of Objective 1

#### Motivation

Action Flow V2 and V2.1 train the VLM and continuous action expert jointly. The flow-matching loss
therefore changes the pretrained VLM weights directly, while the relevance objective supervises
spatial understanding through a separate heatmap head. Training Log 3 instead proposes **Action
Flow V2.2**, a knowledge-insulated policy inspired by
[pi0.5-KI](https://www.pi.website/download/pi05_KI.pdf).

The experiment has three objectives:

1. Represent future RM65 joint commands as offsets from the current joint state so that the policy
   learns motion relative to the current pose rather than memorizing absolute workspace poses.
2. Train the VLM autoregressively to predict prompt-relevant target boxes and FAST-discretized
   action tokens.
3. Train a separate action expert to predict continuous actions through flow matching while
   preventing the flow loss from changing the VLM.

Target boxes replace the dense segmentation objective in this experiment. They provide explicit
camera, entity, role, visibility, and location supervision through the VLM token stream. They are
intended to improve language-to-entity binding, but they remain coarse grounding labels rather
than grasp-point labels.

#### Policy Structure

The implemented policy has one trainable VLM branch and one trainable continuous-action branch:

```text
wrist/top images + prompt + current state
                    |
                 SmolVLM
              /             \
     target-box tokens     FAST action tokens
          box CE loss        FAST CE loss
              \             /
          per-layer VLM prefix K/V
                    |
             stop-gradient
                    |
       flow-matching action expert
       + noisy actions + timestep
                    |
       continuous relative actions
```

The VLM receives gradients from target-box and FAST-token prediction. The action expert receives
gradients from flow matching. The flow loss must not update the vision encoder, connector,
language model, token embeddings, LM head, state projection, or camera embeddings.

The policy is implemented in a separate package:

```text
repos/lerobot/src/lerobot/policies/action_flow_v2_2/
```

The package contains:

| File | Responsibility |
|---|---|
| `configuration_action_flow_v2_2.py` | Token vocabulary, relative-action, loss-weight, model, and inference configuration |
| `data_contracts_action_flow_v2_2.py` | Structured box sidecars, feature metadata, model-batch keys, and schema validation |
| `processor_action_flow_v2_2.py` | Relative-action conversion, normalization, box serialization, and FAST tokenization |
| `modeling_action_flow_v2_2.py` | Autoregressive VLM branch, detached prefix cache, flow expert, losses, and inference |
| `__init__.py` | Policy exports and registration |

Action Flow V2 and V2.1 must remain unchanged so their checkpoints and ablation results stay
reproducible.

##### Relative Action Representation

Only the six RM65 joint targets are converted to relative values. The continuous gripper target
remains absolute:

```text
delta_joint[t, k, d] = absolute_joint[t + k, d] - current_joint[t, d]
gripper_target[t, k] = absolute_gripper[t + k]
```

Here, `k` indexes the future step in a chunk and `d` indexes one of the six joints. This is a
current-state-relative action chunk, not a sequence of frame-to-frame differences.
`observation.state` remains the current absolute robot state.

The preprocessing order must be:

```text
absolute action
-> relative joint conversion
-> normalization with relative-action statistics
-> shared normalized action chunk
   -> FAST tokenizer
   -> flow-matching target
```

The FAST and flow branches must receive the same normalized relative action values. The original
absolute-action statistics cannot be reused for joint deltas. Relative-action statistics must be
computed after conversion and saved with the policy preprocessor. During rollout, the continuous
prediction is unnormalized and converted back to an absolute RM65 command:

```text
absolute_joint[k] = current_joint + predicted_delta_joint[k]
```

The gripper value passes through without the relative offset.

##### Target-Box Labels

The box dataset is derived from the reviewed, entity-specific SAM2 tracks under:

```text
runs/sam/rm65_single_table_preparation_and_bussing_yoloe
```

The union relevance masks used by V2 and V2.1 are not sufficient because a token sequence must
associate each box with one entity. For every frame, the V2.2 sidecar will store one structured
record per prompt-relevant entity and camera:

```text
(camera, entity_name, entity_role, visibility, x1, y1, x2, y2, loss_weight)
```

The first experiment keeps all entities required by the complete prompt active throughout the
episode; it does not yet use phase-dependent or next-subtask labels.

| Task type | Box entities |
|---|---|
| Place one vegetable | Source vegetable and destination plate |
| Arrange two vegetables | Both source vegetables and both destination plates |
| Clear a specified plate | Source vegetable, source plate, and collection box |
| Bus the table | All task vegetables and the collection box |

Each native-image box is transformed with exactly the same aspect-ratio-preserving resize and
left/top padding used for the corresponding VLM image. Coordinates are then normalized in the
512 x 512 policy image and quantized into 256 bins:

```text
location_token = round(clamp(coordinate / 511, 0, 1) * 255)
```

A target that is validly absent from a camera uses `<not_visible>`. An uncertain or incorrect
annotation uses `loss_weight=0` and is excluded from the box loss. These two cases must remain
distinct.

Entities are serialized in deterministic order:

1. Camera order follows the policy image features: wrist, then top.
2. Entities explicitly named in the instruction follow prompt order.
3. Entities implied by `all vegetables` follow a fixed dataset class order.
4. Source precedes destination when two entries otherwise have the same order.

An example teacher-forced target is:

```text
<grounding>
<camera> wrist
<source> tomato <box> <loc_071> <loc_106> <loc_138> <loc_201>
<destination> pink plate <not_visible>
</camera>
<camera> top
<source> tomato <box> <loc_083> <loc_119> <loc_141> <loc_188>
<destination> pink plate <box> <loc_176> <loc_074> <loc_239> <loc_146>
</camera>
</grounding>
<action> <fast_...> ... <action_end>
```

##### SmolVLM Token Vocabulary

The PaliGemma-specific reverse token mapping in the existing PI0-FAST processor must not be copied
unchanged. Action Flow uses SmolVLM2, so Action Flow V2.2 extends the SmolVLM tokenizer and LM
head with non-colliding tokens:

- 256 coordinate tokens: `<loc_000>` through `<loc_255>`.
- Structural tokens for grounding, cameras, roles, boxes, visibility, actions, and sequence ends.
- One dedicated `<fast_N>` language-model token for each FAST tokenizer symbol.

The FAST tokenizer still performs DCT, quantization, and BPE compression. A fixed one-to-one table
maps each FAST symbol to its added SmolVLM token ID. The extended tokenizer, mapping table, resized
input embeddings, and resized LM head must be saved with every checkpoint.

##### Autoregressive VLM Branch

The VLM is trained with teacher forcing on one autoregressive target sequence containing the box
span followed by the FAST-action span. Separate token masks compute independent mean losses:

```text
L_box  = mean cross-entropy over valid target-box tokens
L_FAST = mean cross-entropy over valid FAST-action tokens
```

The target-box and FAST losses are normalized independently so that a longer FAST sequence cannot
silently dominate the shorter grounding sequence. The prefix and target-token attention rules are:

| Query tokens | Allowed keys/values |
|---|---|
| Image, prompt, and state prefix | Valid prefix tokens only |
| Target-box tokens | Complete prefix and earlier/current autoregressive tokens |
| FAST tokens | Complete prefix, target-box tokens, and earlier/current FAST tokens |
| Padding tokens | Masked |

The prefix cannot attend to teacher-forced output tokens. This prevents ground-truth boxes and
future actions from leaking into the VLM representation consumed by the action expert.

##### Knowledge-Insulated Flow Branch

After the prefix pass, the model retains the VLM key/value tensors for every transformer layer.
The action expert receives detached copies:

```text
expert_prefix_k[layer] = vlm_prefix_k[layer].detach()
expert_prefix_v[layer] = vlm_prefix_v[layer].detach()
```

The noisy continuous-action suffix may attend to the detached prefix and to the action-expert
suffix according to the configured expert attention pattern. It must not attend to target-box or
FAST target tokens. Conversely, autoregressive target tokens must not attend to continuous-action
tokens.

The two-stream attention requirements are:

| Query stream | Prefix K/V | Autoregressive K/V | Expert K/V |
|---|---:|---:|---:|
| VLM prefix | Yes | No | No |
| Box/FAST autoregressive stream | Yes | Causal self-attention | No |
| Continuous action expert | Yes, detached | No | Expert attention |

Detaching only the final prefix output is not the implementation target. Gradient insulation must
cover every prefix key/value path used by every expert attention layer.

##### Training Loss and Parameter Ownership

The planned objective is:

```text
L_VLM    = lambda_box * L_box + lambda_FAST * L_FAST
L_expert = L_flow
L_total  = L_VLM + lambda_flow * L_expert
```

All three losses are reduced independently before weighting. Initial experiments will use
`lambda_box=1`, `lambda_FAST=1`, and `lambda_flow=1`; changing a weight must be documented in the
training configuration.

| Parameter group | Box CE | FAST CE | Flow loss |
|---|---:|---:|---:|
| Vision encoder and connector | Yes | Yes | No |
| VLM transformer | Yes | Yes | No |
| Token embeddings and LM head | Yes | Yes | No |
| State projection | Yes | Yes | No |
| Action expert transformer | No | No | Yes |
| Action input/output and timestep projections | No | No | Yes |

The total scalar loss may be backpropagated once, but the graph must enforce this ownership. A
flow-only backward pass must produce no gradients for any VLM-side parameter.

##### Inference

Standard rollout does not autoregressively decode target boxes or FAST actions. It performs:

```text
images + prompt + current state
-> VLM prefix and per-layer K/V cache
-> continuous action expert denoising
-> unnormalize relative actions
-> convert joint deltas to absolute RM65 commands
```

This preserves the fast continuous-action inference path. A separate diagnostic method may decode
target boxes for visualization and grounding evaluation, but those predicted tokens are not fed
back into the action expert in the first V2.2 experiment. Conditioning the expert on generated box
tokens would add autoregressive latency and error propagation and should be evaluated separately
only after the insulated baseline works.

#### Development Order

##### Stage 1: Define Configuration and Data Contracts

**Status:** Implemented on 2026-07-29.

1. Create the `action_flow_v2_2` policy package without modifying V2 or V2.1.
2. Define configuration fields for relative actions, tokenizer paths, token ranges, maximum box
   and FAST sequence lengths, loss weights, and knowledge insulation.
3. Define structured box-sidecar and model-batch keys with explicit shapes, dtypes, camera order,
   visibility, and validity semantics.
4. Add configuration validation for vocabulary size, camera count, action dimensions, sequence
   lengths, and incompatible options.

Completion criterion: configuration serialization and checkpoint reload reproduce the same token
IDs, feature metadata, and policy structure.

The implemented contract keeps the existing 49,280 SmolVLM token IDs unchanged, then allocates 12
structural tokens, 256 coordinate tokens, and 2,048 dedicated FAST tokens. The resulting extended
vocabulary size is 51,596. Model and processor construction are deliberately disabled until their
later development stages are complete; only configuration parsing, serialization, sidecar
metadata, feature validation, and batch-contract inspection are enabled in Stage 1.

##### Stage 2: Build and Inspect the Box Dataset

**Status:** Implemented on 2026-07-30.

1. Read separate SAM2 object tracks rather than union masks.
2. Convert each valid mask to an entity-specific bounding box for every camera frame.
3. Apply the exact 512 x 512 image geometry transform and quantize coordinates to 256 bins.
4. Save `not_visible` and invalid annotations separately.
5. Generate per-episode preview videos and a review report showing entity names, roles, boxes, and
   excluded labels.
6. Join structured box labels to the 203-episode robot dataset by `episode_index`, `frame_index`,
   and camera.

Completion criterion: sampled labels remain aligned with the resized policy images, preserve
entity identity through manipulation, and contain no camera-order mismatch.

The completed dataset is:

```text
runs/dataset/rm65_single_table_preparation_and_bussing_afv2_2_yoloe_boxes
```

| Item | Value |
|---|---|
| Robot dataset | `runs/dataset/rm65_single_table_preparation_and_bussing` |
| Reviewed entity tracks | `runs/sam/rm65_single_table_preparation_and_bussing_yoloe` |
| Reviewed entity definitions | `runs/yoloe/rm65_single_table_preparation_and_bussing` |
| Episodes | 203 |
| Frames | 92,545 |
| Camera order | `wrist`, `top` |
| Policy image size | 512 x 512 |
| Coordinate bins | 256 |
| Maximum entity slots per frame | 8 |
| Global entity vocabulary | 11 names |

The builder is
`examples/action_flow_v2/build_action_flow_v2_2_box_dataset.py`. It reads the compressed SAM2
run-length encoding directly, so it does not materialize all dense masks in memory. The original
robot rows are preserved and the following frame-aligned features are added:

| Feature | Shape | Meaning |
|---|---:|---|
| `observation.target_boxes` | `(2, 8, 4)` | Normalized `xyxy` boxes in padded policy-image coordinates |
| `observation.target_box_bins` | `(2, 8, 4)` | Box coordinates quantized to integers in `[0, 255]` |
| `observation.target_box_visibility` | `(2, 8)` | Visible or validly not visible in each camera |
| `observation.target_box_loss_weight` | `(2, 8)` | Zero excludes an invalid camera/entity annotation |
| `observation.target_entity_ids` | `(8,)` | Stable IDs in the dataset-wide entity vocabulary |
| `observation.target_entity_roles` | `(8,)` | Source, destination, or container role IDs |
| `observation.target_entity_valid` | `(8,)` | Valid prompt entities versus padded slots |

Long-form records are retained under `annotations/boxes/episode-*/camera-*.parquet` for debugging.
The review report is `review/index.html`, and 12 evenly distributed episodes have side-by-side
wrist/top preview videos under `review/videos`. The builder applies
`runs/yoloe/rm65_single_table_preparation_and_bussing/review/decisions.json` before resolving track
visibility. Validation found 1,382 camera/entity tracks: 1,365 contain visible masks and only the 17
targets explicitly reviewed as absent are supervised as `not_visible`. No track was marked invalid
or unexpectedly never visible. LeRobot loads all generated features with the expected shapes,
including the source file that contains episodes 36 through 59.

Because `runs/dataset` is stored on an exFAT volume, it does not support hard links or symbolic
links. The unchanged source videos are therefore copied into this dataset rather than linked.

##### Stage 3: Implement Relative-Action Processing

**Status:** Implemented on 2026-07-30.

1. Convert six joint targets relative to the current state while excluding the gripper.
2. Compute and save statistics from the converted joint deltas and absolute gripper values.
3. Normalize the shared relative action chunk before both FAST tokenization and flow matching.
4. Implement inverse normalization and relative-to-absolute conversion for rollout.

Completion criterion: preprocessing followed by postprocessing reconstructs the original absolute
action chunk within floating-point tolerance, including batched training and single-step rollout.

The processor is implemented in
`repos/lerobot/src/lerobot/policies/action_flow_v2_2/processor_action_flow_v2_2.py` and registered
with the LeRobot processor factory. Its action path is:

```text
training:
absolute future actions
-> subtract the current state from joint_1 through joint_6
-> keep gripper.pos absolute
-> normalize with relative-action statistics

rollout:
normalized policy output
-> unnormalize
-> add the current state to joint_1 through joint_6
-> keep gripper.pos absolute
-> send absolute RM65 commands
```

Both FAST tokenization and flow matching consume the same normalized tensor produced by this
pipeline. The raw current state is cached before state normalization, which prevents
normalized state values from being added to physical joint commands during rollout.

Relative statistics were prepared for the Stage 2 dataset with a 50-frame action chunk:

| Item | Value |
|---|---|
| Valid chunk starts | 82,598 |
| Future-action samples | 4,129,900 |
| Relative dimensions | `joint_1.pos` through `joint_6.pos` |
| Absolute dimension | `gripper.pos` |
| Installed statistics | `meta/stats.json` |
| Original statistics backup | `meta/stats.absolute_actions.json` |
| Transformation record | `meta/action_flow_v2_2_relative_stats.json` |

The preparation utility is
`examples/action_flow_v2/prepare_action_flow_v2_2_relative_stats.py`. It replaces only the action
statistics; state, image, and target-box metadata remain unchanged. Processor validation covers
the seven-dimensional RM65 action contract, six-relative/one-absolute mask, factory construction,
checkpoint save/reload, batched chunks, and inverse conversion. A real dataset chunk of shape
`(1, 50, 7)` reconstructed its original absolute actions with zero measured round-trip error.

##### Stage 4: Extend and Validate the Tokenizer

**Status:** Implemented on 2026-07-30.

1. Add structural, coordinate, and FAST tokens to the local SmolVLM tokenizer.
2. Resize the VLM input embeddings and LM head without changing existing token IDs.
3. Implement deterministic structured-box serialization and parsing.
4. Map FAST symbols to dedicated SmolVLM token IDs and implement the inverse mapping for
   inspection.
5. Save and reload all tokenizer artifacts from a policy checkpoint.

Completion criterion: box and FAST sequences round-trip exactly, no generated token ID exceeds the
resized vocabulary, and no task-language token collides with an action token.

The implementation is in
`repos/lerobot/src/lerobot/policies/action_flow_v2_2/tokenization_action_flow_v2_2.py`. The
unmodified SmolVLM tokenizer has 49,280 entries: 49,152 base language tokens followed by 128
existing multimodal/special-token entries. V2.2 appends tokens without replacing or reordering
those entries:

| Token group | Count | Inclusive ID range |
|---|---:|---:|
| Existing SmolVLM vocabulary | 49,280 | `0-49279` |
| Structural tokens | 12 | `49280-49291` |
| Coordinate tokens | 256 | `49292-49547` |
| Dedicated FAST tokens | 2,048 | `49548-51595` |

The final vocabulary size is 51,596. Every new string encodes as exactly one token, and the
manifest records a SHA-256 digest over the complete ordered extension. Loading fails if token
order, camera order, vocabulary size, or token-contract metadata differs from the policy
configuration.

Grounding serialization uses the configured wrist-then-top camera order and prompt-relative entity
slot order. Visible entities encode their role, name, `<box>`, and four coordinate tokens. Validly
absent entities use `<not_visible>`. Padded entities and annotations with zero loss weight are not
serialized. The parser reconstructs camera name, entity name, role, visibility, and coordinate
bins, while preserving the distinction between not-visible and excluded annotations.

The same normalized relative action chunk prepared in Stage 3 is passed to the 2,048-symbol FAST
tokenizer. FAST symbol `n` maps directly to VLM token
`49548 + n`; the inverse mapping is exact. FAST's conversion between continuous actions and its
discrete symbols remains intentionally lossy.

The generated checkpoint-ready artifact is:

```text
artifacts/tokenizers/action_flow_v2_2_smolvlm_fast
```

It contains the extended VLM tokenizer, the complete FAST processor and BPE tokenizer, the token
contract manifest, and a validation summary. The preparation command is
`examples/action_flow_v2/prepare_action_flow_v2_2_tokenizer.py`. Validation on a real 50-frame
dataset chunk produced 56 grounding tokens and 34 FAST symbols. Structured boxes, FAST-symbol
mapping, and checkpoint save/reload were exact; continuous FAST reconstruction had a mean absolute
error of 0.0116 in normalized action space. A full scan of all 92,545 frames found a maximum
grounding length of 76 tokens, leaving substantial margin within the configured 192-token budget.

The actual SmolVLM input embedding and LM-head tables were resized from `(49280, 960)` to
`(51596, 960)`. Sampled rows from the original vocabulary were unchanged after resizing.

##### Stage 5: Implement Joint Knowledge-Insulated Training

**Status:** Implemented on 2026-07-30.

The VLM and action expert are optimized together from the beginning. There is no separate VLM-only
training stage followed by an expert-only stage.

1. Serialize normalized relative action chunks and structured boxes in the preprocessor.
2. Run one teacher-forced VLM pass over the multimodal context followed by grounding and FAST
   targets.
3. Use prefix-LM attention for image, prompt, and state tokens, then causal attention for each
   autoregressive target token. Context queries cannot attend to teacher targets.
4. Compute independently normalized `box_ce_loss` and `fast_ce_loss`.
5. Keep only the context portion of every layer's VLM key/value cache. Remove grounding and FAST
   target K/V entries, then detach the retained context K/V tensors.
6. Condition the continuous flow expert on that detached cache and compute `flow_loss` from the
   same normalized relative action chunk used by FAST.
7. Combine all three per-sample losses and perform one backward pass:

```text
loss = box_loss_weight * box_ce_loss
     + fast_loss_weight * fast_ce_loss
     + flow_loss_weight * flow_loss
```

This trains both branches at every optimizer step while preserving parameter ownership. Box and
FAST losses update the vision-language branch; flow loss updates the action expert but cannot cross
the detached cache boundary into the VLM.

The implementation is in
`repos/lerobot/src/lerobot/policies/action_flow_v2_2/modeling_action_flow_v2_2.py`. The complete
preprocessor now performs:

```text
absolute actions
-> relative joint conversion
-> shared normalization
-> grounding and FAST target serialization
-> device transfer
```

A real batch with shape `(1, 50, 7)` completed preprocessing, joint forward, and backward. All three
losses were finite, and both the VLM and expert received nonzero gradients from the combined
objective.

The gradient-isolation checks pass: flow-only backward produces zero VLM gradients and nonzero
expert gradients, while box/FAST-only backward produces nonzero VLM gradients and zero expert
gradients.

##### Stage 6: Implement Inference and Diagnostics

1. Use prefix-cache plus continuous denoising for normal rollout.
2. Restore absolute RM65 commands after unnormalization.
3. Retain action queue, temporal ensemble, and RTC compatibility only where their behavior is
   explicitly tested.
4. Add optional target-box decoding and camera overlays without requiring it for action inference.

Completion criterion: repeated inference returns finite actions with the correct RM65 dimensions,
the gripper remains absolute, and diagnostic box decoding can be enabled without changing the
continuous action result.

##### Stage 7: Validate Checkpoints and Run a Small-Set Overfit

1. Save the extended VLM tokenizer and FAST tokenizer inside every policy checkpoint.
2. Reload the policy and processors without depending on the original tokenizer artifact path.
3. Overfit 4-8 short episodes with all three losses enabled.
4. Verify that box and FAST losses decrease and that flow predictions fit the same action chunks.
5. Decode teacher targets and sampled VLM outputs to check box IoU, visibility accuracy, FAST
   reconstruction error, and truncation.

Completion criterion: a moved checkpoint reloads locally, reproduces finite action inference, and
the small-set run can fit all three objectives without violating gradient insulation.

##### Stage 8: Run Controlled Ablations

Train and compare the following policies on identical dataset splits and rollout scenes:

| Experiment | Relative actions | FAST CE | Box CE | Flow-to-VLM gradient |
|---|---:|---:|---:|---:|
| Relative flow baseline | Yes | No | No | Enabled |
| KI + FAST | Yes | Yes | No | Blocked |
| KI + FAST + boxes | Yes | Yes | Yes | Blocked |

Action Flow V2.1 remains the dense-segmentation comparison. Evaluation must separately report:

- Task completion and language-following success.
- Correct source and destination selection.
- Grasp success, hold success, and placement success.
- Box visibility accuracy and box IoU per camera.
- FAST reconstruction error and token truncation rate.
- Continuous action smoothness and rollout latency.

The first full training run should begin only after the small-set overfit test, tokenizer
round-trip test, relative-action round-trip test, attention-mask test, and gradient-insulation test
all pass.
