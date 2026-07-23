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

### Motivation

Action Flow V1 uses a ResNet observation encoder and does not encode the task prompt. It can learn smooth continuous trajectories, but the evaluation shows that it often follows fixed positions and demonstrated action sequences instead of selecting the target from the current image.

Action Flow V2 replaces the V1 ResNet and action decoder with a PI0.5-style coupled VLM and action expert. The initial implementation uses the pretrained SmolVLM2/SmolVLA representation because it is smaller and practical to fine-tune locally. The task prompt and camera images are processed together, and a lightweight auxiliary head predicts a prompt-conditioned task-relevance heatmap from the same shared features.

The first experiment asks one controlled question:

> Does explicit task-relevance supervision improve visual grounding and the resulting continuous actions?

The heatmap is auxiliary supervision. It is not a detector result passed directly into the controller, and it is not a replacement for the normal VLM representation.

### Policy Structure

The implemented policy structure is:

~~~text
camera images + task prompt
            |
            v
       shared VLM
            |
            +------------------------------+
            |                              |
            v                              v
  shared condition tokens     task-relevance heatmap head
            |                              |
robot state + noisy actions               v
            |                  relevance heatmap (optional
            v                       during inference)
    flow action expert
            |
            v
 continuous action chunk
~~~

The action expert always receives the normal prompt-conditioned VLM tokens. In the first implementation, the predicted heatmap is not pooled, thresholded, or passed back to the action expert. This keeps the action interface unchanged and isolates the effect of representation learning from explicit heatmap conditioning.

Multiple cameras share the VLM image encoder. Camera identity must be retained when image tokens are concatenated, so the action expert and relevance head can distinguish observations from different viewpoints.

### Shared VLM Output

The shared VLM must expose image tokens while preserving camera and patch-grid correspondence. The condition used by the action expert is:

~~~text
[VLM image tokens, VLM language tokens, robot-state token]
~~~

The noisy action chunk attends to these condition tokens while predicting the flow vector field. As in PI0.5, the multimodal prefix and noisy-action suffix are processed by coupled transformer streams, and the prefix key/value states are cached during iterative inference. Camera embeddings are added before the shared transformer so both the action expert and relevance head retain viewpoint identity.

The relevance head reads the final prompt-conditioned image tokens after multimodal transformer processing. It therefore receives language context directly instead of learning a generic objectness map from isolated vision features.

### Task-Relevance Head

The relevance head remains lightweight:

1. Select the VLM tokens belonging to each camera.
2. Project them to a smaller grounding dimension.
3. Apply a small MLP or cross-attention block.
4. Produce one scalar logit per image patch.
5. Reshape the logits to the original patch grid.
6. Optionally upsample the grid for visualization.

For B samples, C supervised cameras, and a VLM patch grid of H_p by W_p, the output shape is:

~~~text
relevance_heatmap_logits: (B, C, H_p, W_p)
relevance_loss_weight:    (B, C)
~~~

An all-zero relevance mask is a valid negative label and teaches the policy that no relevant entity is visible in that camera. `relevance_loss_weight=0` is reserved for frames whose annotations are uncertain or intentionally excluded. Visibility and annotation validity must not be represented by the same value.

### Relevance Definition

The training target is a binary spatial mask containing every entity needed to interpret the prompt. Depending on the instruction, this can include manipulated objects, destination regions, containers, or multiple relevant objects. It is an attention target, not a grasp point or manipulation trajectory.

The first version deliberately does not predict:

- Grasp points or gripper orientations.
- Source-versus-destination classes.
- Manipulation phases.
- Arm-activity classes.
- Ordered semantic subtasks.

Object IDs, names, roles, and task steps are retained in the annotation sidecar for inspection and future experiments, but the first heatmap target is the union of the relevant per-object masks.

For multi-step prompts, the first experiment highlights all entities named by the instruction for the complete episode. It does not change the target mask after each subtask. Phase-dependent relevance can be added later after the static prompt-conditioned experiment is validated.

### Dataset Configuration

Grounding semantics are dataset-dependent, while video decoding, detector inference, ranking, review artifacts, and mask serialization are shared. Each dataset is represented by a `GroundingDatasetConfig` containing:

- A configuration name and default dataset root.
- Camera names.
- Detector query aliases.
- A task-prompt parser that produces deterministic entity specifications.
- Grounding thresholds.
- Optional camera-specific entity restrictions.

Each entity specification records its object ID, semantic name, role, detector query, task step, cardinality, and candidate-order rule. Multi-instance entities can reserve multiple deterministic IDs. The same object ID retains the same meaning across the camera views in an episode.

Adding another dataset requires a task parser and one registered dataclass configuration. Model execution and artifact generation do not need to be copied.

### Dataset Annotation Pipeline

#### Grounding Initialization

`examples/action_flow_v2/annotate_grounding.py` performs the initialization stage:

1. Read the episode prompt and build its entity specifications from the selected dataset configuration.
2. Sample frames from each configured camera.
3. Run Grounding DINO with dataset-specific query aliases.
4. Remove duplicate proposals and order multiple instances deterministically.
5. Rank candidates using confidence, visible area, border clearance, sharpness, and episode progress.
6. Save selected detections, alternative candidates, contact sheets, and an HTML inspection index.

Grounding DINO only provides initialization proposals. It is not treated as ground truth. Similar objects, occlusion, unusual viewpoints, and visually identical workspace regions can still produce incorrect boxes.

#### Grounding Review

Every episode-camera result must be inspected before segmentation. The review can:

- Accept the selected candidate.
- Select another ranked candidate.
- Mark an entity as not visible in that camera.
- Require manual initialization at a specified frame.

The review decisions are stored separately from generated detector artifacts, so annotation can resume without rerunning accepted results.

#### SAM2 Tracking And Correction

`examples/action_flow_v2/annotate_sam2.py` imports the reviewed boxes as SAM2 prompts. It preserves each entity as a separate object track, propagates masks through the episode, and provides a local UI for adding positive or negative clicks, clearing prompts, setting object bounds, and excluding uncertain frames.

To control memory use, SAM2 can track a configurable subset of video frames. Intermediate frames reuse the mask from the nearest tracked frame, while explicit prompt frames and object bounds are always retained.

Batch mode processes all resolved episode-camera pairs and skips unresolved pairs instead of silently producing incorrect labels. Processing is resumable, and its report records completed, skipped, unresolved, and failed pairs.

#### Saved Annotation Data

Per-object tracks are saved under:

~~~text
runs/sam/<dataset>/data/episode-<index>/camera-<camera>.parquet
~~~

Each row includes `episode_index`, `frame_index`, `camera`, `object_id`, `object_label`, image dimensions, and a compressed binary mask. Keeping separate tracks allows later experiments to recover entity-specific targets.

The first training target is a merged union mask saved under:

~~~text
runs/dataset/<annotation-dataset>/data/episode-<index>/camera-<camera>.parquet
~~~

Each merged row contains the episode, frame, camera, image dimensions, compressed relevance mask, and `loss_weight`. Masks use compressed run-length encoding. The training loader decodes them and resizes them to the VLM patch grid.

Preview videos with mask overlays must be reviewed before training. Incorrect tracks should be corrected; uncertain frames should use `loss_weight=0`, not an invented positive mask.

### Training Objective

The first training objective is:

~~~text
L_total = L_flow + lambda_relevance * L_relevance
~~~

Where:

- `L_flow` is the SmolVLA-style action-expert flow-matching loss.
- `L_relevance` compares predicted patch logits with the resized binary relevance masks.
- `lambda_relevance` controls the auxiliary supervision strength.

The initial relevance objective should combine `BCEWithLogitsLoss` with a Dice term so that small relevant regions are not overwhelmed by background patches. The loss is computed independently per camera, multiplied by the saved `loss_weight`, and normalized by the number of valid camera frames. All-zero masks remain valid training targets.

Both losses must be logged independently. Start with a conservative `lambda_relevance`, such as `0.1`, then compare loss scales, gradient norms, mask accuracy, and action quality before increasing it. The auxiliary loss must not dominate flow matching.

### Gradient Requirement

The auxiliary head improves action generation only if its loss changes a representation shared with the action expert.

If the complete VLM is frozen and the relevance head is attached directly to fixed VLM tokens, the relevance loss trains only the new head. This is a useful diagnostic baseline, but it cannot make the shared VLM representation more spatially grounded.

At least one shared component must therefore be trainable:

- LoRA adapters inside the VLM.
- A trainable projector shared by the action expert and relevance head.
- Selected final VLM or vision layers.

### Implementation

The policy is registered as `action_flow_v2` under `repos/lerobot/src/lerobot/policies/action_flow_v2`. Before training, `examples/action_flow_v2/merge_segmentation_dataset.py` joins the frame-aligned SAM2 union masks to the robot data using `episode_index` and `frame_index`. The resulting dataset is a self-contained LeRobot dataset with model-ready segmentation labels.

The relevant policy settings are:

| Parameter | Role |
| --- | --- |
| `relevance_loss_weight` | Weight of the complete auxiliary relevance loss. |
| `relevance_bce_weight` | Weight of the per-patch binary cross-entropy term. |
| `relevance_dice_weight` | Weight of the Dice term. |
| `relevance_target_size` | Patch-grid target size; `(8, 8)` matches the default 512-pixel SmolVLM input. |

The merged dataset stores `observation.relevance_mask` with shape `(num_cameras, 8, 8)` and `observation.relevance_loss_weight` with shape `(num_cameras,)`. They pass through the standard SmolVLA processor but are excluded from policy input features and state normalization. Initialize the policy weights from `smolvla_base` by setting `policy.type=action_flow_v2` and `policy.pretrained_path=<smolvla_base_path>`.

### Training Log 1 of Action Flow V2 (Planned)

#### Objective

The first Action Flow V2 experiment uses the single-arm table-preparation and table-bussing task. The policy must use the language prompt to select the relevant vegetables, plates, or collection box instead of reproducing a fixed object-position trajectory.

The experiment has two connected objectives:

1. Predict continuous RM65 joint and gripper action chunks with the SmolVLA-style flow action expert.
2. Predict a task-relevance heatmap for the top and wrist cameras from the shared prompt-conditioned VLM tokens.

The relevance target contains the union of every entity needed to interpret the complete prompt. For example, `Place the tomato on the pink plate.` highlights both the tomato and pink plate. A multi-step arrangement prompt highlights all requested vegetables and plates throughout the episode. This first experiment does not use phase-dependent masks, source/destination classes, or predicted heatmaps as direct action-expert inputs.

The main question is whether the auxiliary relevance loss improves target recognition and action accuracy. A proper comparison should train two policies with otherwise identical settings:

- Primary policy: `policy.relevance_loss_weight=0.1`.
- Ablation policy: `policy.relevance_loss_weight=0.0`.

Evaluation should include familiar task combinations, unseen object layouts, arrange-two prompts with garlic or cucumber first, and four-vegetable table bussing. The latter two conditions were intentionally excluded from the demonstrations and test prompt-conditioned generalization.

#### Dataset

Base dataset:

```text
runs/dataset/rm65_single_table_preparation_and_bussing
```

The cleaned dataset contains 203 successful episodes, 92,545 frames, and 52 prompt variants recorded at 30 FPS.
Each frame contains:

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

The reviewed, full-resolution SAM2 annotations remain available separately:

```text
runs/dataset/rm65_single_table_preparation_and_bussing_afv2
```

This source contains one Parquet file per episode and camera, for 406 files in total. Each row stores `episode_index`, `frame_index`, `camera`, the compressed union mask, and `loss_weight`. The SAM2 tracks were generated every three frames and propagated to intermediate frames using the nearest tracked mask.

For training, these labels are merged with the robot dataset and converted to the default 8 x 8 SmolVLM patch grid:

```text
runs/dataset/rm65_single_table_preparation_and_bussing_afv2_segmentation
```

The explicit `_segmentation` suffix distinguishes this dataset from future variants containing detection boxes or other spatial supervision.

All-zero masks remain valid negative labels when no relevant entity is visible. Only uncertain or intentionally excluded labels use `loss_weight=0`.

#### Training Configuration

Base policy:

```text
models/smolvla_base
```

Local VLM:

```text
models/SmolVLM2-500M-Video-Instruct
```

Suggested output policy:

```text
rm65_action_flow_v2_single_table_preparation_and_bussing_episodes_203_steps_40000_batch_8_chunk_50_relevance_0.1
```

| Parameter | Suggested value | Action Flow V2 default |
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
> `policy.train_expert_only=false`, `policy.load_vlm_weights=true`, `policy.relevance_loss_weight=0.1`,
> `policy.relevance_bce_weight=1.0`, `policy.relevance_dice_weight=1.0`, and
> `policy.relevance_target_size=[8, 8]`. Use BF16 mixed precision when the GPU supports it. Two GPU processes
> produce a global batch size of 16. Start with 40,000 steps to validate both action and relevance learning;
> continue toward 80,000 steps only if validation rollouts and held-out heatmaps are still improving.

The complete VLM and action expert are trainable because the relevance objective must modify the representation shared with action prediction. The lower learning rate of `3e-5` reduces the risk of damaging the pretrained VLM while the newly initialized camera embeddings and relevance head learn the RM65 workspace.

During training, log at least:

- `flow_loss`
- `relevance_loss`
- `relevance_bce_loss`
- `relevance_dice_loss`
- `relevance_valid_camera_fraction`

Training should be stopped and inspected if the relevance loss dominates the flow loss after multiplication by `policy.relevance_loss_weight`, or if heatmap quality improves while rollout action quality degrades.
