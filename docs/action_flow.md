# Action Flow Training Log

*Last updated: July 17, 2026*

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

Action Flow V1 uses a ResNet observation encoder and does not encode the task prompt. It can learn smooth
continuous trajectories, but the evaluation shows that it often follows fixed positions and demonstrated action
sequences instead of selecting the target from the current image.

Action Flow V2 should preserve the V1 flow-matching action decoder while replacing the observation path with a
shared pretrained vision-language model (VLM). The normal VLM output remains the condition consumed by the action
expert. A lightweight auxiliary grounding head predicts a prompt-conditioned target heatmap from the same shared
features.

The first experiment is intentionally narrower than PI0.5. It does not generate semantic subtasks, require
destination labels, or feed a hard detector output into the controller. It tests one question:

> Does auxiliary target-heatmap supervision improve the shared VLM representation and the resulting actions?

### First Experiment

The first architecture is:

~~~text
camera images + task prompt
            |
            v
       shared VLM
            |
            +------------------------------+
            |                              |
            v                              v
  shared condition tokens        auxiliary heatmap head
            |                              |
robot state + noisy actions               v
            |                    target heatmap (optional
            v                       during inference)
    flow action expert
            |
            v
 continuous action chunk
~~~

The action expert always receives the normal shared VLM tokens. In the first experiment, the predicted heatmap
is not pooled, thresholded, or passed back to the action expert. This keeps the action interface unchanged and
isolates the effect of auxiliary spatial supervision.

A pretrained PaliGemma-style VLM is a practical first backbone because the project already uses this architecture
for PI0 and has local model assets. Multiple cameras can share the same image encoder. Camera-identity embeddings
must be added before the tokens are concatenated so the action expert can distinguish overview and wrist views.

### Shared VLM Output

The shared VLM must produce image tokens that retain their camera and patch-grid correspondence. The action expert
receives:

~~~text
[VLM image tokens, VLM language tokens, robot-state token]
~~~

The noisy action chunk attends to these condition tokens while predicting the flow vector field. This is similar
to the broad VLM-plus-action-expert structure used by PI0 and PI0.5, but Action Flow V2 retains the existing V1
flow decoder so that the perception change can be evaluated independently.

The heatmap head should read the prompt-conditioned image-token outputs. If the selected VLM exposes image tokens
before language fusion, the grounding head must first apply a small cross-attention layer from the image tokens to
the language tokens. Otherwise, the head could learn a generic objectness map that does not change with the
prompt.

### Auxiliary Target-Heatmap Head

The grounding head should remain lightweight:

1. Select the VLM tokens belonging to each camera.
2. Project them to a smaller grounding dimension.
3. Apply a small MLP or one cross-attention block.
4. Produce one scalar logit per image patch.
5. Reshape the logits to the original patch grid.
6. Optionally upsample the grid for visualization.

For B samples, C supervised cameras, and a VLM patch grid of H_p by W_p, the output shape is:

~~~text
target_heatmap_logits: (B, C, H_p, W_p)
target_heatmap_valid:  (B, C)
~~~

The validity mask indicates whether a reliable target label exists in that camera. Invalid or fully occluded
targets must not contribute to the heatmap loss.

The first version should predict only the requested target heatmap. It should not require:

- A destination heatmap.
- A source-versus-destination class.
- A grasp point or gripper orientation.
- A manipulation-phase class.
- An arm-activity class.

These can be introduced later only when a measured failure requires them.

### Target Definition

The heatmap represents the spatial region that the prompt asks the policy to attend to. It is an object-location
label, not necessarily a grasp point.

| Prompt example | Initial heatmap target |
|---|---|
| Place the tomato on the pink plate. | Tomato |
| Transfer the bottle to the right arm. | Bottle |
| Clear the pink plate. | Pink plate and its relevant region |
| Put all vegetables into the box. | Ambiguous; exclude from the first experiment |

A handover does not require a destination label in the first version: the bottle remains the target, while the
prompt and full VLM context tell the action expert that a handover is required.

Sequential prompts, such as arranging two vegetables, require the target label to change over time. They should
not be included until the single-target experiment works reliably.

### Grounding Labels

The initial grounding dataset should use the Place one vegetable subset of
rm65_single_table_preparation_and_bussing. These episodes have one unambiguous manipulated target throughout
most of the trajectory.

The annotation sidecar should contain compact labels rather than precomputed dense heatmaps:

~~~text
episode_index
frame_index
camera
target_query
target_visible
target_point_u
target_point_v
target_box_x1
target_box_y1
target_box_x2
target_box_y2
annotation_confidence
~~~

Coordinates should be normalized to [0, 1]. The data loader can generate a Gaussian heatmap on the VLM patch
grid from a point, box center, or mask center. The current Grounding DINO probe can provide initial boxes.
SAM/SAM2 is optional when a mask or more stable tracking is needed.

Automatic labels must be reviewed as video overlays. Frames with an incorrect target, severe occlusion, or
uncertain tracking should be corrected or marked invalid rather than used as positive supervision.

### Training Objective

The first training objective is:

~~~text
L_total = L_flow + lambda_heatmap * L_target_heatmap
~~~

Where:

- L_flow is the existing Action Flow V1 flow-matching loss.
- L_target_heatmap is focal BCE or KL divergence between the predicted and Gaussian target heatmaps.
- lambda_heatmap controls the auxiliary supervision strength.

Both losses should be averaged before weighting, and they must be logged independently. Start with a conservative
heatmap weight, such as 0.1, then compare gradient norms and heatmap accuracy before increasing it. The heatmap
loss should not dominate the flow loss.

### Gradient Requirement

The auxiliary head improves action generation only if its loss changes a representation shared with the action
expert.

If the complete VLM is frozen and the heatmap head is attached directly to fixed VLM tokens, the heatmap loss
trains only the new head. This is still a useful diagnostic baseline, but it cannot make the VLM representation
more spatially grounded.

At least one shared component must therefore be trainable:

- LoRA adapters inside the VLM.
- A trainable projector shared by the action expert and heatmap head.
- Selected final VLM or vision layers.

### VLM Adaptation Experiments

All three adaptation strategies should be tested under the same dataset split, initialization, action decoder,
batch size, number of steps, heatmap loss weight, and random seeds.

#### Frozen Control

Freeze the complete VLM and train only the action expert and heatmap head. This control measures how much target
location is already linearly recoverable from the pretrained features. It does not test representation
adaptation.

#### Shared Projector

Keep the VLM frozen and add a trainable projector after its output. Both the action expert and heatmap head must
consume the projected tokens:

~~~text
frozen VLM tokens -> shared trainable projector -> action expert
                                             \-> heatmap head
~~~

This is the smallest and least expensive adaptation strategy. Although it does not modify the VLM backbone, it
allows the heatmap loss to reshape the exact representation consumed by the action expert.

#### LoRA

Keep the base VLM weights frozen and add LoRA adapters to selected attention projections. The first LoRA
experiment should target the visual and multimodal/language attention q_proj and v_proj layers, not only the
action expert.

The action expert, auxiliary head, and LoRA weights are trainable. Initial LoRA ranks of 8 or 16 are reasonable
comparison points. The LoRA learning rate can be lower than the new head's learning rate.

#### Partial Unfreezing

Unfreeze only the last one or two vision blocks and, if necessary, the last one or two multimodal language
blocks. Keep the earlier pretrained layers frozen.

Use a backbone learning rate substantially smaller than the action expert and heatmap head learning rates.
Gradient checkpointing should be enabled. This strategy has the highest memory cost and the greatest risk of
overfitting or damaging pretrained semantics.

### Controlled Experiment Matrix

| Experiment | Shared representation training | Heatmap loss | Heatmap used by action expert |
|---|---|---:|---:|
| V1 baseline | ResNet trained by action loss | No | No |
| V2 VLM baseline | Frozen VLM | No | No |
| V2 frozen diagnostic | Frozen VLM | Yes | No |
| V2 shared projector | Shared projector | Yes | No |
| V2 LoRA | VLM LoRA adapters | Yes | No |
| V2 partial unfreezing | Final VLM/vision blocks | Yes | No |

The frozen diagnostic separates heatmap readout quality from representation learning. The comparison between the
V2 VLM baseline and the three adaptation variants measures whether heatmap supervision improves action quality.

A later experiment may construct a heatmap-weighted target token and feed it to the action expert. That is not
part of the first experiment because it would mix two effects:

1. Improving shared features with an auxiliary loss.
2. Explicitly conditioning actions on the predicted heatmap.

### Dataset Splits And Counterfactual Tests

Dataset splits must be made by episode and object layout, not by individual frames. Otherwise, nearly identical
neighboring frames can appear in both training and validation.

The initial split should preserve unseen combinations of:

- Target vegetable.
- Target position.
- Plate position and color.
- Distractor layout.

Grounding-only counterfactual examples can reuse the same image with different target queries and corresponding
heatmaps. They do not require additional robot actions. These pairs are important because they force the
heatmap to change with the prompt instead of learning a fixed workspace prior.

### Staged Implementation Plan

#### Stage 1: Label Audit

1. Annotate a small set of Place one vegetable episodes.
2. Generate Gaussian target heatmaps on the VLM patch grid.
3. Overlay the labels on the top-camera video.
4. Correct false detections and mark occluded frames invalid.

#### Stage 2: VLM Action Baseline

1. Add the pretrained VLM observation path.
2. Feed its normal output tokens to the V1 action expert.
3. Train without a heatmap loss.
4. Confirm that the policy and rollout pipeline work before adding auxiliary supervision.

#### Stage 3: Heatmap Diagnostic

1. Add the auxiliary heatmap head.
2. Keep the VLM frozen.
3. Train the action expert and heatmap head.
4. Evaluate whether pretrained VLM tokens can localize the prompted target.

#### Stage 4: Representation Adaptation

Train and compare the shared-projector, LoRA, and partial-unfreezing variants. Do not change the action decoder or
rollout parameters between variants.

#### Stage 5: Optional Explicit Grounding

Only if the best auxiliary-supervision policy predicts accurate heatmaps but action target selection remains poor,
feed a heatmap-pooled target token to the action expert and compare it with the auxiliary-only policy.

### Inference And Rollout

The heatmap head must be optional during inference:

- Disabled: skip the final heatmap projection and run the normal VLM-to-action path.
- Enabled: compute the heatmap and record it for visualization without changing the action-expert input.

For a fixed observation, random seed, and action state, enabling heatmap output should not change the generated
action. This should be covered by a unit test.

When enabled, rollout should record:

- Target heatmap overlaid on the supervised camera.
- Target query and visibility mask.
- Heatmap peak and confidence.
- Raw action chunk.
- RTC-guided action chunk.

RTC remains independent of grounding. It addresses consistency between action chunks; it cannot correct a
heatmap that selects the wrong object.

### Evaluation

#### Offline Grounding Metrics

- Heatmap loss on held-out episodes.
- Peak-point error in pixels.
- Percentage of heatmap peaks inside the target box or mask.
- Target-selection accuracy when distractors are present.
- Counterfactual prompt consistency.
- Performance on unseen object layouts.

#### Real-Robot Metrics

- Correct-target approach rate.
- Gripper-alignment accuracy.
- Grasp success rate.
- Placement success rate.
- Complete task success rate.
- Action discontinuity and RTC correction magnitude.

Failures should be separated into:

1. Wrong heatmap and wrong action target.
2. Correct heatmap but wrong action target.
3. Correct target approach but inaccurate grasp.
4. Successful grasp but failed later trajectory.

This separation determines whether the next change belongs in grounding, action conditioning, grasp supervision,
or rollout execution.

### Proposed Configuration

The first implementation should expose configuration fields equivalent to:

~~~text
vlm_model_name
vlm_adaptation_mode            # frozen, shared_projector, lora, partial
freeze_vlm
shared_projector_dim
lora_rank
lora_alpha
num_unfrozen_vision_layers
num_unfrozen_language_layers
use_target_heatmap_head
target_heatmap_loss_weight
compute_target_heatmap_at_inference
chunk_size
num_inference_steps
rtc_config
~~~

The final field names should follow LeRobot's policy configuration conventions. Invalid combinations should fail
early; for example, vlm_adaptation_mode=lora must require LoRA support and a pretrained VLM path.

### Relationship To PI0.5

Action Flow V2 and PI0.5 share a broad VLM-plus-action-expert structure. PI0.5 additionally uses heterogeneous
web and robot co-training, FAST action-token pretraining, autoregressive semantic-subtask prediction, and
hierarchical inference before continuous flow actions.

The first Action Flow V2 experiment does not reproduce those components. Its contribution is narrower and
inspectable: use an auxiliary prompt-conditioned heatmap to improve or diagnose the shared representation while
leaving the normal VLM-to-action path unchanged.

### Initial Success Criteria

The first experiment is successful when:

1. The heatmap selects the prompted target instead of a fixed workspace location.
2. Changing the prompt for the same image changes the heatmap.
3. Moving the target while keeping the prompt fixed moves the heatmap.
4. At least one adaptation strategy improves held-out heatmap accuracy over the frozen diagnostic.
5. The same strategy improves correct-target approach or grasp success over the V2 VLM baseline.
6. Disabling heatmap output at inference does not change actions.
7. Any action improvement is reported separately from RTC or rollout-hyperparameter changes.
