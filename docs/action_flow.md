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

#### Training

Similar to [ACT Training Log 4 of Objective 2](act.md#training-log-4-of-objective-2), I train `Action Flow V1` with different `chunk_size` values: 20, 30, and 40. Each policy is trained for 60,000 steps with batch size 8.

#### Evaluation

In this rollout evaluation, I enable Real-Time Chunking (RTC) with `RTC_QUEUE_THRESHOLD=10`, `RTC_EXECUTION_HORIZON=10`, `RTC_MAX_GUIDANCE_WEIGHT=10` and `RTC_PREFIX_ATTENTION_SCHEDULE="LINEAR"`. RTC is more suitable than ACT-style temporal ensembling for this flow-matching policy because it explicitly reuses the unexecuted part of the previous chunk during the next denoising process.

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

`Action Flow V1` improves action generation by replacing direct action regression with flow matching, but
its observation encoder is still close to ACT. It encodes camera images with a ResNet and trains the visual
features only through the final action-flow loss. More importantly, the current V1 model does not encode the
dataset task prompt. The prompt passed to `lerobot-rollout` is therefore useful for dataset metadata, but it
cannot change the actions predicted by the policy.

This explains an important limitation observed in Objective 1: the policy can learn a correlation between a
scene and a demonstrated trajectory, but it is not explicitly required to determine:

- Which object is referred to by the instruction.
- Whether that object is visible or absent.
- Where the object should be grasped.
- Where the object should be placed.
- Which arm should be active.
- Which stage of the manipulation task is currently being executed.

Changing ResNet to a stronger visual backbone may improve image features, but it does not by itself guarantee
that the policy uses the prompt or localizes the target. `Action Flow V2` should introduce an explicit,
inspectable grounding representation between perception and action generation.

### Design Goal

The main goal of `Action Flow V2` is to build a prompt-conditioned flow policy that first forms an explicit
representation of the requested target and destination, then conditions continuous action generation on that
representation.

The intended behavior is:

```text
instruction + images
        |
        v
identify target, destination, visibility, and manipulation phase
        |
        v
generate a continuous action chunk with flow matching
```

"Recognize first" does not need to mean that the model generates a discrete text answer and then starts robot
control. A hard discrete boundary would prevent end-to-end gradients and make every detector error fatal.
Instead, V2 should predict supervised spatial heatmaps and grounded visual tokens. These outputs can be
visualized and evaluated independently while remaining differentiable inputs to the action decoder.

The first V2 implementation is not intended to reproduce a complete general-purpose VLA such as PI0. It should
remain small enough to train and debug on the available RM65 datasets while directly testing whether explicit
visual grounding improves conditional behavior and grasp accuracy.

### Proposed Architecture

```text
                                    +-----------------------+
task prompt ----------------------->| Text encoder          |
                                    +-----------+-----------+
                                                |
                                                | prompt tokens
                                                v
overview camera images --> visual features --> Grounding module
                                                |
                         +----------------------+----------------------+
                         |                      |                      |
                         v                      v                      v
                 target heatmap        destination heatmap    presence/phase logits
                         |                      |                      |
                         +-----------+----------+----------------------+
                                     |
                                     v
                              grounded tokens
                                     |
wrist images + robot state ----------+--------> Action Flow decoder
                                                |
                                                v
                                      continuous action chunk
```

The policy should contain four main components:

1. A pretrained visual-language feature extractor.
2. A prompt-conditioned grounding module.
3. A grounded condition encoder for camera, robot-state, and task information.
4. The existing flow-matching action decoder, initially kept as close to V1 as possible.

Keeping the action decoder stable makes the experiment easier to interpret. If V2 improves target selection or
empty-side behavior, the improvement can be attributed mainly to perception and grounding rather than to an
unrelated change in action generation.

#### Visual And Language Encoders

The recommended first implementation uses complementary pretrained encoders:

- DINOv2 provides dense spatial image features that are useful for localization and geometry.
- SigLIP provides aligned image and text features that are useful for matching the prompt to an object.
- A small learned projector maps both feature types to `dim_model`.

This follows the general motivation of the fused DINOv2 and SigLIP representation used by
[OpenVLA](https://arxiv.org/abs/2406.09246), while retaining a continuous flow-matching action head.

The encoders should initially be frozen. The grounding projections and action model can then learn without
destroying the pretrained semantic representation. After the grounding outputs become reliable, the last one or
two visual blocks can optionally be unfrozen with a learning rate substantially lower than the action decoder's
learning rate.

PaliGemma is an alternative when broader language reasoning is required. However, combining PaliGemma with a
flow-matching action expert would make V2 structurally close to [PI0](https://arxiv.org/abs/2410.24164). Because
PI0 is already available in LeRobot and is being evaluated separately, DINOv2 plus SigLIP is a more useful first
V2 experiment: it isolates explicit grounding without duplicating the complete PI0 architecture.

For multiple cameras, all cameras should share the same visual backbone, but each camera's tokens must receive a
learned camera-identity embedding. The overview cameras should primarily support target and destination
localization. Wrist-camera features should provide close-range information during approach, grasp, and transfer.

#### Prompt-Conditioned Grounding Module

The grounding module should use learned queries that cross-attend to prompt tokens and dense image tokens. The
initial query set should include:

- `source_query`: the object requested by the prompt.
- `destination_query`: the requested placement area or container.
- `left_arm_query`: whether the left arm has a valid target and should move.
- `right_arm_query`: whether the right arm has a valid target and should move.
- `phase_query`: the current manipulation stage.

For a single-arm dataset, only one arm-activity query is required. Keeping arm activity explicit is particularly
important for the V1 failure where an arm moves toward an empty position.

The grounding heads should predict:

| Output | Meaning | Suggested representation |
| --- | --- | --- |
| Target presence | Whether the requested source object is visible | Binary logit per relevant camera or arm |
| Target location | Spatial distribution of the requested object | Low-resolution heatmap per camera |
| Grasp point | Preferred contact point rather than only object center | Normalized `(u, v)` or soft-argmax of a heatmap |
| Grasp orientation | Preferred gripper alignment | `(sin(theta), cos(theta))` |
| Destination presence | Whether the destination is visible | Binary logit |
| Destination location | Region where the object should be placed | Heatmap or normalized point |
| Arm activity | Whether each arm should move | Binary logit per arm |
| Manipulation phase | Current high-level stage | Approach, grasp, transfer, place, finish |

A bounding-box center or segmentation-mask center is not always a valid grasp point. This distinction is
important for thin assembly parts, screwdrivers, and other elongated objects. V2 should therefore supervise a
grasp affordance point and orientation when those labels are available. This is related to the motivation of
[RoboPoint](https://arxiv.org/abs/2406.10721), which predicts language-conditioned spatial affordance points
instead of relying only on broad object boxes.

#### Grounded Token Construction

The action decoder should not receive only a hard `argmax` coordinate. A hard coordinate discards uncertainty,
appearance, and surrounding context. It also blocks gradients through the selection operation.

Instead, each predicted heatmap should softly pool its corresponding dense image features:

```python
target_weights = softmax(target_heatmap.flatten(-2), dim=-1)
target_token = sum(target_weights * spatial_image_features)
```

The resulting token contains information about both the selected location and its local appearance. A 2D
position embedding should be added so that the action model can recover the selected image coordinates. The
condition encoder should retain the complete scene tokens as well, because the robot still needs table,
obstacle, gripper, and destination context.

The final condition sequence can contain:

```text
[condition, prompt, robot_state, target, destination, phase, arm_activity, full_image_tokens]
```

The noisy action chunk then attends to this condition sequence through the existing transformer decoder.

### Supervision And Dataset Features

V2 requires supervision that is not present in a normal action-only LeRobot dataset. Dense segmentation masks
are optional and expensive to store. A practical initial representation is to store compact labels in parquet
and generate Gaussian heatmaps during loading:

```text
target.visible                 bool
target.camera                  camera identifier
target.point                   float32[2], normalized to [0, 1]
target.orientation             float32[2], represented as sin/cos
destination.visible            bool
destination.point              float32[2], normalized to [0, 1]
arm_active                     bool[num_arms]
manipulation_phase             integer class
```

An optional sidecar annotation directory can store masks when mask supervision is useful. Invalid spatial
labels must have a corresponding validity mask so that absent or occluded targets do not contribute to point or
orientation losses.

The task prompt must explicitly identify the source and destination. Prompts such as "grasp the tomato and place
it on the light-green plate" are suitable. A generic prompt such as "grasp the parts" cannot teach the policy to
distinguish one instance or category from another.

#### Offline Annotation Pipeline

Grounding DINO and SAM/SAM2 should initially be used as offline annotation teachers, rather than as mandatory
models in the real-time control loop:

1. Run Grounding DINO with the episode's source and destination phrases.
2. Pass the detected boxes to SAM to obtain object masks.
3. Use SAM2 or another tracker to propagate accepted masks through the episode.
4. Derive object centers, approximate orientations, and visibility labels.
5. Display the labels over the recorded videos for manual review.
6. Correct frames or episodes where the teacher selected a distractor.

Grounding DINO plus SAM supports open-vocabulary text-conditioned detection and segmentation, as described by
[Grounded SAM](https://arxiv.org/abs/2401.14159). It should not be trusted without validation for small,
domain-specific assembly parts. For those objects, manually annotating one or more keyframes and tracking the
selection through the episode may be more reliable.

The grasp point should preferably describe the demonstrated contact location, not merely the mask center. It can
be annotated at the final pre-grasp frame and propagated to nearby approach frames. Grasp orientation can be
annotated manually or estimated from an elongated mask and then corrected.

### Dataset Design Requirements

Explicit grounding cannot compensate for a dataset that allows a fixed-position shortcut. The V2 dataset must
break correlations between prompt, object, and location. It should include:

- The same target object at several positions and orientations.
- Different target objects at the same position.
- Distractor objects that must not be grasped.
- Multiple prompts applied to visually similar scenes.
- Target-absent and destination-absent observations.
- Empty left-side and right-side examples for bimanual tasks.
- Examples where one arm remains stationary while the other arm acts.
- Different source and destination combinations.
- Occlusion during approach and grasp.
- Successful centered grasps with consistent gripper orientation.

Prompt paraphrases can improve language robustness, but they should preserve an unambiguous source and
destination. More demonstrations alone are insufficient if every demonstration keeps the same object-position
relationship.

A useful causal test can be created without collecting a robot trajectory for every prompt. The same image can
be paired with several prompts for grounding-only training, as long as the source and destination labels are
known. Action-flow training still uses the prompt associated with the demonstrated action, while a separate
grounding pretraining stage can use these additional prompt-image pairs.

### Training Objectives

The main loss should combine action flow matching with explicit grounding losses:

```text
L_total = L_flow
        + lambda_heatmap * L_target_heatmap
        + lambda_point * L_grasp_point
        + lambda_orientation * L_grasp_orientation
        + lambda_destination * L_destination
        + lambda_presence * L_presence
        + lambda_arm * L_arm_activity
        + lambda_phase * L_phase
```

Suggested loss types are:

- `L_flow`: the existing padded flow-matching MSE.
- `L_target_heatmap`: focal BCE or KL divergence against a Gaussian target heatmap.
- `L_grasp_point`: Huber loss on normalized `(u, v)` coordinates.
- `L_grasp_orientation`: cosine or MSE loss on normalized `(sin(theta), cos(theta))`.
- `L_destination`: the same heatmap or point loss used for the source target.
- `L_presence`: binary cross entropy.
- `L_arm_activity`: binary cross entropy for each arm.
- `L_phase`: cross entropy over manipulation stages.

All losses should be normalized to similar numerical scales before selecting final weights. Reasonable initial
weights are `1.0` for flow, heatmap, point, and destination losses; `0.2` for orientation, presence, and arm
activity; and `0.1` for phase classification. These are starting values only. Training logs should report every
loss independently, along with the gradient norm, so that a large auxiliary loss cannot silently dominate action
learning.

### Staged Training Plan

V2 should be developed in stages rather than trained end to end immediately.

#### Stage 1: Dataset And Label Audit

1. Select a small representative subset of episodes.
2. Produce target, destination, visibility, orientation, and phase labels.
3. Build a visualization script that overlays all labels on every camera.
4. Reject or correct visibly wrong labels before model training.

#### Stage 2: Grounding Probe

1. Freeze DINOv2 and SigLIP.
2. Train only the projectors, grounding queries, and grounding heads.
3. Evaluate target presence, point error, orientation error, and destination localization.
4. Test the same image with different prompts and verify that the selected target changes.

No robot rollout should be attempted until this stage reliably selects the prompted target and returns low
confidence for absent targets.

#### Stage 3: Frozen-Grounder Action Training

1. Load the trained grounding module.
2. Freeze the visual, language, and grounding modules.
3. Train the V1 flow action decoder using grounded tokens as additional conditions.
4. Compare against V1 using the same dataset, chunk size, training steps, RTC settings, and evaluation scenes.

This stage determines whether correct grounding information actually improves action generation.

#### Stage 4: Joint Fine-Tuning

1. Unfreeze the grounding projectors and query transformer.
2. Continue optimizing the combined loss with a smaller grounding learning rate.
3. If necessary, unfreeze only the final visual blocks with an even smaller learning rate.
4. Keep the pretrained text encoder frozen unless prompt understanding is demonstrably inadequate.

Joint fine-tuning should be stopped if grounding metrics deteriorate while action loss improves. That pattern
would indicate that action shortcuts are overwriting the explicit visual representation.

### Inference And Rollout

The inference path should be:

1. Tokenize and encode the task prompt. Cache the language tokens for the episode.
2. Encode the latest camera observations.
3. Predict target, destination, presence, arm activity, and phase outputs.
4. Construct differentiable grounded tokens.
5. Encode the full observation condition once per action-chunk prediction.
6. Run flow denoising and return a continuous action chunk.
7. Use RTC to maintain consistency with the unexecuted portion of the previous chunk.

RTC and visual grounding solve different problems. RTC improves continuity between action chunks, while the
grounding module decides what the chunk should act on. V2 should retain RTC support but should not treat RTC as a
solution for incorrect target selection.

For debugging, rollout should optionally record:

- Target and destination heatmaps overlaid on each camera frame.
- Presence probabilities.
- Per-arm activity probabilities.
- Predicted manipulation phase.
- Selected grasp point and orientation.
- The generated action chunk and RTC-guided action chunk.

Initially, low target-presence confidence should only produce a warning and visualization. After the presence
head is validated, a configurable safety rule can hold the corresponding arm when the target is absent. A hard
gate should not be enabled before measuring false-negative rates, because an incorrect absence prediction would
prevent a valid task from starting.

### Evaluation Plan

V2 should be evaluated at both perception and control levels.

#### Offline Perception Metrics

- Target-presence precision, recall, and F1 score.
- Target-point error in pixels and as a fraction of the image diagonal.
- Percentage of predicted points inside the target mask.
- Grasp-orientation angular error.
- Destination-point error.
- Arm-activity accuracy, especially for empty-side observations.
- Manipulation-phase accuracy and confusion matrix.

#### Counterfactual Grounding Tests

These tests are necessary to distinguish prompt understanding from position memorization:

1. Keep the image fixed and change the target name in the prompt.
2. Keep the prompt fixed and move the target to a different position.
3. Remove the target while keeping distractors visible.
4. Swap source and destination descriptions.
5. Hide the object on one side of a bimanual scene.

The predicted heatmap and arm activity should change consistently with the prompt and scene.

#### Real-Robot Metrics

- Correct-target selection rate.
- Empty-side false-motion rate.
- Grasp success rate.
- Grasp-point and gripper-orientation quality.
- Drop rate during transfer.
- Placement success rate.
- Complete task success rate.
- Action discontinuity and RTC correction magnitude.

Evaluation should separately report perception failures and control failures. For example, "correct target but
missed grasp" is a controller or affordance error, while "moved to a distractor" is a grounding error. This
separation is necessary to determine what V3 should improve.

### Required Ablations

The following controlled comparisons should be performed:

| Experiment | Purpose |
| --- | --- |
| V1 ResNet baseline | Establish the existing behavior |
| DINOv2 without language | Measure the benefit of stronger spatial features alone |
| DINOv2 plus SigLIP prompt tokens | Measure prompt conditioning without explicit grounding loss |
| Add target/presence supervision | Measure explicit object grounding |
| Add grasp point and orientation | Measure affordance precision |
| Add phase and arm-activity heads | Measure conditional and empty-side behavior |
| Frozen versus partially unfrozen backbone | Measure adaptation versus representation forgetting |
| With and without RTC | Separate perception gains from chunk-continuity gains |

Only one major component should be changed in each ablation. Otherwise, improved performance cannot be assigned
to the visual backbone, language conditioning, grounding supervision, or rollout strategy.

### Implementation Plan

The initial repository structure should be:

```text
repos/lerobot/src/lerobot/policies/action_flow_v2/
|-- __init__.py
|-- configuration_action_flow_v2.py
|-- modeling_action_flow_v2.py
|-- processor_action_flow_v2.py
`-- grounding_action_flow_v2.py

examples/action_flow_v2/
|-- annotate_grounding.py
|-- inspect_grounding_labels.py
|-- inspect_grounding_predictions.py
`-- compare_prompts.py
```

`ActionFlowV2Config` should initially expose:

```text
vision_model_name
text_model_name
freeze_vision_encoder
freeze_text_encoder
grounding_dim
grounding_num_heads
grounding_num_layers
use_target_heatmap
use_destination_heatmap
use_grasp_orientation
use_phase_head
use_arm_activity_head
grounding_loss_weight
presence_loss_weight
orientation_loss_weight
phase_loss_weight
chunk_size
num_inference_steps
rtc_config
```

The implementation should reuse V1's flow timestep embedding, action projection, velocity head, flow objective,
action queue, normalization processors, and RTC integration. This minimizes duplicated code and keeps the V2
experiment focused on grounded conditioning.

### Initial Success Criteria

Before considering V2 successful, it should satisfy all of the following:

1. Changing the prompt for the same image changes the predicted target heatmap.
2. Moving the target changes the predicted heatmap without changing the prompt.
3. Removing the target produces a low presence probability and suppresses false arm motion.
4. Grounding visualizations select the demonstrated object rather than a fixed workspace position.
5. Grasp-point error is lower than using the object or mask center alone.
6. Real-robot correct-target selection improves over V1 under matched training data.
7. Grasp success improves without increasing action discontinuity or table contact.

If V2 improves grounding metrics but not grasp success, the remaining bottleneck is likely 3D geometry,
affordance prediction, calibration, or low-level control rather than target recognition. In that case, the next
step should add depth-aware target features or explicit 3D grasp-point supervision instead of increasing model
size alone.
