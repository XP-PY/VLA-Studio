# Pi0 or Pi0-FAST Training Log

- [Pi0 paper](https://arxiv.org/abs/2410.24164)
- [Pi0-FAST paper](https://arxiv.org/abs/2501.09747)

*Last updated: July 20, 2026*

---

## Model Assets

Before training, download the base PI0 policy and the paligemma model used by the policy:

```bash
./scripts/model_download.sh lerobot/pi0_base
./scripts/model_download.sh google/paligemma-3b-pt-224
```

The local paths used in the training command are:

- SmolVLA base policy: `${STORAGE_ROOT}/models/pi0_base`
- SmolVLM model: `${STORAGE_ROOT}/models/paligemma-3b-pt-224`

---

## Training Objective 1: Single-Arm Table Preparation and Bussing

This experiment evaluates whether PI0 can use language instructions to prepare and clear a table containing small colored plates and vegetable models. Each plate can hold only one vegetable. The policy must identify the requested vegetable or plate, execute grasp-and-place actions, and leave unrelated objects untouched. All objects are placed within the reachable workspace of a single RM65 arm.

### Task Design

The complete object set contains 12 vegetable models and five plate colors, but recording every combination would make the first experiment unnecessarily large. This experiment therefore uses four visually distinct vegetables and three plate colors:

- **Vegetables:** tomato, garlic, corn, and cucumber
- **Plate colors:** pink, yellow, and blue

These vegetables differ clearly in shape, color, and texture: the tomato is red and round, the garlic is white and bulb-shaped, the corn is yellow and textured, and the cucumber is green and elongated. The selected plate colors are also reasonably distinct. Every plate can hold at most one vegetable. The remaining objects can be introduced in a later dataset after this smaller task set works reliably.

Experimental environment setup:

- Use one [Realman RM65 Robotic Arm](https://github.com/RealManRobot/rm_models/tree/main/RM65) and two Intel RealSense D435 cameras:

    - Name: Intel RealSense D435
    - Type: RealSense
    - Default stream profile:
        - Stream_type: Color
        - Format: rgb8
        - Width: 640
        - Height: 480
        - FPS: 30

- Experimental scene:

    ![ob_1_scene](./assets/pi_0/ob_1_scene.jpg)

- camera view:

    <table>
    <tr>
        <td align="center">
        <img src="assets/pi_0/ob_1_top_camera.png" width="400"><br>
        <b>Top Camera</b><br>
        <sub>Looking down at the workspace</sub>
        </td>
        <td align="center">
        <img src="assets/pi_0/ob_1_wrist_camera.png" width="400"><br>
        <b>Wrist Camera</b><br>
        <sub>Mounted on the robotic arm end effector</sub>
        </td>
    </tr>
    </table>

The dataset contains four related tasks:

1. **Place one vegetable:** Select a named vegetable and place it on a named plate while leaving the distractors unchanged. This tests vegetable identity, plate-color grounding, and grasp accuracy.
2. **Arrange two vegetables:** Place two different vegetables on two different plates in the order given by the instruction. This introduces sequencing while preserving the one-vegetable-per-plate constraint.
3. **Clear a specified plate:** Move the vegetable from the named plate into the collection box while leaving vegetables on the other plates unchanged. This tests plate-color grounding and selective action.
4. **Bus the table:** Put every vegetable into the collection box while leaving the plates in place. The number of vegetables varies, so the policy must repeat the grasp-and-place behavior and recognize when the task is complete.

Plate colors should be moved among different physical positions so that the policy learns their visual appearance rather than associating each color with a fixed location. The requested vegetable should also appear with at least one distractor whenever the task permits it.

---

### Training Log 1 of Objective 1

#### Dataset Recording

Use the prompt templates exactly as written. Save only successful demonstrations, begin each episode from the same observation pose, and include a short pause after stable grasps, placements, and the completed final state. Keep the cameras, lighting, and collection-box appearance unchanged while varying object layouts within safe reachable regions.

| Task | Prompt template | Episode allocation | Episodes | Total frames |
|---|---|---|---:|---:|
| Place one vegetable | `Place the {vegetable} on the {plate_color} plate.` | Record every combination of 4 vegetables and 3 plate colors four times: `4 x 3 x 4 = 48`. Show all three plates, keep the requested plate empty, and add one to three distractor vegetables. Use a different reachable object and plate layout for each repetition. | 48 | 16,212 |
| Arrange two vegetables | `Place the {vegetable_1} on the {plate_color_1} plate, then place the {vegetable_2} on the {plate_color_2} plate.` | Record one demonstration for every combination in which tomato or corn is placed first: 2 first-vegetable choices x 3 second-vegetable choices x 6 ordered plate pairs = 36. Keep both requested plates empty and vary which requested vegetable starts closer to the arm. Reserve all 36 combinations with garlic or cucumber first for sequence-generalization evaluation. | 36 | 20,196 |
| Clear a specified plate | `Put the vegetable from the {plate_color} plate into the collection box.` | Use the same two predetermined collection-box locations used by the bussing task. At each box location, record 12 episodes for each of the 3 requested plate colors: `2 x 3 x 12 = 72`. Include at least one occupied non-target plate and leave every non-target vegetable unchanged. Vary the plate positions, vegetable identities, and layouts across episodes. | 72 | 22,149 |
| Bus the table | `Put all vegetables into the collection box.` | Use the same two predetermined collection-box locations as the selective-clearing task. At each box location, record 12 episodes with 2 vegetables and 12 episodes with 3 vegetables: `2 x 2 x 12 = 48`. Vary which vegetables begin on plates or directly on the table. Exclude all 4-vegetable demonstrations from training and use them only to evaluate object-count generalization. | 48 | 34,642 |
| **Total** | **52 language prompts** |  | **204** | **93,199** |

#### Training Configuration

The absolute joint-angle dataset was copied to a PI0-specific dataset whose action statistics were recomputed in relative space. During training, the six joint targets are represented as offsets from the current joint state, while `gripper.pos` remains absolute. The stored trajectories themselves remain unchanged.

| Parameter | Configured value | Default value |
|---|---:|---:|
| `dataset.return_uint8` | `true` | `false` |
| `policy.dtype` | `bfloat16` | `float32` |
| `policy.use_relative_actions` | `true` | `false` |
| `policy.freeze_vision_encoder` | `true` | `false` |
| `policy.train_expert_only` | `true` | `false` |
| `policy.gradient_checkpointing` | `true` | `false` |
| `policy.push_to_hub` | `false` | `true` |
| `batch_size` | `16` | `8` |
| `steps` | `30,000` | `100,000` |
| `eval_freq` | `0` (disabled) | `20,000` |
| `save_freq` | `10,000` | `20,000` |
| `log_freq` | `100` | `200` |
| `wandb.enable` | `true` | `false` |
| `wandb.project` | `lerobot_pi0_rm65` | `lerobot` |
| `wandb.disable_artifact` | `true` | `false` |

> **Note:** Parameters not listed in this table use their default values.

#### Evaluation

The policy was evaluated with the prompt `Place the tomato on the pink plate.`. These initial rollouts were used to identify qualitative failure modes; a formal success rate has not yet been measured.

Several flow-matching and RTC configurations were tested:

| Parameter | Tested values |
|---|---|
| `policy.num_inference_steps` | `5`, `8`, `10` |
| `inference.rtc.execution_horizon` | `8`, `10`, `15` |
| `inference.rtc.max_guidance_weight` | `8`, `10` |
| `inference.rtc.prefix_attention_schedule` | `EXP`, `LINEAR` |
| `inference.queue_threshold` | `15`, `20`, `30` |

All tested configurations produced similar behavior. Some combinations were slightly better than others, but none removed the main failure modes described below.

- Rollout 1:

  <video controls width="600">
    <source src="assets/pi_0/ob_1_log_1_rollout_0.mp4" type="video/mp4">
  </video>

- Rollout 2:

  <video controls width="600">
    <source src="assets/pi_0/ob_1_log_1_rollout_1.mp4" type="video/mp4">
  </video>

##### Experimental Phenomena

1. The motion is not continuous. Consecutive action updates sometimes conflict, causing the arm to move toward the target and then move backward before approaching it again.
2. The policy shows partial prompt and object recognition. With the tomato instruction, the arm often moves toward the tomato, but it cannot align the gripper with sufficient precision. In some trials, it also approaches the wrong object.
3. When the gripper misses the tomato, the policy does not stop or retry the grasp. Instead, it continues executing the later lift-and-place portion of the demonstrated trajectory as though the grasp had succeeded.

##### Analysis

1. The forward-and-backward motion suggests inconsistency between consecutive predicted action chunks. Each flow-matching inference begins from newly sampled noise, and RTC must reconcile the new prediction with the unexecuted portion of the previous chunk. Because broad changes to the inference-step and RTC parameters produced only minor differences, the oscillation is unlikely to be caused by one poorly chosen RTC setting alone. The learned action distribution, chunk-to-chunk variation, observation-to-action latency, and relative-action execution may all contribute.
2. The policy has learned a coarse association between the language instruction and the target area, but not sufficiently accurate visual grounding for grasping. Each exact vegetable-and-plate combination has only four demonstrations, while the full dataset contains 52 prompts. In addition, the vision encoder is frozen, so adaptation to the cameras, toy vegetables, and workspace must mainly occur through the action expert.
3. Continuing after a failed grasp is expected from success-only behavior cloning. The dataset contains trajectories in which approaching the object is always followed by lifting and placement, but it does not teach the conditional rule that lifting should occur only after a successful grasp. The policy also has no explicit grasp-success or object-contact signal, so it tends to replay the learned phase sequence instead of recovering from failure.
4. The next evaluation should measure these problems separately: target-selection accuracy, gripper-alignment accuracy, grasp success, and post-failure recovery. This will distinguish visual-grounding errors from action-continuity and recovery errors.

---

### Training Log 2 of Objective 1

This run uses the same 204-episode PI0 dataset as Training Log 1, but changes the optimization scope from expert-only training to full-model fine-tuning. The run uses a chunk size of 50, a per-device batch size of 8, and 100,000 optimization steps.

#### Why Fine-Tune More Than the Action Expert

Training Log 1 froze the PaliGemma VLM and trained only the action expert and projection layers. Its evaluation showed some coarse prompt-to-target association, but target selection and gripper alignment remained unreliable. With expert-only training, the policy can learn actions from the VLM's existing features, but the robot dataset cannot improve those features. This is restrictive because the wrist and top-camera views, toy vegetables, colored plates, and robot workspace differ substantially from PaliGemma's pretraining data.

For this run, `policy.freeze_vision_encoder=false` and `policy.train_expert_only=false`. The action loss can therefore update the PaliGemma vision tower and multimodal model together with the action expert and projection layers. The intended benefit is stronger adaptation of object identity, plate color, spatial relationships, and prompt-image alignment to the recorded environment. This should be tested through target-selection and grasp accuracy rather than training loss alone.

#### Training Configuration

The relative-action conversion is unchanged from Training Log 1: the six arm joints are trained as offsets from the current state, while `gripper.pos` remains absolute. The following table lists only values that differ from the current LeRobot PI0 training defaults.

| Parameter | Configured value | Default value |
|---|---:|---:|
| `dataset.return_uint8` | `true` | `false` |
| `policy.dtype` | `bfloat16` | `float32` |
| `policy.use_relative_actions` | `true` | `false` |
| `policy.scheduler_decay_steps` | `100,000` | `30,000` |
| `policy.push_to_hub` | `false` | `true` |
| `eval_freq` | `0` (disabled) | `20,000` |
| `save_freq` | `10,000` | `20,000` |
| `log_freq` | `100` | `200` |
| `wandb.enable` | `true` | `false` |
| `wandb.project` | `lerobot_pi0_rm65` | `lerobot` |
| `wandb.disable_artifact` | `true` | `false` |

The explicitly configured `policy.freeze_vision_encoder=false`, `policy.train_expert_only=false`, `policy.gradient_checkpointing=false`, `policy.chunk_size=50`, `batch_size=8`, and `steps=100,000` already match their defaults, so they are not repeated in the override table.

> **Note:** Parameters not listed in this table use their default values.
