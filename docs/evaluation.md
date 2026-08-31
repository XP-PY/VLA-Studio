# SmolVLA Baseline vs. Action Flow V2.3 and V2.4

## Objective

This evaluation compares the trained SmolVLA baseline with Action Flow V2.3 (AFV2.3) and Action
Flow V2.4 (AFV2.4). Each policy is measured at the target-selection, grasp, transport, placement,
and complete-task stages. The comparison evaluates complete training recipes rather than isolating
architecture: AFV2.3 and AFV2.4 use additional planning, grounding, and discrete-action labels that
are unavailable to the baseline.

## Policy Overview

| Policy | Structure | Executed action |
|---|---|---|
| SmolVLA baseline | Wrist/top images, task prompt, and robot state are encoded by SmolVLM; a flow-matching expert directly predicts an action chunk. | Absolute six-joint angles and gripper position. |
| Action Flow V2.3 | The same visual-language backbone additionally predicts the current Plan, target boxes, and auxiliary FAST action tokens. The predicted Plan conditions a knowledge-insulated flow expert. | Relative six-joint commands and absolute gripper position. FAST tokens are auxiliary and are not executed. |
| Action Flow V2.4 | A pretrained $\pi_{0.5}$ PaliGemma 2B prefix predicts Plan, target-box, and auxiliary FAST tokens. Its predicted Plan conditions the pretrained Gemma 300M flow-matching action expert. | Relative six-joint commands and absolute gripper position. FAST tokens are auxiliary and are not executed. |

AFV2.3 uses ground-truth Plan tokens as teacher-forcing context during training and its own predicted
Plan during rollout. Continuous-action supervision is limited to the current stage so an action
chunk does not cross from grasp into transport or from transport into placement.

AFV2.4 retains the same Plan-conditioned, stage-bounded objective but starts from the released
$\pi_{0.5}$ visual-language and action-expert weights. Unlike AFV2.3 knowledge insulation, AFV2.4
allows flow loss to update the shared PaliGemma prefix during joint fine-tuning.

## Training Data

All three policies use the same `442` RM65 demonstrations (`239,641` frames at `30 FPS`) and `52` task
prompts. Observations contain a wrist camera, a top camera, and seven robot-state values (six joints
plus the gripper). Actions have the same seven physical dimensions.

| Policy | Additional supervision |
|---|---|
| SmolVLA baseline | None beyond task prompts, observations, and robot actions. |
| Action Flow V2.3 | Per-frame stage/next-step labels, stage boundaries, target entities, and target bounding boxes. |
| Action Flow V2.4 | The same Plan, stage-boundary, target-box, and stage-local FAST labels as AFV2.3. |

## Training Configuration

| Parameter | SmolVLA baseline | Action Flow V2.3 | Action Flow V2.4 |
|---|---:|---:|---:|
| Initialization | `lerobot/smolvla_base` | `lerobot/smolvla_base` | `lerobot/pi05_base` |
| VLM / action expert | SmolVLM2 500M, 16 layers | SmolVLM2 500M, 16 layers | PaliGemma 2B / Gemma 300M |
| Training steps | `100,000` | `100,000` | `100,000` |
| Global batch size | `16` | `16` | `16` |
| Action chunk size | `30` | `30` | `30` |
| Approximate dataset epochs | `6.68` | `6.68` | `6.68` |
| Optimizer | AdamW | AdamW | AdamW |
| Peak learning rate | `3e-5` | `3e-5` | `2.5e-5` |
| Warmup | `2,000` steps | `2,000` steps | `1,000` steps |
| Learning-rate schedule | Cosine decay to `2.5e-6` | Constant `3e-5` after warmup | Cosine decay to `2.5e-6` |
| Vision encoder frozen | No | No | No |
| Image augmentation | Disabled | Disabled | Disabled |
| Arm action representation | Absolute | Relative | Relative |
| Gripper action representation | Absolute | Absolute | Absolute |
| Plan / Box / FAST / Flow weights | Not applicable | `1.0 / 1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0 / 1.0` |
| Grounding-keyframe sampling | Not applicable | `0.05` | `0.05` |
| FAST vocabulary | Not applicable | `2,048` robot-adapted tokens | `2,048` robot-adapted tokens |
| Flow gradient into VLM prefix | Yes | No | Yes |

Training losses are not directly comparable because SmolVLA optimizes only continuous flow
matching, whereas AFV2.3 and AFV2.4 sum Plan, Box, FAST, and Flow objectives.

## Inference Configuration

All three policies were evaluated with the same effective inference and robot-control settings.
Only parameters that affect policy inference or action execution are listed.

| Parameter | Value |
|---|---:|
| `policy.n_action_steps` | `1` |
| `policy.num_inference_steps` | `4` |
| `fps` / `dataset.fps` | `10` |
| `inference.type` | `rtc` |
| `inference.rtc.execution_horizon` | `2` |
| `inference.queue_threshold` | `10` |
| `inference.rtc.max_guidance_weight` | `10` |
| `inference.rtc.prefix_attention_schedule` | `LINEAR` |
| `robot.single_arm.canfd_follow` | `false` |

`EVAL_NUM_STEPS=4` overrides the general `NUM_INFERENCE_STEP=10` setting, so the effective number
of flow-matching inference steps in these evaluations is `4`.

## Evaluation Metrics

| Metric | Definition |
|---|---|
| Grasp success | The requested object is securely lifted from its initial support. |
| Transport success | Given a successful grasp, the object remains held until it reaches the destination area. |
| Placement success | Given successful transport, the object is released on or inside the requested destination. |
| Complete-task success | Every required object manipulation in the prompt is completed correctly. |

The scene setup, prompt set, initial robot pose, and rollout settings are kept consistent across
policies. Conditional stage rates and unconditional end-to-end rates are both reported so an early
grasp failure cannot be mistaken for a placement failure.

The rollout artifact groups supporting this comparison are registered in
`configs/evaluations.yaml`. Some interactive evaluation directories contain camera videos but show
zero finalized episodes in LeRobot metadata, so the manually adjudicated trial counts in this
document are authoritative and must not be replaced by `meta/info.json` totals.

## Evaluation Results

The SmolVLA and AFV2.3 evaluations contain `15` trials and `15` evaluated object cycles. AFV2.4 was
evaluated for `18` trials and `18` object cycles with the same prompt set, scene construction,
camera placement, initial robot pose, rollout settings, and stopping rule. For `Put all vegetables
into the collection box`, a rollout stopped after the first unsuccessful grasp attempt counts as one
evaluated object cycle.

An *object cycle* is one requested source-object-to-destination manipulation. Success metrics are
cumulative: for example, `Transported` means that the correct object was grasped and remained held
until reaching the destination. Assign exactly one primary failure stage to each unsuccessful
object cycle, using the earliest stage that prevents completion.

| Type | Metric | SmolVLA baseline | AFV2.3 | AFV2.3 $\Delta$ | AFV2.4 | AFV2.4 $\Delta$ |
|---|---|---:|---:|---:|---:|---:|
| Sample size | Trials | `15` | `15` | - | `18` | - |
| Sample size | Evaluated object cycles | `15` | `15` | - | `18` | - |
| Success | Correct source object and destination selected | `2/15 (13.3%)` | `2/15 (13.3%)` | `0.0` | `18/18 (100.0%)` | `+86.7` |
| Success | Correct object grasped | `2/15 (13.3%)` | `0/15 (0.0%)` | `-13.3` | `18/18 (100.0%)` | `+86.7` |
| Success | Object transported without dropping | `2/15 (13.3%)` | `0/15 (0.0%)` | `-13.3` | `16/18 (88.9%)` | `+75.6` |
| Success | Object placed at the requested destination | `1/15 (6.7%)` | `0/15 (0.0%)` | `-6.7` | `16/18 (88.9%)` | `+82.2` |
| Success | Complete task | `1/15 (6.7%)` | `0/15 (0.0%)` | `-6.7` | `16/18 (88.9%)` | `+82.2` |
| Primary failure | Wrong object or destination selected | `13/15 (86.7%)` | `6/15 (40.0%)` | `-46.7` | `0/18 (0.0%)` | `-86.7` |
| Primary failure | Failed to align with or close on the object | `0/15 (0.0%)` | `2/15 (13.3%)` | `+13.3` | `0/18 (0.0%)` | `0.0` |
| Primary failure | Object dropped during transport | `0/15 (0.0%)` | `0/15 (0.0%)` | `0.0` | `2/18 (11.1%)` | `+11.1` |
| Primary failure | Failed to place or release the object | `0/15 (0.0%)` | `0/15 (0.0%)` | `0.0` | `0/18 (0.0%)` | `0.0` |
| Primary failure | Incorrect stage transition, timeout, or no meaningful action | `1/15 (6.7%)` | `7/15 (46.7%)` | `+40.0` | `0/18 (0.0%)` | `-6.7` |
| Recovery | Plan returned to `grasp` after a detected drop | Not observed | Not observed | - | `2/2 (100.0%)` | - |
| Recovery | Valid recovery completed after a drop | Not observed | Not observed | - | `0/2 (0.0%)` | - |

Success and primary-failure metrics use evaluated object cycles as their denominator; recovery rates
are conditional on a transport drop. Each $\Delta$ is the Action Flow result minus the SmolVLA
baseline in percentage points. Positive success differences are improvements; negative
primary-failure differences mean fewer failures of that type.

### Prompts and Observed Behavior

#### Task 1: Place One Vegetable

- Place the tomato on the yellow plate.
- Place the tomato on the pink plate.
- Place the garlic on the blue plate.
- Place the garlic on the yellow plate.
- Place the corn on the pink plate.
- Place the corn on the blue plate.
- Place the cucumber on the yellow plate.
- Place the cucumber on the pink plate.

The SmolVLA baseline mostly ignored the prompt and repeatedly attempted to move the garlic to the
blue plate. AFV2.3 showed the same fixed target-destination preference for most prompts. AFV2.4
selected the instructed vegetable and plate correctly and maintained the correct Plan stage in all
tested layouts.

#### Task 2: Clear a Specified Plate

- Put the vegetable from the pink plate into the collection box.
- Put the vegetable from the yellow plate into the collection box.
- Put the vegetable from the blue plate into the collection box.

The SmolVLA baseline always attempted to grasp the garlic instead of identifying the vegetable on
the specified plate. In every AFV2.3 trial, the first predicted Plan was `put collection box`, so it
skipped the required grasp stage. AFV2.4 identified both the vegetable on the specified plate and the
collection box correctly, beginning with `grasp` before transitioning to `put`.

#### Task 3: Bus the Table

- Put all vegetables into the collection box.

The SmolVLA baseline again attempted to grasp the garlic. AFV2.3's predicted Plan changed repeatedly
during its first grasp attempt. Those two rollouts were stopped after that attempt failed; later
object cycles were not evaluated. AFV2.4 correctly recognized the active object and collection box
and kept its Plan aligned with the observed manipulation stage.

### Analysis

1. **AFV2.4 resolves the main semantic failure of the smaller policies.** It selected the instructed
   source and destination in `18/18` cycles, whereas SmolVLA and AFV2.3 each achieved only `2/15`.
   Its Plan predictions were also correct at every observed stage. This indicates that the pretrained
   PaliGemma prefix provides substantially stronger prompt grounding and visual task understanding.
2. **The semantic improvement transfers to end-to-end control.** AFV2.4 completed `16/18` cycles
   (`88.9%`), compared with `1/15` for SmolVLA and `0/15` for AFV2.3. It had no wrong-target,
   initial-grasp, placement, or incorrect-stage primary failures.
3. **Plan recovery is responsive, but physical recovery remains incomplete.** In both transport-drop
   cases, AFV2.4 quickly detected the changed scene and returned its Plan from `put` to `grasp`.
   Neither recovery completed successfully, so correct high-level state estimation alone is not
   sufficient for robust regrasping.
4. **The first recovery failure is caused by an out-of-distribution object arrangement.** After the
   dropped object came into contact with the collection box, the recovery grasp lifted both together.
   The policy recognized the intended object but lacked demonstrations for separating or safely
   grasping objects that touch the destination container.
5. **The second recovery failure exposes a low-level gripper-state loop.** The Plan correctly returned
   to `grasp`, but the gripper retained its partially closed transport width instead of reopening to
   the demonstrated pre-grasp width. As the gripper approached the object, the policy visually
   inferred grasp success too early, switched to `put`, then returned to `grasp` after lifting without
   the object. This repeated because the gripper was never fully reset before another attempt.
6. **The remaining bottleneck has shifted from recognition to recovery control.** Future data should
   emphasize dropped-object recovery, fully reopening before regrasp, object-container contact, and
   explicit evidence of grasp success before the `grasp`-to-`put` transition. The result supports the
   complete AFV2.4 training recipe, but it does not isolate whether the gain comes from model scale,
   $\pi_{0.5}$ pretraining, architecture, or their combination.
7. **The comparison remains small and is not perfectly balanced.** AFV2.4 has `18` cycles versus `15`
   for the other policies. Its large margin is clear in this setup, but a larger paired evaluation is
   still needed for a statistically stronger comparison.

### Conclusion

AFV2.4 is the first evaluated policy in this series to combine reliable instruction grounding,
correct stage prediction, and high end-to-end task success. Its `88.9%` completion rate is a large
practical improvement over both the SmolVLA baseline and AFV2.3. The next iteration should preserve
the current semantic pipeline and focus on gripper reset, grasp confirmation, and recovery examples
rather than further changing the target-recognition mechanism.
