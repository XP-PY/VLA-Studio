# SmolVLA and $\pi_{0.5}$ Baselines vs. Action Flow V2.3 and V2.4

## Objective

This evaluation compares two directly post-trained baselines, SmolVLA and $\pi_{0.5}$, with Action
Flow V2.3 (AFV2.3) and Action Flow V2.4 (AFV2.4). Each policy is measured at the target-selection,
grasp, transport, placement, and complete-task stages. The comparison evaluates complete training
recipes rather than isolating architecture: the baselines use standard LeRobot robot-trajectory
supervision, whereas AFV2.3 and AFV2.4 additionally use planning, grounding, stage-boundary, and
discrete-action labels.

## Policy Overview

| Policy | Structure | Executed action |
|---|---|---|
| SmolVLA baseline | Wrist/top images, task prompt, and robot state are encoded by SmolVLM; a flow-matching expert directly predicts an action chunk. | Absolute six-joint angles and gripper position. |
| $\pi_{0.5}$ baseline | The standard LeRobot $\pi_{0.5}$ policy uses its PaliGemma 2B prefix and Gemma 300M flow-matching action expert to predict actions directly. It has no explicit Plan, target-box, or auxiliary FAST prediction branch. | Continuous robot action chunk predicted directly by the standard $\pi_{0.5}$ policy. |
| Action Flow V2.3 | The same visual-language backbone additionally predicts the current Plan, target boxes, and auxiliary FAST action tokens. The predicted Plan conditions a knowledge-insulated flow expert. | Relative six-joint commands and absolute gripper position. FAST tokens are auxiliary and are not executed. |
| Action Flow V2.4 | A pretrained $\pi_{0.5}$ PaliGemma 2B prefix predicts Plan, target-box, and auxiliary FAST tokens. Its predicted Plan conditions the pretrained Gemma 300M flow-matching action expert. | Relative six-joint commands and absolute gripper position. FAST tokens are auxiliary and are not executed. |

AFV2.3 uses ground-truth Plan tokens as teacher-forcing context during training and its own predicted
Plan during rollout. Continuous-action supervision is limited to the current stage so an action
chunk does not cross from grasp into transport or from transport into placement.

AFV2.4 retains the same Plan-conditioned, stage-bounded objective but starts from the released
$\pi_{0.5}$ visual-language and action-expert weights. Unlike AFV2.3 knowledge insulation, AFV2.4
allows flow loss to update the shared PaliGemma prefix during joint fine-tuning.

The $\pi_{0.5}$ baseline is initialized from `lerobot/pi05_base` and trained with LeRobot's standard
$\pi_{0.5}$ post-training script. Its trajectory supervision consists only of camera images and
robot joint state/action data, with the task prompt used as ordinary policy conditioning. It does
not use AFV2.4's per-frame Plan, target-box, FAST-token, or stage-boundary labels. This makes
$\pi_{0.5}$ the most relevant baseline for assessing what the additional AFV2.4 planning pipeline
contributes on top of the same pretrained model family.

## Training Data

All four policies use the same `442` RM65 demonstrations (`239,641` frames at `30 FPS`) and `52` task
prompts. Observations contain a wrist camera, a top camera, and seven robot-state values (six joints
plus the gripper). Actions have the same seven physical dimensions.

| Policy | Additional supervision |
|---|---|
| SmolVLA baseline | None beyond task prompts, observations, and robot actions. |
| $\pi_{0.5}$ baseline | Camera images and robot joint state/action trajectories only; no AFV-specific labels. |
| Action Flow V2.3 | Per-frame stage/next-step labels, stage boundaries, target entities, and target bounding boxes. |
| Action Flow V2.4 | The same Plan, stage-boundary, target-box, and stage-local FAST labels as AFV2.3. |

## Training Configuration

| Parameter | SmolVLA baseline | Action Flow V2.3 | Action Flow V2.4 | $\pi_{0.5}$ baseline |
|---|---:|---:|---:|---:|
| Initialization | `lerobot/smolvla_base` | `lerobot/smolvla_base` | `lerobot/pi05_base` | `lerobot/pi05_base` |
| VLM / action expert | SmolVLM2 500M, 16 layers | SmolVLM2 500M, 16 layers | PaliGemma 2B / Gemma 300M | PaliGemma 2B / Gemma 300M |
| Training steps | `100,000` | `100,000` | `100,000` | `100,000` |
| Global batch size | `16` | `16` | `16` | `16` |
| Action chunk size | `30` | `30` | `30` | `30` |
| Approximate dataset epochs | `6.68` | `6.68` | `6.68` | `6.68` |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Peak learning rate | `3e-5` | `3e-5` | `2.5e-5` | `2.5e-5` |
| Warmup | `2,000` steps | `2,000` steps | `1,000` steps | `1,000` steps |
| Learning-rate schedule | Cosine decay to `2.5e-6` | Constant `3e-5` after warmup | Cosine decay to `2.5e-6` | Cosine decay to `2.5e-6` |
| Vision encoder frozen | No | No | No | No |
| Image augmentation | Disabled | Disabled | Disabled | Disabled |
| Arm action representation | Absolute | Relative | Relative | Relative |
| Gripper action representation | Absolute | Absolute | Absolute | Absolute |
| Plan / Box / FAST / Flow weights | Not applicable | `1.0 / 1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0 / 1.0` | `N/A / N/A / N/A / 1.0` |
| Grounding-keyframe sampling | Not applicable | `0.05` | `0.05` | Not applicable |
| FAST vocabulary | Not applicable | `2,048` robot-adapted tokens | `2,048` robot-adapted tokens | Not applicable |
| Flow gradient into VLM prefix | Yes | No | Yes | Yes |

The $\pi_{0.5}$ baseline uses the same training steps, batch size, action chunk, optimizer, learning
rate, warmup, scheduler, image augmentation, and action representation as AFV2.4. Their intentional
differences are AFV2.4's additional prediction heads and supervision: $\pi_{0.5}$ follows the
standard LeRobot post-training path and optimizes only its continuous flow-matching action objective,
whereas AFV2.4 jointly optimizes Plan, Box, FAST, and Flow objectives.

## Inference Configuration

All four policies were evaluated with the same effective inference and robot-control settings. In
particular, the $\pi_{0.5}$ baseline and AFV2.4 use identical inference parameters. Only parameters
that affect policy inference or action execution are listed.

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
| Transport success | The requested object eventually reaches the destination area, including after a successful recovery. |
| Placement success | Given successful transport, the object is released on or inside the requested destination. |
| Complete-task success | Every required object manipulation in the prompt is completed correctly. |

The scene setup, prompt set, initial robot pose, and rollout settings are kept consistent across
policies. Conditional stage rates and unconditional end-to-end rates are both reported so an early
grasp failure cannot be mistaken for a placement failure.

The SmolVLA, AFV2.3, and AFV2.4 rollout artifact groups are registered in
`configs/evaluations.yaml`; the $\pi_{0.5}$ baseline counts are currently recorded directly in this
document. Some interactive evaluation directories contain camera videos but show zero finalized
episodes in LeRobot metadata, so the manually adjudicated trial counts here are authoritative and
must not be replaced by `meta/info.json` totals.

## Evaluation Results

The SmolVLA and AFV2.3 evaluations contain `15` trials and `15` evaluated object cycles. AFV2.4 was
evaluated for `18` trials and `18` object cycles. The $\pi_{0.5}$ baseline artifact set contains `45`
recorded trials, from which `18` object cycles were included in the stage-level comparison below.
The compared cycles use the same prompt set, scene construction, camera placement, initial robot
pose, rollout settings, and stopping rule. For `Put all vegetables into the collection box`, a
rollout stopped after the first unsuccessful grasp attempt counts as one evaluated object cycle.

An *object cycle* is one requested source-object-to-destination manipulation. Success metrics are
cumulative: for example, `Transported` means that the correct object ultimately reached the
destination area, including after a successful recovery. Assign exactly one primary failure stage
to each unsuccessful object cycle, using the earliest stage that prevents completion.

| Type | Metric | SmolVLA baseline | AFV2.3 | AFV2.4 | $\pi_{0.5}$ baseline |
|---|---|---:|---:|---:|---:|
| Sample size | Recorded trials | `15` | `15` | `18` | `45` |
| Sample size | Evaluated object cycles | `15` | `15` | `18` | `18` |
| Success | Correct source object and destination selected | `2/15 (13.3%)` | `2/15 (13.3%)` | `18/18 (100.0%)` | `18/18 (100.0%)` |
| Success | Correct object grasped | `2/15 (13.3%)` | `0/15 (0.0%)` | `18/18 (100.0%)` | `18/18 (100.0%)` |
| Success | Object successfully transported | `2/15 (13.3%)` | `0/15 (0.0%)` | `16/18 (88.9%)` | `18/18 (100.0%)` |
| Success | Object placed at the requested destination | `1/15 (6.7%)` | `0/15 (0.0%)` | `16/18 (88.9%)` | `18/18 (100.0%)` |
| Success | Complete task | `1/15 (6.7%)` | `0/15 (0.0%)` | `16/18 (88.9%)` | `18/18 (100.0%)` |
| Primary failure | Wrong object or destination selected | `13/15 (86.7%)` | `6/15 (40.0%)` | `0/18 (0.0%)` | `0/18 (0.0%)` |
| Primary failure | Failed to align with or close on the object | `0/15 (0.0%)` | `2/15 (13.3%)` | `0/18 (0.0%)` | `0/18 (0.0%)` |
| Primary failure | Object dropped during transport | `0/15 (0.0%)` | `0/15 (0.0%)` | `2/18 (11.1%)` | `0/18 (0.0%)` |
| Primary failure | Failed to place or release the object | `0/15 (0.0%)` | `0/15 (0.0%)` | `0/18 (0.0%)` | `0/18 (0.0%)` |
| Primary failure | Incorrect stage transition, timeout, or no meaningful action | `1/15 (6.7%)` | `7/15 (46.7%)` | `0/18 (0.0%)` | `0/18 (0.0%)` |
| Recovery | Explicit Plan returned to `grasp` after a detected drop | Not observed | Not observed | `2/2 (100.0%)` | Not applicable |
| Recovery | Valid recovery completed after a drop | Not observed | Not observed | `0/2 (0.0%)` | `1/1 (100.0%)` |

Success and primary-failure metrics use evaluated object cycles as their denominator; recovery rates
are conditional on a transport drop. Primary failures describe the event that ultimately prevented
an object cycle from completing, so the $\pi_{0.5}$ baseline's recovered drop is represented by its
`1/1` recovery rate rather than counted as a drop-caused primary failure. An explicit Plan-transition
rate is not reported for the $\pi_{0.5}$ baseline because the standard policy directly predicts
continuous actions and does not expose AFV2.4-style Plan tokens.

#### Focused AFV2.4 vs. $\pi_{0.5}$ Comparison

| Metric | AFV2.4 | $\pi_{0.5}$ baseline | Difference ($\pi_{0.5}$ - AFV2.4) |
|---|---:|---:|---:|
| Correct source and destination | `18/18 (100.0%)` | `18/18 (100.0%)` | `0.0` pp |
| Complete task | `16/18 (88.9%)` | `18/18 (100.0%)` | `+11.1` pp |
| Drop causing cycle failure | `2/18 (11.1%)` | `0/18 (0.0%)` | `-11.1` pp |
| Valid recovery after a drop | `0/2 (0.0%)` | `1/1 (100.0%)` | Not comparable |

The $\pi_{0.5}$ baseline completed all `18` object cycles, two more than AFV2.4 in this sample, and
completed its one observed recovery. These aggregate outcomes do not capture how quickly each policy
recognizes and responds to an unexpected state change. That distinction is documented qualitatively
below.

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

#### Unexpected-Event Response

Beyond their final success counts, AFV2.4 and the $\pi_{0.5}$ baseline also differ in their transient
response. When a grasp failed, the gripper closed without securing the object, or a held object was
dropped, AFV2.4 reacted to the changed observation promptly. Its explicit predicted Plan returned to
`grasp`, and the continuous action trajectory adjusted toward another grasp attempt with little
visible delay.

The standard $\pi_{0.5}$ baseline reacted more slowly in comparable failures. After an empty grasp,
it could continue along the previously intended transport trajectory for a short distance before
correcting its actions and attempting to recover, although its observed recovery ultimately
succeeded. Manual video inspection estimates that AFV2.4 began the corrective response approximately
`10-15` recorded frames earlier than the $\pi_{0.5}$ baseline. This is an observational estimate
rather than a frame-annotated latency benchmark, so it should be treated as evidence for a follow-up
measurement, not as a precise timing result.

Because the $\pi_{0.5}$ baseline has no explicit Plan output, this comparison does not claim that its
internal plan changed later. It compares the first externally observable corrective action with
AFV2.4's visible Plan transition and action adjustment.

### Analysis

1. **The $\pi_{0.5}$ model family resolves the main semantic failure of the smaller policies.** AFV2.4
   and the $\pi_{0.5}$ baseline both selected the instructed source and destination in `18/18` cycles,
   whereas SmolVLA and AFV2.3 each achieved only `2/15`. AFV2.4's explicit Plan predictions were also
   correct at every observed stage. The shared PaliGemma-based initialization is therefore a major
   contributor to prompt grounding, although this evaluation does not isolate pretraining from the
   other recipe differences.
2. **The $\pi_{0.5}$ baseline has the stronger aggregate result in this evaluation.** It completed
   `18/18` cycles (`100.0%`), two more than AFV2.4's `16/18` (`88.9%`), and successfully recovered
   from its one observed drop. With only `18` evaluated cycles per policy and few recovery events,
   the `11.1` percentage-point difference remains descriptive rather than a general robustness rank.
3. **AFV2.4 responds faster to unexpected outcomes.** Manual rollout inspection shows that failed or
   empty grasps trigger an immediate Plan correction and action adjustment in AFV2.4. The standard
   $\pi_{0.5}$ baseline can continue the stale transport motion before correcting, with AFV2.4's
   observable response estimated to begin `10-15` recorded frames earlier. This suggests that
   explicit per-frame Plan and stage supervision improves closed-loop state-change sensitivity. It
   is complementary to, and does not override, the aggregate outcome advantage of $\pi_{0.5}$.
4. **Response latency and recovery outcome are distinct measurements.** AFV2.4 returned its Plan from
   `put` to `grasp` quickly in both transport-drop cases, but neither recovery completed. The
   $\pi_{0.5}$ baseline reacted more slowly but completed its one observed recovery. On this small
   sample, AFV2.4 detects and responds sooner, while only $\pi_{0.5}$ demonstrates a successful
   physical recovery.
5. **AFV2.4's recovery failures expose missing low-level behaviors.** In one case, the dropped object
   contacted the collection box and the recovery grasp lifted both together. In the other, the
   gripper retained its partially closed transport width rather than reopening fully, causing an
   early false grasp-success transition and a repeated `grasp`/`put` loop.
6. **The main remaining AFV2.4 data gap is recovery, not target recognition.** Future demonstrations
   should include empty grasps, dropped objects, full gripper reset before regrasp,
   object-container contact, and explicit visual evidence of grasp success. A dedicated evaluation
   should separately annotate the failure frame, first corrective-action frame, and final recovery
   outcome for each policy.
7. **This remains a complete-recipe comparison.** Relative to the standard $\pi_{0.5}$ post-training
   baseline, AFV2.4 changes supervision, action-chunk boundaries, and the control interface by adding
   Plan, Box, FAST, and stage labels. The observed latency improvement cannot yet be attributed to a
   single component without ablations.

### Conclusion

AFV2.4 and the standard $\pi_{0.5}$ baseline both provide reliable instruction grounding and high
end-to-end success, far exceeding SmolVLA and AFV2.3 in this setup. The $\pi_{0.5}$ baseline achieved
the stronger aggregate result (`18/18` versus `16/18`) and completed its one observed recovery,
while AFV2.4 showed the faster and more interpretable initial response to failed grasps and drops
through its explicit Plan transition. The next AFV iteration should retain this rapid planning
response while improving gripper reset, grasp confirmation, and physical recovery behavior. A
larger paired evaluation should measure both response latency and recovery completion before drawing
a general robustness ranking.
