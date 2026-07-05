# Action Flow Training Log

*Last updated: July 5, 2026*

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
