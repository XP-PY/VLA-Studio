# Experiment Profiles

Each profile contains one reproducible training/evaluation configuration. Common paths, cache,
robot, rollout, resume, and W&B settings remain in `scripts/env.sh`.

The default profile is `action_flow_v2_3_no_ki.env`. Select another profile without editing scripts:

```bash
EXPERIMENT_CONFIG=configs/experiments/smolvla_baseline.env ./scripts/train.sh
```

Environment variables such as `STORAGE_ROOT`, `PER_DEVICE_BATCH_SIZE`, `NUM_GPUS`, and `GPU_IDS`
may be set before the command. Use `CHUNK_SIZE=30` to replace a profile's chunk-size sweep with one
value.

Validate all profiles, registered dataset metadata, and local tokenizer artifacts without starting
training:

```bash
./scripts/maintenance/validate_experiment_profiles.sh
```

| Profile | Purpose |
|---|---|
| `action_flow_v2_3_no_ki.env` | Current AFV2.3 no-KI comparison policy. |
| `action_flow_v2_3_ki.env` | Knowledge-insulated AFV2.3 comparison policy. |
| `action_flow_v2_4.env` | PI0.5-based AFV2.4 cloud training. |
| `smolvla_baseline.env` | Robot-only SmolVLA comparison baseline. |
| `action_flow_v2_2_boxes.env` | Historical AFV2.2 structured-box experiment. |
| `action_flow_v2_1_dense.env` | Historical AFV2.1 dense-segmentation experiment. |
| `act_bimanual_assembly_sweep.env` | Historical ACT chunk-size sweep. |
| `action_flow_v1_bimanual_assembly_sweep.env` | Historical Action Flow V1 sweep. |

No PI0 profile is provided because its earlier converted `_pi` dataset is no longer present in the
current dataset registry. The legacy `pi0` branch remains in `scripts/train.sh` for reproducibility.
