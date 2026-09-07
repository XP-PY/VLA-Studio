# Script Index

The root-level scripts are stable command-line entry points. They remain in place so existing notes
and workflows do not break.

## Daily Workflows

| Script | Role |
|---|---|
| `common_env.sh` | Shared storage, cache, Python path, and robot defaults. |
| `env.sh` | Shared storage, cache, robot, rollout, and experiment-profile loader. |
| `record_env.sh` | Recording-profile loader and output-path resolver. |
| `record.sh` | Record RM65 demonstrations into a selected direct source subset. |
| `train.sh` | Train the policy selected by an experiment profile. |
| `resume.sh` | Resume a saved training run. |
| `rollout.sh` | Run a policy-aware single- or bimanual rollout. |
| `evaluator_multi_prompts.sh` | Interactive multi-prompt evaluation for supported policies. |

## Dataset and Media

| Script | Role |
|---|---|
| `dataset_concatenation.sh` | Combine compatible LeRobot datasets. |
| `dataset_convert_to_pi.sh` | Build the relative-action dataset variant used by PI policies. |
| `recompute_dataset_stats.py` | Recompute dataset statistics after conversion or concatenation. |
| `remove_feature.sh` | Remove an unwanted dataset feature. |
| `video_concatenation.sh` | Concatenate selected videos for review. |
| `insid3_boxes.sh` | Run the INSID3 box-annotation workflow. |

## Experiments and Infrastructure

| Script | Role |
|---|---|
| `overfit_afv23_fast.sh` | AFV2.3 FAST-token overfitting diagnostic. |
| `train_afv23_fast_balanced.sh` | Balanced AFV2.3 FAST experiment. |
| `hf_download.sh` | Download a Hugging Face repository into the shared cache. |
| `upload_to_server.sh` | Upload selected project assets to a training server. |
| `test.sh` | Local scratch test entry point. |

## Experiment Profiles

Training parameters are stored in `configs/experiments/*.env`; the command dispatcher remains in
`train.sh`. The default profile is AFV2.3 without knowledge insulation:

```bash
./scripts/train.sh
```

Select a different experiment without editing either script:

```bash
EXPERIMENT_CONFIG=configs/experiments/smolvla_baseline.env ./scripts/train.sh
```

Use `CHUNK_SIZE=30` to run one value from a sweep. Machine-specific values such as `STORAGE_ROOT`,
`GPU_IDS`, `NUM_GPUS`, and `PER_DEVICE_BATCH_SIZE` can be supplied before the command. The profile
catalog and field ownership rules are documented in `configs/experiments/README.md`.

## Recording Profiles

Recording settings are independent from training experiments. The default appends one episode to
the `place_one_vegetable` source subset:

```bash
./scripts/record.sh
```

Select another direct source subset and optionally override its task prompt:

```bash
RECORD_CONFIG=configs/recordings/single_table_clear_specified_plate.env \
RECORD_TASK="Put the vegetable from the blue plate into the collection box." \
RECORD_NUM_EPISODES=3 \
./scripts/record.sh
```

`RECORD_NUM_EPISODES` means new episodes in the current invocation. See
`configs/recordings/README.md` for the profile catalog and dry-run command.

## Standard Rollout

`rollout.sh` selects the latest checkpoint from the active experiment profile. It reads the saved
policy config and passes `num_inference_steps`, `num_steps`, or neither, according to the policy's
actual fields. It also reads the RM65 YAML mode and builds single- or bimanual robot arguments.

Inspect a rollout without loading a policy or connecting hardware:

```bash
ROLLOUT_DRY_RUN=true ./scripts/rollout.sh
```

Select AFV2.4 and override its inference steps for one rollout:

```bash
EXPERIMENT_CONFIG=configs/experiments/action_flow_v2_4.env \
ROLLOUT_NUM_STEPS=4 \
./scripts/rollout.sh
```

For a bimanual profile, also select its matching robot configuration:

```bash
EXPERIMENT_CONFIG=configs/experiments/act_bimanual_assembly_sweep.env \
ROBOT_CONFIG_PATH=configs/rm65_left_right_wrist_front_top.yaml \
./scripts/rollout.sh
```

## Maintenance

| Script | Role |
|---|---|
| `maintenance/inventory_runs.py` | Generate JSON and Markdown inventories for runs and datasets. |
| `maintenance/validate_datasets.py` | Validate `configs/datasets.yaml` against stored metadata and lineage. |
| `maintenance/validate_evaluations.py` | Validate rollout and offline-evaluation groups and path coverage. |
| `maintenance/validate_experiment_profiles.sh` | Validate every experiment profile and its local dataset/tokenizer dependencies. |
| `maintenance/validate_recording_profiles.sh` | Validate recording targets and robot configurations without connecting hardware. |
| `maintenance/prune_checkpoints.py` | Apply the explicit checkpoint-retention policy; dry-run by default. |
| `maintenance/checkpoint_retention.json` | Select resumable runs and the default historical policy. |
