# VLA Studio

## LeRobot Extensions

### RM65

RM65 robotic arm class:

```text
repos/lerobot/src/lerobot/robots/rm65
```

And its config file for single and bimanual robotic arm:

```text
configs/rm65_wrist_front.yaml
configs/rm65_left_right_wrist_front_top.yaml
```

### Policy

#### Action Flow

Detailed information about the Action Flow policies is available in the
[Action Flow training log](docs/action_flow.md).

### Dataset record

Dataset recording script:

```text
repos/lerobot/examples/rm65/passive_record.py
```

Note: this project uses passive dataset recording instead of the standard `lerobot-record` workflow because the master arm directly controls the RM65, but its joint/action data is not available to LeRobot. The recording script therefore saves the RM65's executed joint positions as actions, together with camera observations.

---

## Documentation

- [VLA training logs](docs/README.md)
- [Policy comparison](docs/evaluation.md)
- [Project structure and storage maintenance](docs/project_structure.md)

## Project Layout

```text
configs/                 RM65 configurations and the curated dataset registry
configs/experiments/     reproducible policy training profiles
configs/recordings/      direct source-dataset recording profiles
docs/                    experiment logs, evaluations, and project notes
examples/                inspection tools and data-annotation programs
artifacts/               portable tokenizer and policy artifacts
repos/lerobot/           the modified LeRobot source tree
scripts/                 stable recording, training, and evaluation entry points
scripts/maintenance/     inventory and checkpoint-retention tools
runs/                    datasets, checkpoints, rollouts, and generated inspection files
logs/                    runtime and training logs
```

`runs`, `logs`, and `.cache` may be symbolic links to the external storage disk. Generated outputs
belong in those directories rather than in `examples` or the repository root.

---

## Installation

### Environment Setup

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
```

### Install LeRobot

Before installing LeRobot with all optional dependencies, install the required build tools and media libraries on Ubuntu:

```bash
sudo apt update
sudo apt install -y \
  cmake \
  build-essential \
  python3-dev \
  pkg-config \
  libavformat-dev \
  libavcodec-dev \
  libavdevice-dev \
  libavutil-dev \
  libswscale-dev \
  libswresample-dev \
  libavfilter-dev
```

Then install LeRobot:

```bash
cd repos/lerobot
pip install -e ".[all]"
```

---

## Quick start

### Recording Dataset

```bash
./scripts/record.sh
```

The default adds one `place_one_vegetable` episode. Select another source subset independently of
the training profile:

```bash
RECORD_CONFIG=configs/recordings/single_table_bus_table.env ./scripts/record.sh
```

### Training Model

```bash
./scripts/train.sh
```

The default profile is AFV2.3 without knowledge insulation. Select another reproducible profile
without editing `env.sh` or `train.sh`:

```bash
EXPERIMENT_CONFIG=configs/experiments/smolvla_baseline.env ./scripts/train.sh
```

### Resume Training Model

```bash
./scripts/resume.sh
```

### Evaluation

```bash
./scripts/rollout.sh
```

The rollout command automatically selects the policy's supported inference-step field and the
robot configuration's single or bimanual mode. Inspect the resolved command without hardware:

```bash
ROLLOUT_DRY_RUN=true ./scripts/rollout.sh
```

### Hugging Face Download

```bash
./scripts/hf_download.sh <MODEL_ID>
```

### Concatenate Multiple datasets

```bash
./scripts/dataset_concatenation.sh
```

### Inspect Stored Assets

```bash
python scripts/maintenance/validate_datasets.py
python scripts/maintenance/validate_evaluations.py
./scripts/maintenance/validate_experiment_profiles.sh
./scripts/maintenance/validate_recording_profiles.sh
python scripts/maintenance/inventory_runs.py
```

The generated inventory is written to `runs/manifests`. Checkpoint cleanup is dry-run by default:

```bash
python scripts/maintenance/prune_checkpoints.py
```

---
