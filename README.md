# VLA Studio

## What I add in Lerobot

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

Detailed information about `Action Flow` policy can be found in its [doc](docs/action_flow.md)

### Dataset record

Dataset recording script:

```text
repos/lerobot/examples/rm65/passive_record.py
```

Note: this project uses passive dataset recording instead of the standard `lerobot-record` workflow because the master arm directly controls the RM65, but its joint/action data is not available to LeRobot. The recording script therefore saves the RM65's executed joint positions as actions, together with camera observations.

---

## Documentation

- [VLA training logs](docs/README.md)

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

### Training Model

```bash
./scripts/train.sh
```

### Resume Training Model

```bash
./scripts/resume.sh
```

### Evaluation

```bash
./scripts/rollout.sh
```

### Model Download

```bash
./scripts/model_download.sh <MODEL_ID>
```

### Concatenate Multiple datasets

```bash
./scripts/dataset_concatenation.sh
```

---
