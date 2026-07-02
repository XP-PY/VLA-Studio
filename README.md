# VLA Studio

## What I add in Lerobot

### RM65

RM65 robotic arm class:

```text
repos/lerobot/src/lerobot/robots/rm65
```

And its config file:

```text
configs/rm65_realsense.yaml
```

### Dataset record

Dataset recording script:

```text
repos/lerobot/examples/rm65/passive_record.py
```

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

### Recording dataset

```bash
./scripts/record.sh
```

### Training model

```bash
./scripts/train.sh
```

### Evaluation

```bash
./scripts/rollout.sh
```

### Model download

```bash
./scripts/model_download.sh <MODEL_ID>
```

---
