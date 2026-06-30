* [ ] 

# VLA Studio

## What I add

### RM65

RM65 robotic arm class:

```text
repos/lerobot/src/lerobot/robots/rm65
```

And its config file:

```text
repos/lerobot/configs/rm65_realsense.yaml
```

### Dataset record

Dataset recording script:

```text
repos/lerobot/examples/rm65/passive_record.py
```

---

## Installation

### Environment Setup

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
```

### Install LeRobot

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

### Evalution

```bash
./scripts/rollout.sh
```

### Model download

```bash
./scripts/model_download.sh <MODEL_ID>
```

---

