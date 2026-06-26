# VLA Studio

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