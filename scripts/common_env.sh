#!/usr/bin/env bash

# Shared machine-level environment. Training and recording task parameters do
# not belong here; they are loaded by their respective profile loaders.

unset http_proxy
unset https_proxy
unset no_proxy
unset NO_PROXY
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Override STORAGE_ROOT before sourcing this file on another machine.
export STORAGE_ROOT="${STORAGE_ROOT:-/media/ubuntu/BW_500G/vla_studio_storage}"
if [[ ! -d "$STORAGE_ROOT" ]]; then
    export STORAGE_ROOT="$PROJECT_ROOT"
fi

export TORCH_HOME="$STORAGE_ROOT/.cache/torch"
export HF_HOME="$STORAGE_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_LEROBOT_HOME="$STORAGE_ROOT/.cache/lerobot"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export YOLOE_CACHE_DIR="$STORAGE_ROOT/.cache/yoloe"
export YOLOE_WEIGHT_DIR="$YOLOE_CACHE_DIR/weights"
export YOLOE_DATASET_DIR="$YOLOE_CACHE_DIR/datasets"
export YOLOE_RUNS_DIR="$YOLOE_CACHE_DIR/runs"

export PYTHONPATH="$PROJECT_ROOT/repos/lerobot/src:${PYTHONPATH:-}"

export ROBOT_CONFIG_PATH="${ROBOT_CONFIG_PATH:-configs/rm65_wrist_top.yaml}"
