#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-root@connect.westc.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-10860}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/vla_studio}"
DRY_RUN="${DRY_RUN:-false}"

SOURCES=(
    "./configs"
    "./models/pi0_base"
    "./models/paligemma-3b-pt-224"
    "./repos/lerobot"
    "./runs/dataset/rm65_single_table_preparation_and_bussing_pi"
    "./scripts"
)

for source in "${SOURCES[@]}"; do
    if [[ ! -e "${PROJECT_ROOT}/${source#./}" ]]; then
        echo "Error: source does not exist: ${PROJECT_ROOT}/${source#./}" >&2
        exit 1
    fi
done

SSH_OPTIONS=(
    -p "$REMOTE_PORT"
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=10
)

RSYNC_OPTIONS=(
    --archive
    --relative
    --human-readable
    --partial
    --append-verify
    --info=progress2,stats2
    --exclude=.git/
    --exclude=.cache/
    --exclude=__pycache__/
    --exclude=.pytest_cache/
    --exclude=.mypy_cache/
    --exclude=.ruff_cache/
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='*.egg-info/'
    --exclude=build/
    --exclude=dist/
)

if [[ "$DRY_RUN" == "true" ]]; then
    RSYNC_OPTIONS+=(--dry-run)
fi

printf -v REMOTE_MKDIR 'mkdir -p -- %q' "$REMOTE_ROOT"

echo "Remote: ${REMOTE_HOST}:${REMOTE_ROOT}"
echo "Creating remote directory..."
ssh "${SSH_OPTIONS[@]}" "$REMOTE_HOST" "$REMOTE_MKDIR"

echo "Uploading project assets..."
cd "$PROJECT_ROOT"
rsync \
    "${RSYNC_OPTIONS[@]}" \
    -e "ssh -p ${REMOTE_PORT} -o ServerAliveInterval=30 -o ServerAliveCountMax=10" \
    "${SOURCES[@]}" \
    "${REMOTE_HOST}:${REMOTE_ROOT}/"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run completed; no files were uploaded."
else
    echo "Upload completed: ${REMOTE_HOST}:${REMOTE_ROOT}"
fi
