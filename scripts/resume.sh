#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

TRAINING_ID="rm65_pi0_single_table_preparation_and_bussing_episodes_204_steps_30000_batch_16_chunk_50"
SOURCE_RUN_DIR="${STORAGE_ROOT}/runs/train/${TRAINING_ID}"
RESUME_OUTPUT_DIR="${SOURCE_RUN_DIR}"
LOG_PATH="${PROJECT_ROOT}/logs/resume_${TRAINING_ID}.log"

# LeRobot skips the checkpoints/last symlink when the exFAT drive rejects it.
# Numbered checkpoints remain complete and can be resumed directly.
if [ ! -w "${RESUME_OUTPUT_DIR}" ]; then
    echo "Resume output directory is not writable: ${RESUME_OUTPUT_DIR}"
    echo "Remount or repair BW_500G as writable before resuming."
    exit 1
fi

latest_step=-1
latest_checkpoint=""
for checkpoints_dir in "${SOURCE_RUN_DIR}/checkpoints"; do
    [ -d "${checkpoints_dir}" ] || continue
    for checkpoint in "${checkpoints_dir}"/[0-9][0-9][0-9][0-9][0-9][0-9]; do
        [ -d "${checkpoint}" ] || continue
        [ -f "${checkpoint}/pretrained_model/train_config.json" ] || continue
        [ -f "${checkpoint}/training_state/training_step.json" ] || continue

        step_name="$(basename "${checkpoint}")"
        step=$((10#${step_name}))
        if [ "${step}" -gt "${latest_step}" ]; then
            latest_step="${step}"
            latest_checkpoint="${checkpoint}"
        fi
    done
done

if [ -z "${latest_checkpoint}" ]; then
    echo "No complete numeric checkpoint found under:"
    echo "  ${SOURCE_RUN_DIR}/checkpoints"
    exit 1
fi

CONFIG_PATH="${latest_checkpoint}/pretrained_model/train_config.json"
mkdir -p "$(dirname "${LOG_PATH}")"

echo "Resuming ${TRAINING_ID} from step ${latest_step}."
echo "Checkpoint: ${latest_checkpoint}"
echo "New checkpoints: ${RESUME_OUTPUT_DIR}"
echo "Log: ${LOG_PATH}"

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    --config_path="${CONFIG_PATH}" \
    --resume=true \
    --output_dir="${RESUME_OUTPUT_DIR}" \
    2>&1 | tee -a "${LOG_PATH}"
