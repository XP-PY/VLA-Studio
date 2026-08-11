#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${INSID3_PYTHON:-/home/ubuntu/miniconda3/envs/insid3/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${WORKSPACE}/runs/dataset/rm65_single_table_preparation_and_bussing}"
PLANNING_ROOT="${PLANNING_ROOT:-${WORKSPACE}/runs/planning/rm65_single_table_preparation_and_bussing}"
REFERENCE_MANIFEST="${REFERENCE_MANIFEST:-${WORKSPACE}/artifacts/insid3/references/rm65_tabletop/references.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE}/runs/insid3/rm65_single_table_preparation_and_bussing}"
EPISODES="${EPISODES:-0}"
MAX_ITEMS="${MAX_ITEMS:-8}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "INSID3 Python is unavailable: ${PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "Dataset is unavailable: ${DATASET_ROOT}" >&2
    echo "Mount BW_500G before running this script." >&2
    exit 1
fi
if [[ ! -f "${PLANNING_ROOT}/review/segments.parquet" ]]; then
    echo "Reviewed planning segments are unavailable: ${PLANNING_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${REFERENCE_MANIFEST}" ]]; then
    echo "INSID3 reference manifest is unavailable: ${REFERENCE_MANIFEST}" >&2
    exit 1
fi

unset ALL_PROXY all_proxy
export HF_HOME="${HF_HOME:-${WORKSPACE}/artifacts/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

COMMAND=(
    "${PYTHON}" -u "${WORKSPACE}/examples/action_flow_v2/annotate_insid3_boxes.py"
    --dataset-root "${DATASET_ROOT}"
    --planning-root "${PLANNING_ROOT}"
    --reference-manifest "${REFERENCE_MANIFEST}"
    --output-root "${OUTPUT_ROOT}"
    --episodes "${EPISODES}"
    --cameras top wrist
    --image-size 512
    --resume
)

if (( MAX_ITEMS > 0 )); then
    COMMAND+=(--max-items "${MAX_ITEMS}")
fi

printf 'Episodes: %s\nMaximum items: %s\nOutput: %s\n' "${EPISODES}" "${MAX_ITEMS}" "${OUTPUT_ROOT}"
"${COMMAND[@]}"
