#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

usage() {
    cat <<'EOF'
Usage:
  scripts/model_download.sh REPO_ID [OUTPUT_DIR] [--repo-type model|dataset|space] [--revision REV] [--include PATTERN] [--exclude PATTERN]

Examples:
  scripts/model_download.sh lerobot/act_aloha_sim_transfer_cube_human
  scripts/model_download.sh microsoft/resnet-50 ./models/resnet-50
  scripts/model_download.sh openai/clip-vit-base-patch32 ./models/clip --include "*.json" --include "*.safetensors"

Environment:
  HF_ENDPOINT defaults to https://hf-mirror.com
  HF_HOME defaults to $STORAGE_ROOT/cache/huggingface
  HF_HUB_DISABLE_XET defaults to 1, which helps with mirrors that cannot serve Xet blobs.
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

REPO_ID="$1"
shift

OUTPUT_DIR="${1:-${STORAGE_ROOT}/models/${REPO_ID##*/}}"
if [[ $# -gt 0 && "${1:0:2}" != "--" ]]; then
    shift
fi

REPO_TYPE="model"
REVISION=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-type)
            REPO_TYPE="$2"
            shift 2
            ;;
        --revision)
            REVISION="$2"
            shift 2
            ;;
        --include)
            EXTRA_ARGS+=(--include "$2")
            shift 2
            ;;
        --exclude)
            EXTRA_ARGS+=(--exclude "$2")
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

CMD=(hf download "$REPO_ID" --repo-type "$REPO_TYPE" --local-dir "$OUTPUT_DIR")
if [[ -n "$REVISION" ]]; then
    CMD+=(--revision "$REVISION")
fi
CMD+=("${EXTRA_ARGS[@]}")

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "Downloading $REPO_TYPE '$REPO_ID' to $OUTPUT_DIR"
if ! "${CMD[@]}"; then
    cat >&2 <<EOF

Download failed.
EOF
    exit 1
fi
