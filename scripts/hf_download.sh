#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

usage() {
    cat <<'EOF'
Usage:
  scripts/hf_download.sh REPO_ID [--repo-type model|dataset|space] [--revision REV] [--include PATTERN] [--exclude PATTERN]

Examples:
  scripts/hf_download.sh lerobot/act_aloha_sim_transfer_cube_human
  scripts/hf_download.sh lerobot/pusht --repo-type dataset
  scripts/hf_download.sh openai/clip-vit-base-patch32 --include "*.json" --include "*.safetensors"

Environment:
  HF_ENDPOINT is configured by scripts/env.sh.
  HF_HOME defaults to $STORAGE_ROOT/.cache/huggingface.
  HF_HUB_CACHE defaults to $HF_HOME/hub.
  HF_HUB_DISABLE_XET defaults to 1, which helps with mirrors that cannot serve Xet blobs.

Downloaded repositories use the standard Hugging Face cache layout. They can
later be loaded by repository ID, for example AutoModel.from_pretrained(REPO_ID).
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

REPO_ID="$1"
shift

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

case "$REPO_TYPE" in
    model | dataset | space) ;;
    *)
        echo "Unsupported repo type: $REPO_TYPE" >&2
        usage >&2
        exit 2
        ;;
esac

mkdir -p "$HF_HUB_CACHE"

CMD=(hf download "$REPO_ID" --repo-type "$REPO_TYPE" --cache-dir "$HF_HUB_CACHE")
if [[ -n "$REVISION" ]]; then
    CMD+=(--revision "$REVISION")
fi
CMD+=("${EXTRA_ARGS[@]}")

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "Downloading $REPO_TYPE '$REPO_ID' to the Hugging Face cache"
if ! "${CMD[@]}"; then
    cat >&2 <<EOF

Download failed.
EOF
    exit 1
fi
