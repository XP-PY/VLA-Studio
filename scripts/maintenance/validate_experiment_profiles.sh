#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_SCRIPT="$PROJECT_ROOT/scripts/env.sh"

SUPPORTED_POLICIES=(
    act
    action_flow_v1
    action_flow_v2
    action_flow_v2_1
    action_flow_v2_2
    action_flow_v2_3
    action_flow_v2_4
    pi0
    smolvla
)

contains_policy() {
    local requested="$1"
    local supported
    for supported in "${SUPPORTED_POLICIES[@]}"; do
        if [[ "$requested" == "$supported" ]]; then
            return 0
        fi
    done
    return 1
}

validate_path_if_local() {
    local label="$1"
    local value="$2"

    if [[ -z "$value" || "$value" != /* ]]; then
        return 0
    fi
    if [[ ! -e "$value" ]]; then
        echo "  ERROR: $label does not exist: $value" >&2
        return 1
    fi
}

profiles=("$PROJECT_ROOT"/configs/experiments/*.env)
if [[ ! -e "${profiles[0]}" ]]; then
    echo "No experiment profiles found." >&2
    exit 1
fi

failures=0
for profile in "${profiles[@]}"; do
    if ! (
        export EXPERIMENT_CONFIG="$profile"
        # shellcheck source=/dev/null
        source "$ENV_SCRIPT"

        echo "$(basename "$profile"): policy=$POLICY_TYPE steps=$STEPS batch=$PER_DEVICE_BATCH_SIZE chunks=${CHUNK_SIZES[*]}"

        contains_policy "$POLICY_TYPE" || {
            echo "  ERROR: unsupported policy type: $POLICY_TYPE" >&2
            exit 1
        }

        if [[ ! -f "$TASK_ROOT/meta/info.json" ]]; then
            echo "  ERROR: dataset metadata is missing: $TASK_ROOT/meta/info.json" >&2
            exit 1
        fi

        policy_root="$STORAGE_ROOT/runs/train/$POLICY_ID"
        if [[ -d "$policy_root" ]]; then
            echo "  existing policy: $policy_root"
        else
            echo "  note: no existing policy directory (valid for a new run)"
        fi

        validate_path_if_local "AFV2.2 tokenizer artifact" "${AFV22_TOKENIZER_ARTIFACT:-}"
        validate_path_if_local "AFV2.3 tokenizer artifact" "${AFV23_TOKENIZER_ARTIFACT:-}"
        validate_path_if_local "AFV2.4 FAST tokenizer" "${AFV24_FAST_TOKENIZER:-}"
    ); then
        failures=$((failures + 1))
    fi
done

if (( failures > 0 )); then
    echo "Experiment profile validation failed: $failures profile(s)." >&2
    exit 1
fi

echo "Experiment profile validation passed: ${#profiles[@]} profile(s)."
