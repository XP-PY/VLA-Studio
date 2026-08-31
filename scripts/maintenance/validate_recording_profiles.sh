#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RECORD_ENV="$PROJECT_ROOT/scripts/record_env.sh"
DATASET_REGISTRY="$PROJECT_ROOT/configs/datasets.yaml"

profiles=("$PROJECT_ROOT"/configs/recordings/*.env)
if [[ ! -e "${profiles[0]}" ]]; then
    echo "No recording profiles found." >&2
    exit 1
fi

failures=0
for profile in "${profiles[@]}"; do
    if ! (
        export RECORD_CONFIG="$profile"
        # shellcheck source=/dev/null
        source "$RECORD_ENV"

        echo "$(basename "$profile"): output=$RECORD_OUTPUT_ID episodes=$RECORD_NUM_EPISODES fps=$RECORD_FPS"
        if [[ ! -f "$RECORD_ROBOT_CONFIG_PATH" ]]; then
            echo "  ERROR: robot config does not exist: $RECORD_ROBOT_CONFIG_PATH" >&2
            exit 1
        fi

        registry_block=$(
            awk -v key="  $RECORD_REGISTRY_KEY:" '
                $0 == key { found = 1; next }
                found && /^  [A-Za-z0-9_]+:/ { exit }
                found { print }
            ' "$DATASET_REGISTRY"
        )
        if [[ -z "$registry_block" ]]; then
            echo "  ERROR: dataset registry key is missing: $RECORD_REGISTRY_KEY" >&2
            exit 1
        fi
        if ! grep -q '^[[:space:]]*role: source_subset$' <<<"$registry_block"; then
            echo "  ERROR: recording target is not registered as source_subset: $RECORD_REGISTRY_KEY" >&2
            exit 1
        fi

        registered_path=$(sed -nE 's/^[[:space:]]*path:[[:space:]]*(.+)$/\1/p' <<<"$registry_block" | head -n 1)
        expected_path="${RECORD_OUTPUT_ROOT#"$STORAGE_ROOT/runs/dataset/"}"
        if [[ "$registered_path" != "$expected_path" ]]; then
            echo "  ERROR: registered path differs from recording output" >&2
            echo "    registry: $registered_path" >&2
            echo "    output:   $expected_path" >&2
            exit 1
        fi

        if [[ -f "$RECORD_OUTPUT_ROOT/meta/info.json" ]]; then
            echo "  append target: $RECORD_OUTPUT_ROOT"
        else
            echo "  note: output dataset does not exist yet and will be created"
        fi
    ); then
        failures=$((failures + 1))
    fi
done

if (( failures > 0 )); then
    echo "Recording profile validation failed: $failures profile(s)." >&2
    exit 1
fi

echo "Recording profile validation passed: ${#profiles[@]} profile(s)."
