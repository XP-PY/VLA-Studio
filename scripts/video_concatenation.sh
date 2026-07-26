#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

ROLLOUT_DIR="runs/rollout_dataset/rollout_${POLICY_ID}_multi_prompt"
VIDEO_DIR="${ROLLOUT_DIR}/videos"

POLICY="action_flow"
CHUNK_ID="chunk-000"
FILE_ID="file-010"
# OUTPUT="runs/temp/${FILE_ID}.mp4"
OUTPUT="docs/assets/${POLICY}/v2_ob_1_log_1_rollout_19.mp4"


# Find all observation images folders
CAM_DIRS=($(find "$VIDEO_DIR" -maxdepth 1 -type d -name "observation.images.*" | sort))
# CAM_DIRS=($(find "$VIDEO_DIR" -maxdepth 1 -type d \( -name "observation.images.*" -o -name "heatmap.images.*" \) | sort))

if [ ${#CAM_DIRS[@]} -eq 0 ]; then
    echo "No observation.images.* folders found!"
    exit 1
fi


INPUTS=()
FILTERS=()

INDEX=0

echo "Found cameras:"

for cam in "${CAM_DIRS[@]}"; do
    VIDEO_FILE="${cam}/${CHUNK_ID}/${FILE_ID}.mp4"

    if [ -f "$VIDEO_FILE" ]; then
        echo "  $VIDEO_FILE"
        INPUTS+=("-i" "$VIDEO_FILE")
        FILTERS+=("[$INDEX:v]")
        INDEX=$((INDEX+1))
    else
        echo "Missing: $VIDEO_FILE"
    fi
done


NUM_VIDEOS=${#FILTERS[@]}

if [ "$NUM_VIDEOS" -eq 0 ]; then
    echo "No videos found!"
    exit 1
fi


echo "Total videos: $NUM_VIDEOS"


# Generate ffmpeg layout
if [ "$NUM_VIDEOS" -eq 1 ]; then

    FILTER_COMPLEX="${FILTERS[0]}null[out]"

elif [ "$NUM_VIDEOS" -eq 2 ]; then

    FILTER_COMPLEX="${FILTERS[0]}${FILTERS[1]}hstack=inputs=2[out]"

elif [ "$NUM_VIDEOS" -eq 3 ]; then

    FILTER_COMPLEX="${FILTERS[0]}${FILTERS[1]}hstack=inputs=2[top]; \
${FILTERS[2]}pad=iw:ih*2[bottom]; \
[top][bottom]vstack=inputs=2[out]"

else

    # 4 or more: use xstack automatically
    INPUT_LABELS=""
    for f in "${FILTERS[@]}"; do
        INPUT_LABELS+="$f"
    done

    FILTER_COMPLEX="${INPUT_LABELS}xstack=inputs=${NUM_VIDEOS}:layout=0_0|w0_0|0_h0|w0_h0[out]"

fi


echo "Running ffmpeg..."

ffmpeg \
    "${INPUTS[@]}" \
    -filter_complex "$FILTER_COMPLEX" \
    -map "[out]" \
    -c:v libx264 \
    -preset fast \
    -crf 23 \
    "$OUTPUT"