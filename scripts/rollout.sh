#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

if [ -d "$PROJECT_ROOT/runs/rollout_dataset/rollout_$POLICY_ID" ]; then
    RESUME=true
else
    RESUME=false
fi

lerobot-rollout \
    --config_path="$ROBOT_CONFIG_PATH" \
    --strategy.type=highlight \
    --inference.type=sync \
    --robot.left_arm.canfd_follow=$ROLLOUT_CANFD_FOLLOW \
    --robot.right_arm.canfd_follow=$ROLLOUT_CANFD_FOLLOW \
    --robot.left_arm.max_relative_target=1000.0 \
    --robot.right_arm.max_relative_target=1000.0 \
    --policy.path="$PROJECT_ROOT/runs/train/$POLICY_ID/checkpoints/last/pretrained_model" \
    --policy.temporal_ensemble_coeff=0.01 \
    --policy.n_action_steps=1 \
    --dataset.repo_id="local/rollout_$POLICY_ID" \
    --dataset.root="$PROJECT_ROOT/runs/rollout_dataset/rollout_$POLICY_ID" \
    --dataset.single_task="$TASK_PROMPT" \
    --dataset.push_to_hub=false \
    --resume=$RESUME \
    --device=cuda \
    --fps=$ROLLOUT_FPS \
    --duration=$ROLLOUT_DURATION \
    --task="$TASK_PROMPT" \
    --return_to_initial_position=false \
    2>&1 | tee ./logs/rollout_$POLICY_ID.log
