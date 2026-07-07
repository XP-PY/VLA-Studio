## ====== ACT ======

PRETRAINED=rm65_action_flow_v1_bimanual_grasp_and_place_episodes_20_steps_20000_batch_16_chunk_50
TASK_PROMPT="grasp two assembly parts and place them in the center"

TASK_NAME=rollout_$PRETRAINED
FOLDER_PATH="./runs/rollout_dataset/$TASK_NAME"

if [ -d "$FOLDER_PATH" ]; then
    RESUME=true
else
    RESUME=false
fi

lerobot-rollout \
    --config_path=configs/rm65_left_right_wrist_front_top.yaml \
    --strategy.type=highlight \
    --inference.type=sync \
    --robot.left_arm.canfd_follow=false \
    --robot.right_arm.canfd_follow=false \
    --robot.left_arm.max_relative_target=1000.0 \
    --robot.right_arm.max_relative_target=1000.0 \
    --policy.path=./runs/train/$PRETRAINED/checkpoints/last/pretrained_model \
    --policy.n_action_steps=20 \
    --dataset.repo_id=local/$TASK_NAME \
    --dataset.root=$FOLDER_PATH \
    --dataset.single_task="$TASK_PROMPT" \
    --dataset.push_to_hub=false \
    --resume=$RESUME \
    --device=cuda \
    --fps=10 \
    --duration=120 \
    --task="$TASK_PROMPT" \
    --return_to_initial_position=false \
    2>&1 | tee ./logs/rollout_$PRETRAINED.log

# lerobot-rollout \
#     --config_path=configs/rm65_left_right_wrist_front_top.yaml \
#     --strategy.type=highlight \
#     --inference.type=sync \
#     --robot.left_arm.canfd_follow=false \
#     --robot.right_arm.canfd_follow=false \
#     --robot.left_arm.max_relative_target=1000.0 \
#     --robot.right_arm.max_relative_target=1000.0 \
#     --policy.path=./runs/train/$PRETRAINED/checkpoints/last/pretrained_model \
#     --policy.temporal_ensemble_coeff=0.01 \
#     --policy.n_action_steps=1 \
#     --dataset.repo_id=local/$TASK_NAME \
#     --dataset.root=./runs/rollout_dataset/$TASK_NAME \
#     --dataset.single_task="$TASK_PROMPT" \
#     --dataset.push_to_hub=false \
#     --resume=true \
#     --device=cuda \
#     --fps=10 \
#     --duration=120 \
#     --task="$TASK_PROMPT" \
#     --return_to_initial_position=false \
#     2>&1 | tee ./logs/rollout_$PRETRAINED.log
