CONFIG_PATH=configs/rm65_left_right_wrist_front_top.yaml
TASK_NAME=rm65_bimanual_bottle_left_to_right_box_handover
TASK_PROMPT="transfer the bottle from the left side to the right arm and place it in the right box"

python repos/lerobot/examples/rm65/passive_record.py \
    --config-path $CONFIG_PATH \
    --repo-id local/$TASK_NAME \
    --root runs/dataset/$TASK_NAME \
    --task "$TASK_PROMPT" \
    --fps 30 \
    --num-episodes 50 \
    --io-read-source aloha \
    --control-arm right \
    --teleop-active-io 1 \
    --record-button-io 2 \
    --record-button-mode toggle
