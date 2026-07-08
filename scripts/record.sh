CONFIG_PATH=configs/rm65_left_right_wrist_front_top.yaml
TASK_NAME=rm65_bimanual_bottle_right_to_center_box
TASK_PROMPT="place the bottle from the right side into the center boxs"

python repos/lerobot/examples/rm65/passive_record.py \
    --config-path $CONFIG_PATH \
    --repo-id local/$TASK_NAME \
    --root runs/dataset/$TASK_NAME \
    --task "$TASK_PROMPT" \
    --fps 30 \
    --num-episodes 30 \
    --log-path logs/record_$TASK_NAME.log \
    --io-read-source aloha \
    --control-arm right \
    --teleop-active-io 1 \
    --record-button-io 2 \
    --record-button-mode toggle
