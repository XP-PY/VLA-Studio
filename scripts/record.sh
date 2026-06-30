TASK_NAME=rm65_aloha_pick_screwdriver_to_fixed_box_closer_front
TASK_PROMPT="pick screwdriver and put it in box"

python repos/lerobot/examples/rm65/passive_record.py \
    --config-path configs/rm65_realsense.yaml \
    --repo-id local/$TASK_NAME \
    --root ./runs/dataset/$TASK_NAME \
    --task "$TASK_PROMPT" \
    --fps 30 \
    --episode-time-s 30 \
    --num-episodes 50 \
    --teleop-active-io 1 \
    --record-button-io 2 \
    --record-button-mode toggle \
    --io-read-source aloha \
    --aloha-target-ip 192.168.1.100 \
    --log-path ./logs/record_$TASK_NAME.log
