CONFIG_PATH=configs/rm65_left_right_wrist_front_top.yaml
TASK_NAME=rm65_bimanual_farthest_to_closest_assembly_parts_to_center_delete_situation_4
TASK_PROMPT="grasp the visible assembly parts from farthest to closest relative to the front camera, and place each pair in the center"

SUBDIR_NAME=rm65_bimanual_farthest_to_closest_assembly_parts_to_center_subset

python repos/lerobot/examples/rm65/passive_record.py \
    --config-path $CONFIG_PATH \
    --repo-id local/$TASK_NAME \
    --root runs/dataset/$SUBDIR_NAME/$TASK_NAME \
    --task "$TASK_PROMPT" \
    --fps 30 \
    --num-episodes 60 \
    --log-path logs/record_$TASK_NAME.log \
    --io-read-source aloha \
    --control-arm right \
    --teleop-active-io 1 \
    --record-button-io 2 \
    --record-button-mode toggle