HF_HOME=./.cache/huggingface \
HF_DATASETS_CACHE=./.cache/huggingface/datasets \
PYTHONPATH=repos/lerobot/src \

python -m lerobot.scripts.lerobot_edit_dataset \
    --new_repo_id local/rm65_bimanual_bottle_multitask \
    --new_root runs/dataset/rm65_bimanual_bottle_multitask \
    --operation.type merge \
    --operation.repo_ids "['local/rm65_bimanual_bottle_left_to_center_box', 'local/rm65_bimanual_bottle_right_to_center_box', 'local/rm65_bimanual_bottle_left_to_right_box_handover', 'local/rm65_bimanual_bottle_right_to_left_box_handover']" \
    --operation.roots "['runs/dataset/rm65_bimanual_bottle_left_to_center_box', 'runs/dataset/rm65_bimanual_bottle_right_to_center_box', 'runs/dataset/rm65_bimanual_bottle_left_to_right_box_handover', 'runs/dataset/rm65_bimanual_bottle_right_to_left_box_handover']" \
    --operation.concatenate_videos false \
    --operation.concatenate_data false \
    --push_to_hub false