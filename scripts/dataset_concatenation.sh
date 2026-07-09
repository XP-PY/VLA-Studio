HF_HOME=./.cache/huggingface \
HF_DATASETS_CACHE=./.cache/huggingface/datasets \
PYTHONPATH=repos/lerobot/src \

python -m lerobot.scripts.lerobot_edit_dataset \
    --new_repo_id local/rm65_bimanual_farthest_to_closest_assembly_parts_to_center \
    --new_root runs/dataset/rm65_bimanual_farthest_to_closest_assembly_parts_to_center \
    --operation.type merge \
    --operation.repo_ids "['local/rm65_bimanual_farthest_to_closest_assembly_parts_to_center_delete_situation_1', 'local/rm65_bimanual_farthest_to_closest_assembly_parts_to_center_delete_situation_2', 'local/rm65_bimanual_farthest_to_closest_assembly_parts_to_center_delete_situation_3', 'local/rm65_bimanual_farthest_to_closest_assembly_parts_to_center_delete_situation_4']" \
    --operation.roots "['runs/dataset/rm65_bimanual_farthest_to_closest_assembly_parts_to_center_subset/rm65_bimanual_farthest_to_closest_assembly_parts_to_center_delete_situation_1', 'runs/dataset/rm65_bimanual_farthest_to_closest_assembly_parts_to_center_subset/rm65_bimanual_farthest_to_closest_assembly_parts_to_center_delete_situation_2', 'runs/dataset/rm65_bimanual_farthest_to_closest_assembly_parts_to_center_subset/rm65_bimanual_farthest_to_closest_assembly_parts_to_center_delete_situation_3', 'runs/dataset/rm65_bimanual_farthest_to_closest_assembly_parts_to_center_subset/rm65_bimanual_farthest_to_closest_assembly_parts_to_center_delete_situation_4']" \
    --operation.concatenate_videos false \
    --operation.concatenate_data false \
    --push_to_hub false