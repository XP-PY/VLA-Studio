PRETRAINED=rm65_act_pick_screwdriver_to_fixed_box_closer_front_episodes_250_steps_40000_batch_32_chunck_50

lerobot-rollout \
    --config_path=configs/rm65_realsense.yaml \
    --strategy.type=base \
    --inference.type=sync \
    --policy.path=./runs/train/$PRETRAINED/checkpoints/last/pretrained_model \
    --policy.temporal_ensemble_coeff=0.01 \
    --policy.n_action_steps=1 \
    --device=cuda \
    --fps=20 \
    --duration=20 \
    --task="pick screwdriver and put it in box" \
    --return_to_initial_position=false \
    2>&1 | tee ./logs/rollout_$PRETRAINED.log
