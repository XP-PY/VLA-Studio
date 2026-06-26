PRETRAINED=rm65_act_pick_screwdriver_to_fixed_box_close_front_episodes_50_steps_20000_batch_8_chunck_20

cd /home/ubuntu/py/vla_studio/repos/lerobot

lerobot-rollout \
    --config_path=configs/rm65_realsense.yaml \
    --strategy.type=base \
    --inference.type=sync \
    --policy.path=/home/ubuntu/py/vla_studio/runs/train/$PRETRAINED/checkpoints/last/pretrained_model \
    --policy.temporal_ensemble_coeff=0.01 \
    --policy.n_action_steps=1 \
    --device=cuda \
    --fps=30 \
    --duration=20 \
    --task="pick screwdriver and put it in box" \
    --return_to_initial_position=false \
    2>&1 | tee /home/ubuntu/py/vla_studio/logs/rollout_$PRETRAINED.log
