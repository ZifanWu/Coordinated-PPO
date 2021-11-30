#!/bin/sh
env="StarCraft2"
map="3s_vs_3z"
algo="rmappo"
exp="ex"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 \
    python train/train_smac.py --n_training_threads 23 --n_rollout_threads 8  --num_env_steps 2000000 --ppo_epoch 15 --num_mini_batch 1 --episode_length 400 --gamma 0.99 --gae_lambda 0.90 \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map}\
    --use_value_active_masks --add_center_xy --use_state_agent --add_agent_id --use_adv --use_recurrent_policy \
    --use_eval
done
