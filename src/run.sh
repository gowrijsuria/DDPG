#!/bin/bash
#SBATCH -A gowri
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
##SBATCH --time=4-00:00:00
#SBATCH --mincpus=2
#SBATCH --mail-type=END
#SBATCH --nodelist=gnode28

module add cuda/10.2
module add cudnn/7.6.5-cuda-10.2

python3 main.py --save_path /ssd_scratch/cvit/gowri/DDPG/ --experiment nov13_batch12_episodes500_steps60_coll0_01_maxForce_40_noforward_thr_2 --max_steps 60 \
        --reward_average_window 1 --num_episodes 500 --averageRewardThreshold 4500 --threshold_dist 2 \
        --batch_size 12 --targetVel 5 --maxForce 50 --reward_collision -0.01 --replay_memory_size 1000

