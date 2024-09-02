#!/usr/bin/env bash
#SBATCH --mem  5GB
#SBATCH --gres gpu:1
#SBATCH --constrain "galadriel|khazadum|rivendell|arwen|balrog"
#SBATCH --cpus-per-task=8                # Number of CPU cores per task
#SBATCH --time=72:00:00                   # Time limit for each task
#SBATCH --array=0-11%12                  
#SBATCH --mail-type FAIL
#SBATCH --mail-user shaohang@kth.se
#SBATCH --output /Midgard/home/%u/workspace/MTGripperDesign/logs/%J_slurm.out
#SBATCH --error  /Midgard/home/%u/workspace/MTGripperDesign/logs/%J_slurm.err


nvidia-smi
source /Midgard/home/shaohang/mambaforge/etc/profile.d/conda.sh
conda info --envs
conda activate cuda12
export LD_LIBRARY_PATH=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64:$CONDA_PREFIX/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib64:$LD_LIBRARY_PATH
cd /Midgard/home/shaohang/workspace/MTGripperDesign


# Define parameters
robustness_values=(true false)
random_seeds=(1 2 3)
perturbs=(true false)
perturb_sigma=2.5
total_timesteps=4000000
checkpoint_freq=5000

# Calculate indices based on SLURM_ARRAY_TASK_ID
total_robustness=${#robustness_values[@]}
total_perturbs=${#perturbs[@]}
total_seeds=${#random_seeds[@]}

# Calculate indices for each parameter
seed_idx=$(( SLURM_ARRAY_TASK_ID / (total_robustness * total_perturbs) ))
perturb_idx=$(( (SLURM_ARRAY_TASK_ID / total_robustness) % total_perturbs ))
robustness_idx=$(( SLURM_ARRAY_TASK_ID % total_robustness ))

# Assign values based on calculated indices
seed=${random_seeds[$seed_idx]}
perturb=${perturbs[$perturb_idx]}
robustness=${robustness_values[$robustness_idx]}

# Construct the wandb group name
wandb_group_name="sac with robustness: ${robustness}, perturb: ${perturb}"

# Print configuration for logging
echo "Running experiment with:"
echo "  Robustness: $robustness"
echo "  Seed: $seed"
echo "  Perturb: $perturb"
echo "  Perturb Sigma: $perturb_sigma"

# Execute the training command
python3 experiments/train.py \
    --env_id panda \
    --algo sac \
    --using_robustness_reward $robustness \
    --reward_weights 1.0 0.01 0.0 1.0 100.0 0.0 0.0 0.0 \
    --random_seed $seed \
    --total_timesteps $total_timesteps \
    --device auto \
    --render_mode rgb_array \
    --perturb $perturb \
    --perturb_sigma $perturb_sigma \
    --checkpoint_freq $checkpoint_freq \
    --wandb_group_name "$wandb_group_name" \
    --wandb_mode online 
