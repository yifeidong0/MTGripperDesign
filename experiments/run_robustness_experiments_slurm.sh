#!/bin/bash

#SBATCH --job-name=codesign_panda_experiments
#SBATCH --account=yifeid                # Replace with your account name
#SBATCH --partition=alvis                # Partition to use
#SBATCH --nodes=4                        # Number of nodes (1 node is sufficient)
#SBATCH --gpus-per-node=A40:1           # Number and type of GPU (1 A100 per task)
#SBATCH --cpus-per-task=8                # Number of CPU cores per task
#SBATCH --mem=16G                        # Memory per task (adjust if necessary)
#SBATCH --time=48:00:00                   # Time limit for each task
#SBATCH --array=0-23%12                   # Array jobs: 24 tasks, run 12 simultaneously
#SBATCH --output=logs/%x_%A_%a.out       # Output log
#SBATCH --error=logs/%x_%A_%a.err        # Error log

# Load necessary modules
module load Python/3.8.0
module load CUDA/11.3

# Define parameters
robustness_values=(true false)
random_seeds=(1 2 3 4 5 6)
perturbs=(true false)
perturb_sigma=2.5
total_timesteps=4000000

# Calculate indices based on SLURM_ARRAY_TASK_ID
robustness_idx=$(( SLURM_ARRAY_TASK_ID % 2 ))
seed_idx=$(( (SLURM_ARRAY_TASK_ID / 2) % 6 ))
perturb_idx=$(( (SLURM_ARRAY_TASK_ID / 12) % 2 ))

robustness=${robustness_values[$robustness_idx]}
seed=${random_seeds[$seed_idx]}
perturb=${perturbs[$perturb_idx]}

# Print configuration for logging
echo "Running experiment with:"
echo "  Robustness: $robustness"
echo "  Seed: $seed"
echo "  Perturb: $perturb"
echo "  Perturb Sigma: $perturb_sigma"

# Execute the training command
python3 experiments/train.py \
    --env_id panda \
    --algo ppo \
    --using_robustness_reward $robustness \
    --n_envs 1 \
    --reward_weights 1.0 0.01 1.0 1.0 100.0 0.0 0.0 0.0 \
    --random_seed $seed \
    --total_timesteps $total_timesteps \
    --device cuda \
    --perturb $perturb \
    --perturb_sigma $perturb_sigma
