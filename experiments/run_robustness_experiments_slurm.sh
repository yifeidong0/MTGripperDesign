#!/bin/bash
#SBATCH -A NAISS2024-5-401 -p alvis
#SBATCH --nodes=4                        # Number of nodes (1 node is sufficient)
#SBATCH --cpus-per-task=64                # Number of CPU cores per task
#SBATCH --time=72:00:00                   # Time limit for each task
#SBATCH --array=0-23%24                   # Array jobs: 24 tasks, run 12 simultaneously

# Load necessary modules
# module load Python/3.10.8-GCCcore-12.2.0
source /mimer/NOBACKUP/groups/softenable-design/mtbo/bin/activate

# Generate a unique identifier for each task
time_stamp=$(date +'%Y-%m-%d_%H-%M-%S')_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# Define parameters
robustness_values=(true false)
random_seeds=(1 2 3 4 5 6)
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
wandb_group_name="ppo with robustness: ${robustness}, perturb: ${perturb}, random seed: ${seed}"

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
    --reward_weights 1.0 0.01 1.0 1.0 100.0 0.0 0.0 0.0 \
    --random_seed $seed \
    --total_timesteps $total_timesteps \
    --device cpu \
    --render_mode rgb_array \
    --perturb $perturb \
    --perturb_sigma $perturb_sigma \
    --checkpoint_freq $checkpoint_freq \
    --wandb_group_name "$wandb_group_name" \
    --wandb_mode online \
    --time_stamp $time_stamp
