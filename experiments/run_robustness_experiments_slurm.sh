#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-401              # Replace with your project ID
#SBATCH -p alvis                        # Partition to use
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --gpus-per-node=A40:1           # Request 4 GPUs on that node
#SBATCH --cpus-per-gpu=16               # Request 8 CPUs per GPU (adjust as needed)
#SBATCH --time=72:00:00                 # Time limit for the job
#SBATCH --array=0-15                    # 16 tasks in total (4 GPUs * 4 tasks per GPU)

# Load necessary modules or environment
source /mimer/NOBACKUP/groups/softenable-design/mtbo/bin/activate

# Define parameters
robustness_values=(true false)
perturbs=(true false)
random_seeds=(1 2 3 4)
perturb_sigma=1.8
total_timesteps=2500000

# Calculate indices based on SLURM_ARRAY_TASK_ID
gpu_id=$(( SLURM_ARRAY_TASK_ID / 4 ))          # Calculate which GPU to use
task_id=$(( SLURM_ARRAY_TASK_ID % 4 ))         # Determine which task to run on that GPU

# Determine the specific combination of parameters for this task
robustness=${robustness_values[$((task_id / 2))]}
perturb=${perturbs[$((task_id % 2))]}
seed=${random_seeds[$gpu_id]}

# Construct the wandb group name
wandb_group_name="robustness-reward-weight 10, perturb ${perturb_sigma}, panda, PPO"

# Generate the time stamp
time_stamp=$(date +'%Y-%m-%d_%H-%M-%S')_${SLURM_ARRAY_TASK_ID}

# Print configuration for logging
echo "Running experiment on GPU $gpu_id with:"
echo "  Robustness: $robustness"
echo "  Perturb: $perturb"
echo "  Seed: $seed"
echo "  Perturb Sigma: $perturb_sigma"
echo "  Timestamp: $time_stamp"

# Set CUDA device to the assigned GPU
export CUDA_VISIBLE_DEVICES=$gpu_id

# Execute the training command
python3 experiments/train.py \
  --env_id panda \
  --algo ppo \
  --using_robustness_reward $robustness \
  --reward_weights 1.0 0.01 1.0 10.0 100.0 0.0 0.0 0.0 \
  --random_seed $seed \
  --total_timesteps $total_timesteps \
  --device cuda \
  --render_mode rgb_array \
  --perturb $perturb \
  --perturb_sigma $perturb_sigma \
  --wandb_group_name "$wandb_group_name" \
  --wandb_mode online \
  --time_stamp $time_stamp