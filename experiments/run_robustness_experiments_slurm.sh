#!/bin/bash
#SBATCH -A NAISS2024-5-401 -p alvis
#SBATCH --time=72:00:00                   # Time limit for each task
#SBATCH --gpus-per-node=T4:4 -N 2 --cpus-per-task=32

# Load necessary modules
source /mimer/NOBACKUP/groups/softenable-design/mtbo/bin/activate

# Generate a unique identifier for each task

# Define parameters
robustness_values=(true false)
random_seeds=(1 2 3)
perturbs=(true false)
perturb_sigma=1.8
total_timesteps=2500000

# Construct the wandb group name
wandb_group_name="robustness-reward-weight 10, perturb 1.8, PPO"

# Print configuration for logging
echo "Running experiment with:"
echo "  Robustness: $robustness"
echo "  Seed: $seed"
echo "  Perturb: $perturb"
echo "  Perturb Sigma: $perturb_sigma"

# Function to run the command in a new VSCode terminal
run_in_vscode_terminal() {
  local robustness=$1
  local seed=$2
  local perturb=$3
  local time_stamp=$4

  cmd="python3 experiments/train.py \
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
        --time_stamp $time_stamp" 
        
  # Open a new VSCode terminal and run the command
  gnome-terminal -- bash -c "$cmd; exec bash"
}

# Loop through each combination of robustness, perturb, and random seeds
for seed in "${random_seeds[@]}"; do
  for perturb in "${perturbs[@]}"; do
    for robustness in "${robustness_values[@]}"; do
      time_stamp=$(date +'%Y-%m-%d_%H-%M-%S')
      run_in_vscode_terminal "$robustness" "$seed" "$perturb" "$time_stamp"
      sleep 10 # sec
    done
  done
done