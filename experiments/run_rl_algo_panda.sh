#!/bin/bash

# panda
# Define the values for reward_weights[4] and reward_weights[5]
robustness_values=(true false)

# Define the random seeds
random_seeds=(0 1 2 3 4)

# Define perturbation values
perturbs=(true)

algos=(ppo sac)

# Define total timesteps
total_timesteps=5000000

# Define GPU IDs to use
gpus=(0 1 2 3)  # Modify this list based on the available GPUs and your requirements

# Function to run the command in a new VSCode terminal
run_in_vscode_terminal() {
  local robustness=$1
  local seed=$2
  local perturb=$3
  local rl_algo=$4
  local wandb_group_name=$5
  local gpu_id=$6  # Added GPU ID

  # Set the CUDA_VISIBLE_DEVICES environment variable to assign the GPU
  CUDA_VISIBLE_DEVICES=$gpu_id python3 experiments/train.py \
        --env_id panda \
        --wandb_group_name "$wandb_group_name" \
        --wandb_mode online \
        --algo $rl_algo \
        --using_robustness_reward $robustness \
        --render_mode rgb_array \
        --reward_weights 1.0 0.01 1.0 1.0 100.0 0.0 0.0 0.0 \
        --total_timesteps $total_timesteps \
        --device auto \
        --perturb $perturb \
        --checkpoint_freq 500000 \
        --random_seed $seed &
}

# Counter for cycling through GPUs
gpu_counter=0

# Loop through each combination of robustness, perturb, and random seeds
for perturb in "${perturbs[@]}"; do
    for robustness in "${robustness_values[@]}"; do
        for rl_algo in "${algos[@]}"; do
            wandb_group_name="${rl_algo}, robustness ${robustness}, perturb ${perturb}"
            for seed in "${random_seeds[@]}"; do
                echo "Running training with using_robustness_reward=$robustness, random_seed=$seed, perturb=$perturb, algo=$rl_algo"
                
                # Assign a GPU from the list
                gpu_id=${gpus[$gpu_counter]}
                
                # Run the training command with the specified GPU
                run_in_vscode_terminal "$robustness" "$seed" "$perturb" "$rl_algo" "$wandb_group_name" "$gpu_id"
                
                # Increment the GPU counter and cycle through available GPUs
                gpu_counter=$(( (gpu_counter + 1) % ${#gpus[@]} ))
                
                sleep 1000
            done
        done
    done
done

wait