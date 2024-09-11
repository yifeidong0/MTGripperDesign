#!/bin/bash

# panda
robustness_values=(true false)

# Define the random seeds
random_seeds=(0 1)

# Define perturbation values
perturbs=(false)

yaw_weight=(0.025 0.05 0.075)

# Define total timesteps
total_timesteps=5000000


# Function to run the command in a new VSCode terminal
run_in_vscode_terminal() {
  local robustness=$1
  local seed=$2
  local perturb=$3
  local wandb_group_name=$4
  local yaw_weight=$5

  # Set the CUDA_VISIBLE_DEVICES environment variable to assign the GPU
  python3 experiments/train.py \
        --env_id panda \
        --wandb_group_name "$wandb_group_name" \
        --wandb_mode online \
        --algo ppo \
        --using_robustness_reward $robustness \
        --render_mode rgb_array \
        --reward_weights $yaw_weight \
        --total_timesteps $total_timesteps \
        --device auto \
        --perturb $perturb \
        --checkpoint_freq 500000 \
        --random_seed $seed &
}

# Loop through each combination of robustness, perturb, and random seeds
for perturb in "${perturbs[@]}"; do
    for robustness in "${robustness_values[@]}"; do
        for yaw_weight in "${yaw_weight[@]}"; do
            wandb_group_name="robustness ${robustness}, perturb ${perturb}, yaw_weight ${yaw_weight}"
            for seed in "${random_seeds[@]}"; do
                echo "Running training with using_robustness_reward=$robustness, random_seed=$seed, perturb=$perturb, yaw_weight=$yaw_weight"
                                
                # Run the training command with the specified GPU
                run_in_vscode_terminal "$robustness" "$seed" "$perturb" "$wandb_group_name" "$yaw_weight"
                
                sleep 1
            done
        done
    done
done

wait