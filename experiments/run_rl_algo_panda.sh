#!/bin/bash

# panda
robustness_values=(true)

# Define the random seeds
random_seeds=(1 2 3)

# Define perturbation values
perturbs=(false)

all_yaw_weight_ee_object=(10)
all_yaw_weight_ee_target=(1)
all_success_reward=(100)
all_object_target_distance_weight=(1)

# Define total timesteps
total_timesteps=1000000


# Function to run the command in a new VSCode terminal
run_in_vscode_terminal() {
  local robustness=$1
  local seed=$2
  local perturb=$3
  local wandb_group_name=$4
  local yaw_weight_ee_object=$5
  local yaw_weight_ee_target=$6
  local success_reward=$7
  local object_target_distance_weight=$8

  python3 experiments/train.py \
        --env_id panda \
        --wandb_group_name "$wandb_group_name" \
        --wandb_mode offline \
        --algo ppo \
        --using_robustness_reward $robustness \
        --render_mode rgb_array \
        --reward_weights $yaw_weight_ee_object $yaw_weight_ee_target $success_reward $object_target_distance_weight \
        --total_timesteps $total_timesteps \
        --device auto \
        --perturb $perturb \
        --random_seed $seed &
}

# Loop through each combination of robustness, perturb, and random seeds
for perturb in "${perturbs[@]}"; do
    for robustness in "${robustness_values[@]}"; do
        for yaw_weight_ee_object in "${all_yaw_weight_ee_object[@]}"; do
            for yaw_weight_ee_target in "${all_yaw_weight_ee_target[@]}"; do
                for success_reward in "${all_success_reward[@]}"; do
                    for object_target_distance_weight in "${all_object_target_distance_weight[@]}"; do
                        wandb_group_name="1: ${robustness}, 3: ${perturb}, 5:$yaw_weight_ee_object, 6:$yaw_weight_ee_target, 7:$success_reward, 8:$object_target_distance_weight"
                        for seed in "${random_seeds[@]}"; do
                            echo "Running training with using_robustness_reward=$robustness, random_seed=$seed, perturb=$perturb, yaw_weight_ee_object=$yaw_weight_ee_object, yaw_weight_ee_target=$yaw_weight_ee_target, success_reward=$success_reward, object_target_distance_weight=$object_target_distance_weight"
                                            
                            # Run the training command with the specified GPU
                            run_in_vscode_terminal "$robustness" "$seed" "$perturb" "$wandb_group_name" "$yaw_weight_ee_object" "$yaw_weight_ee_target" "$success_reward" "$object_target_distance_weight"
                            
                            sleep 1
                        done
                    done
                done
            done
        done
    done
done

wait