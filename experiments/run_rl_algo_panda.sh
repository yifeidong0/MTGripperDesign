#!/bin/bash

# panda
robustness_values=(true false)

# Define the random seeds
random_seeds=(0 1)

# Define perturbation values
perturbs=(false)

all_yaw_weight_ee_object=(0.1)
all_yaw_weight_ee_target=(0.1)
all_grasp_reward=(10 50)

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
  local grasp_reward=$7

  python3 experiments/train.py \
        --env_id panda \
        --wandb_group_name "$wandb_group_name" \
        --wandb_mode online \
        --algo ppo \
        --using_robustness_reward $robustness \
        --render_mode rgb_array \
        --reward_weights $yaw_weight_ee_object $yaw_weight_ee_target $grasp_reward \
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
                for grasp_reward in "${all_grasp_reward[@]}"; do
                    wandb_group_name="robustness ${robustness}, perturb ${perturb}, yaw_weight_ee_object ${yaw_weight_ee_object}, yaw_weight_ee_target ${yaw_weight_ee_target}"
                    for seed in "${random_seeds[@]}"; do
                        echo "Running training with using_robustness_reward=$robustness, random_seed=$seed, perturb=$perturb, yaw_weight_ee_object=$yaw_weight_ee_object, yaw_weight_ee_target=$yaw_weight_ee_target, grasp_reward=$grasp_reward"
                                        
                        # Run the training command with the specified GPU
                        run_in_vscode_terminal "$robustness" "$seed" "$perturb" "$wandb_group_name" "$yaw_weight_ee_object" "$yaw_weight_ee_target" "$grasp_reward"
                        
                        sleep 1
                    done
                done
            done
        done
    done
done

wait