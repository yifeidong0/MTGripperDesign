#!/bin/zsh

# xarm7
robustness_values=(true)

# Define the random seeds
random_seeds=(1)

# Define perturbation values
perturbs=(false)

all_yaw_weight_ee_object=(5)
all_yaw_weight_ee_target=(5)
all_success_reward=(100)
all_object_target_distance_weight=(1)
all_robustness_weight=(5)
all_ee_object_distance_weight=(1)

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
  local robustness_weight=$9
  local ee_object_distande_weight=${10}

  python3 experiments/train.py \
        --env_id xarm7 \
        --wandb_group_name "$wandb_group_name" \
        --wandb_mode offline \
        --algo ppo \
        --using_robustness_reward $robustness \
        --render_mode rgb_array \
        --reward_weights $yaw_weight_ee_object $yaw_weight_ee_target $success_reward $object_target_distance_weight $robustness_weight $ee_object_distande_weight \
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
                        for robustness_weight in "${all_robustness_weight[@]}"; do
                            for ee_object_distande_weight in "${all_ee_object_distance_weight[@]}"; do
                                wandb_group_name="1: ${robustness}, 3: ${perturb}, 5:$yaw_weight_ee_object, 6:$yaw_weight_ee_target, 7:$success_reward, 8:$object_target_distance_weight, 9:$robustness_weight, 10:$ee_object_distande_weight"
                                for seed in "${random_seeds[@]}"; do

                                    # Run the training command with the specified GPU
                                    run_in_vscode_terminal "$robustness" "$seed" "$perturb" "$wandb_group_name" "$yaw_weight_ee_object" "$yaw_weight_ee_target" "$success_reward" "$object_target_distance_weight" "$robustness_weight" "$ee_object_distande_weight"
                                    
                                    sleep 1
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

wait