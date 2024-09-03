#!/bin/bash

# Define the values for reward_weights[4] and reward_weights[5]
reward_weight_2_values=(-0.03)
reward_weight_4_values=(50)
reward_weight_5_values=(50)
robustness_values=(false)
total_timesteps=3000000

# Define the random seeds
random_seeds=(0)

# Function to run the Python training script in a new gnome terminal
run_in_gnome_terminal() {
  local rw4=$1
  local rw5=$2
  local seed=$3
  local robustness=$4
  local rw2=$5

  cmd="python3 experiments/train.py \
        --env_id dlr \
        --algo ppo \
        --using_robustness_reward $robustness \
        --total_timesteps $total_timesteps \
        --random_seed $seed \
        --device cuda \
        --perturb 0 \
        --reward_weights 0.1 0.001 $rw2 1.0 $rw4 $rw5 5e-3 100.0"

        # --render_mode rgb_array \
  # Open a new gnome terminal and run the command
  gnome-terminal -- bash -c "$cmd; exec bash"
}

# Loop through each combination of reward_weights[4] and reward_weights[5]
for robustness in "${robustness_values[@]}"; do
  for rw2 in "${reward_weight_2_values[@]}"; do
    for rw4 in "${reward_weight_4_values[@]}"; do
      for rw5 in "${reward_weight_5_values[@]}"; do
        # Loop through each random seed
        for seed in "${random_seeds[@]}"; do
          echo "Running training with reward_weights[4]=$rw4, reward_weights[5]=$rw5, and random_seed=$seed"

          # Run the Python training script in a new gnome terminal
          run_in_gnome_terminal "$rw4" "$rw5" "$seed" "$robustness" "$rw2"
          
          # Wait for 1 second before the next run
          sleep 3600
        done
      done
    done
  done
done

echo "All training scripts have been launched."