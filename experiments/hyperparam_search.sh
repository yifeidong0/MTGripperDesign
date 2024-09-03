#!/bin/bash

# Define the values for reward_weights[4] and reward_weights[5]
reward_weight_2_values=(-0.05)
reward_weight_4_values=(50)
reward_weight_5_values=(50)
robustness_values=(true false)
total_timesteps=1000000
perturb_values=(true false)

# Define the random seeds
random_seeds=(0)

# Function to run the Python training script in a new gnome terminal
run_in_gnome_terminal() {
  local rw2=$1
  local rw4=$2
  local rw5=$3
  local seed=$4
  local robustness=$5
  local perturb=$6

  cmd="python3 experiments/train.py \
        --env_id dlr \
        --algo ppo \
        --using_robustness_reward $robustness \
        --total_timesteps $total_timesteps \
        --random_seed $seed \
        --device cuda \
        --perturb $perturb \
        --render_mode rgb_array \
        --reward_weights 0.1 0.001 $rw2 1.0 $rw4 $rw5 2e-3 100.0"

  # Open a new gnome terminal and run the command
  gnome-terminal -- bash -c "$cmd; exec bash"
}

# Loop through each combination
for seed in "${random_seeds[@]}"; do
  for robustness in "${robustness_values[@]}"; do
    for perturb in "${perturb_values[@]}"; do
      for rw2 in "${reward_weight_2_values[@]}"; do
        for rw4 in "${reward_weight_4_values[@]}"; do
          for rw5 in "${reward_weight_5_values[@]}"; do
            # Run the Python training script in a new gnome terminal
            run_in_gnome_terminal "$rw2" "$rw4" "$rw5" "$seed" "$robustness" "$perturb"
            
            # Wait for 1 second before the next run
            sleep 60
          done
        done
      done
    done
  done
done

echo "All training scripts have been launched."