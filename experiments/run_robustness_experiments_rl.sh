#!/bin/bash

# Define the values for reward_weights[4] and reward_weights[5]
robustness_values=(true false)

# Define the random seeds
random_seeds=(1 2 3)

# Define perturbation values
perturbs=(true false)

# Function to run the command in a new VSCode terminal
# reward weights original: 5.0 1.0 1.0 1.0 100.0 0.0 0.0 0.0 
# reward weights ppo30-33: 0.2 0.2 0.2 1.0 100.0 0.0 0.0 0.0
run_in_vscode_terminal() {
  local robustness=$1
  local seed=$2
  local perturb=$3

  cmd="python3 experiments/train.py \
        --env_id vpush \
        --algo ppo \
        --using_robustness_reward $robustness \
        --render_mode rgb_array \
        --n_envs 1 \
        --reward_weights 5.0 1.0 1.0 1.0 100.0 0.0 0.0 0.0 \
        --random_seed $seed \
        --device cuda \
        --perturb $perturb "

  # Open a new VSCode terminal and run the command
  gnome-terminal -- bash -c "$cmd; exec bash"
}

# Loop through each combination of robustness, perturb, and random seeds
for seed in "${random_seeds[@]}"; do
  for perturb in "${perturbs[@]}"; do
    for robustness in "${robustness_values[@]}"; do
      run_in_vscode_terminal "$robustness" "$seed" "$perturb"
      sleep 4800 # sec
    done
  done
done

echo "All training scripts have been launched."
