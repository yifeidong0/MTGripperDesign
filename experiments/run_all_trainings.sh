#!/bin/bash

# Define the values for using_robustness_reward
robustness_values=(False True)

# Define the random seeds
random_seeds=(0 1 2)

# Loop through each value of using_robustness_reward
for robustness in "${robustness_values[@]}"; do
  # Loop through each random seed
  for seed in "${random_seeds[@]}"; do
    echo "Running training with using_robustness_reward=$robustness and random_seed=$seed"
    
    # Run the Python training script in the background
    python3 experiments/train.py \
      --env_id PandaUPushEnv-v0 \
      --algo sac \
      --using_robustness_reward "$robustness" \
      --render_mode rgb_array \
      --random_seed "$seed" &
    
    # Wait for 1 second before the next run
    sleep 1
  done
done

# Wait for all background processes to complete
wait

echo "All training scripts have completed."
