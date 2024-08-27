#!/bin/bash

# Define the values for reward_weights[4] and reward_weights[5]
reward_weight_4_values=(10)
reward_weight_5_values=(50)
robustness_values=(true)

# Define the random seeds
random_seeds=(0 1)

# Loop through each combination of reward_weights[4] and reward_weights[5]
for robustness in "${robustness_values[@]}"; do
  for rw4 in "${reward_weight_4_values[@]}"; do
    for rw5 in "${reward_weight_5_values[@]}"; do
      # Loop through each random seed
      for seed in "${random_seeds[@]}"; do
        echo "Running training with reward_weights[4]=$rw4, reward_weights[5]=$rw5, and random_seed=$seed"
        
        # Run the Python training script in the background. TODO: unexpected keyword argument 'using_robustness_reward'
        python3 experiments/train.py \
          --env_id dlr \
          --algo ppo \
          --using_robustness_reward "$robustness" \
          --render_mode rgb_array \
          --random_seed "$seed" \
          --device cuda \
          --reward_weights 0.1 0.001 -0.03 0.1 "$rw4" "$rw5" 5e-3 100.0 &
        
        # Wait for 1 second before the next run
        sleep 1
      done
    done
  done
done

# Wait for all background processes to complete
wait

echo "All training scripts have completed."
