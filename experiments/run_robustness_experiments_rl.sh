#!/bin/bash

# panda
# Define the values for reward_weights[4] and reward_weights[5]
robustness_values=(true false)

# Define the random seeds
random_seeds=(1 2 3)

# Define perturbation values
perturbs=(true false)
perturb_sigma=1.5

# Define total timesteps
total_timesteps=3000000

# Construct the wandb group name
wandb_group_name="robustness-reward-weight 5, perturb 1.5, PPO"

# Function to run the command in a new VSCode terminal
run_in_vscode_terminal() {
  local robustness=$1
  local seed=$2
  local perturb=$3
  local time_stamp=$4

  cmd="python3 experiments/train.py \
        --env_id panda \
        --algo ppo \
        --using_robustness_reward $robustness \
        --reward_weights 1.0 0.01 1.0 5.0 100.0 0.0 0.0 0.0 \
        --random_seed $seed \
        --total_timesteps $total_timesteps \
        --render_mode rgb_array \
        --device cuda \
        --wandb_group_name "$wandb_group_name" \
        --wandb_mode online \
        --perturb $perturb \
        --perturb_sigma $perturb_sigma \
        --time_stamp $time_stamp" 
        

  # Open a new VSCode terminal and run the command
  gnome-terminal -- bash -c "$cmd; exec bash"
}

# Loop through each combination of robustness, perturb, and random seeds
for seed in "${random_seeds[@]}"; do
  for perturb in "${perturbs[@]}"; do
    for robustness in "${robustness_values[@]}"; do
      time_stamp=$(date +'%Y-%m-%d_%H-%M-%S')
      run_in_vscode_terminal "$robustness" "$seed" "$perturb" "$time_stamp"
      sleep 10 # sec
    done
  done
  sleep 28800 # 8 hours
done

echo "All training scripts have been launched."





# #!/bin/bash

# # vpush
# # Define the values for reward_weights[4] and reward_weights[5]
# robustness_values=(true false)

# # Define the random seeds
# random_seeds=(3 4)

# # Define perturbation values
# perturbs=(true false)

# # Define total timesteps
# total_timesteps=5000000  # 5e6

# # Function to run the command in a new VSCode terminal
# # reward weights original: 5.0 1.0 1.0 1.0 100.0 0.0 0.0 0.0 
# # reward weights ppo30-33: 0.2 0.2 0.2 1.0 100.0 0.0 0.0 0.0
# # poor: 0.3 0.3 0.3 3.0 100.0 0.0 0.0 0.0
# run_in_vscode_terminal() {
#   local robustness=$1
#   local seed=$2
#   local perturb=$3

#   cmd="python3 experiments/train.py \
#         --env_id vpush \
#         --algo ppo \
#         --using_robustness_reward $robustness \
#         --render_mode rgb_array \
#         --n_envs 1 \
#         --reward_weights 5.0 1.0 1.0 1.0 100.0 0.0 0.0 0.0 \
#         --random_seed $seed \
#         --device cuda \
#         --total_timesteps $total_timesteps \
#         --perturb $perturb "

#   # Open a new VSCode terminal and run the command
#   gnome-terminal -- bash -c "$cmd; exec bash"
# }

# # Loop through each combination of robustness, perturb, and random seeds
# for seed in "${random_seeds[@]}"; do
#   for perturb in "${perturbs[@]}"; do
#     for robustness in "${robustness_values[@]}"; do
#       run_in_vscode_terminal "$robustness" "$seed" "$perturb"
#       sleep 60 # sec
#     done
#     sleep 14400 # 4 hours
#   done
# done

# echo "All training scripts have been launched."



# #!/bin/bash

# # catch
# # Define the values for reward_weights[4] and reward_weights[5]
# robustness_values=(true false)

# # Define the random seeds
# random_seeds=(5 6)

# # Define perturbation values
# perturbs=(true false)

# # Function to run the command in a new VSCode terminal
# run_in_vscode_terminal() {
#   local robustness=$1
#   local seed=$2
#   local perturb=$3

#   cmd="python3 experiments/train.py \
#         --env_id catch \
#         --algo ppo \
#         --using_robustness_reward $robustness \
#         --render_mode rgb_array \
#         --n_envs 24 \
#         --random_seed $seed \
#         --device cuda \
#         --perturb $perturb "

#   # Open a new VSCode terminal and run the command
#   gnome-terminal -- bash -c "$cmd; exec bash"
# }

# # Loop through each combination of robustness, perturb, and random seeds
# for seed in "${random_seeds[@]}"; do
#   for perturb in "${perturbs[@]}"; do
#     for robustness in "${robustness_values[@]}"; do
#       run_in_vscode_terminal "$robustness" "$seed" "$perturb"
#       sleep 1200 # sec
#     done
#   done
# done

# echo "All training scripts have been launched."