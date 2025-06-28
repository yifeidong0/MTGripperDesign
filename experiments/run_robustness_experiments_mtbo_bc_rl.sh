#!/bin/bash

# Define the random seeds
random_seeds=(1 2 3 4 5 6)

# Function to run the command in a new VSCode terminal
# run_in_vscode_terminal() {
#   local model_path=$1
#   local model_with_robustness_reward=$2
#   local perturb=$3
#   local seed=$4
#   local csv_filename=$5

#   cmd="python3 optimizer/mtbo_bc_rl.py \
#         --env panda \
#         --model_path $model_path \
#         --model_with_robustness_reward $model_with_robustness_reward \
#         --perturb $perturb \
#         --algo ppo \
#         --device cuda \
#         --save_filename $csv_filename \
#         --num_episodes_eval 20 \
#         --num_episodes_eval_best 50 \
#         --render_mode rgb_array \
#         --num_episodes_eval 20 \
#         --num_episodes_eval_best 25 \
#         --max_iterations 50 \
#         --random_seed $seed"

#   # Open a new VSCode terminal and run the command
#   gnome-terminal -- bash -c "$cmd; exec bash"
# }

# Function to run the command directly in the current terminal or as a background job
run_mtbo_experiment() {
  local model_path=$1
  local model_with_robustness_reward=$2
  local perturb=$3
  local seed=$4
  local csv_filename=$5

  python3 optimizer/mtbo_bc_rl.py \
    --env panda \
    --model_path "$model_path" \
    --model_with_robustness_reward "$model_with_robustness_reward" \
    --perturb "$perturb" \
    --algo ppo \
    --device cuda \
    --save_filename "$csv_filename" \
    --num_episodes_eval 20 \
    --num_episodes_eval_best 25 \
    --render_mode rgb_array \
    --max_iterations 50 \
    --random_seed "$seed"
}

# Loop through each i={1,2,3,4,5,6}, corresponding to each random seed
for i in {1..1}; do
  # Get the list of zip files in alphabetical order
  # model_files=($(ls results/paper/panda/1/*.zip | sort)) # remove randomness from RL training
  # model_files=($(ls results/paper/panda_new/$i/*.zip | sort))
  model_file="data/models/PandaUPushEnv-v0_final_model_perturb.zip"
  # echo "model files name ${model_files[0]}"
  echo "model files name ${model_file}"

  # Generate csv_filename paths based on the required format
  timestamp=$(date +%Y%m%d_%H%M%S)
  # csv_file_a="results/paper/panda_new/$i/panda_mtbo_results_${timestamp}_1_1_min_h.csv"
  csv_file_b="results/rl_course/$i/panda_mtbo_bc_rl_results_${timestamp}_0_1.csv"

  # Run the command for each zip file with the corresponding csv_filename
  # run_in_vscode_terminal "${model_files[0]}" 1 1 ${random_seeds[$i-1]} $csv_file_a  # a.zip: robustness=1, perturb=1
  run_mtbo_experiment "${model_file}" 0 1 ${random_seeds[$i-1]} $csv_file_b  # b.zip: robustness=0, perturb=1
  # sleep 1200
done

echo "All training scripts have been launched."