#!/bin/bash

# Define the random seeds
random_seeds=(1 2 3 4 5 6)

# Function to run the command in a new VSCode terminal
run_in_vscode_terminal() {
  local model_path=$1
  local model_with_robustness_reward=$2
  local perturb=$3
  local seed=$4
  local csv_filename=$5

  cmd="python3 optimizer/mtbo.py \
        --env panda \
        --model_path $model_path \
        --model_with_robustness_reward $model_with_robustness_reward \
        --perturb $perturb \
        --algo ppo \
        --device cuda \
        --save_filename $csv_filename \
        --num_episodes_eval 20 \
        --num_episodes_eval_best 50 \
        --render_mode rgb_array \
        --num_episodes_eval 20 \
        --num_episodes_eval_best 25 \
        --max_iterations 50 \
        --random_seed $seed"

  # Open a new VSCode terminal and run the command
  gnome-terminal -- bash -c "$cmd; exec bash"
}

# Loop through each i={1,2,3,4,5,6}, corresponding to each random seed
for i in {1..4}; do
  # Get the list of zip files in alphabetical order
  # model_files=($(ls results/paper/panda/1/*.zip | sort)) # remove randomness from RL training
  model_files=($(ls results/paper/panda_new/$i/*.zip | sort))
  echo "model files name ${model_files[0]}"
  echo "model files name ${model_files[1]}"
  # echo "model files name ${model_files[2]}"
  # echo "model files name ${model_files[3]}"
  
  # Ensure there are exactly 4 zip files
  # if [ ${#model_files[@]} -ne 4 ]; then
  #   echo "Error: Expected 4 zip files in results/paper/panda/$i/"
  #   exit 1
  # fi

  # Generate csv_filename paths based on the required format
  timestamp=$(date +%Y%m%d_%H%M%S)
  csv_file_a="results/paper/panda_new/$i/panda_mtbo_results_${timestamp}_1_1_min_h.csv"
  csv_file_b="results/paper/panda_new/$i/panda_mtbo_results_${timestamp}_0_1_min_h.csv"
  # csv_file_c="results/paper/panda/$i/panda_mtbo_results_${timestamp}_1_0.csv"
  # csv_file_d="results/paper/panda/$i/panda_mtbo_results_${timestamp}_0_0.csv"

  # Run the command for each zip file with the corresponding csv_filename
  run_in_vscode_terminal "${model_files[0]}" 1 1 ${random_seeds[$i-1]} $csv_file_a  # a.zip: robustness=1, perturb=1
  run_in_vscode_terminal "${model_files[1]}" 0 1 ${random_seeds[$i-1]} $csv_file_b  # b.zip: robustness=0, perturb=1
  sleep 1200
  # run_in_vscode_terminal "${model_files[2]}" 1 0 ${random_seeds[$i-1]} $csv_file_c  # c.zip: robustness=1, perturb=0
  # sleep 2000
  # run_in_vscode_terminal "${model_files[3]}" 0 0 ${random_seeds[$i-1]} $csv_file_d  # d.zip: robustness=0, perturb=0
  # sleep 2000
done

echo "All training scripts have been launched."