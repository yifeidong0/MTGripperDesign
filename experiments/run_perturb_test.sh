#!/bin/bash

# Paths to models for different environments
declare -A models
models["vpush"]="results/paper/vpush"
models["catch"]="results/paper/catch"
models["dlr"]="results/paper/dlr"
models["panda"]="results/paper/panda"

# Perturbation levels for each environment
declare -A perturb_levels
perturb_levels["vpush"]="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
perturb_levels["catch"]="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
perturb_levels["panda"]="0.0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2 3.6"
perturb_levels["dlr"]="0.0 0.3 0.6 0.9 1.2 1.5 1.8 2.1 2.4"

# Random seeds
random_seeds=(42 43 44 45 46)
num_episodes=50
render_mode="rgb_array" # "human"

# Iterate over environments
for env in "catch" "vpush" "panda" "dlr"; do
    # env_id=${env_ids[$env]}

    # Iterate over perturbation levels
    for perturb in ${perturb_levels[$env]}; do
        csv_file_a="results/paper/perturb/${env}_perturb_${perturb}_combination_1_robust_1_perturb_1.csv"
        csv_file_b="results/paper/perturb/${env}_perturb_${perturb}_combination_2_robust_0_perturb_1.csv"
        csv_file_c="results/paper/perturb/${env}_perturb_${perturb}_combination_3_robust_1_perturb_0.csv"
        csv_file_d="results/paper/perturb/${env}_perturb_${perturb}_combination_4_robust_0_perturb_0.csv"
        
        # Iterate over 5 random seeds
        for i in {1..5}; do
            model_files=($(ls ${models[$env]}/$i/*.zip | sort))
            random_seed=${random_seeds[$i-1]}

            # Run the 4 different combinations
            python3 experiments/perturb_ablation_test.py --num_episodes "$num_episodes"  --render_mode  "$render_mode" --model_path "${model_files[0]}" --env_id "$env" --perturb_sigma "$perturb" --perturb 1 --random_seed "$random_seed" --output_file "$csv_file_a" 
            python3 experiments/perturb_ablation_test.py --num_episodes "$num_episodes"  --render_mode  "$render_mode" --model_path "${model_files[1]}" --env_id "$env" --perturb_sigma "$perturb" --perturb 1 --random_seed "$random_seed" --output_file "$csv_file_b"
            python3 experiments/perturb_ablation_test.py --num_episodes "$num_episodes"  --render_mode  "$render_mode" --model_path "${model_files[2]}" --env_id "$env" --perturb_sigma "$perturb" --perturb 1 --random_seed "$random_seed" --output_file "$csv_file_c"
            python3 experiments/perturb_ablation_test.py --num_episodes "$num_episodes"  --render_mode  "$render_mode" --model_path "${model_files[3]}" --env_id "$env" --perturb_sigma "$perturb" --perturb 1 --random_seed "$random_seed" --output_file "$csv_file_d"
        done
    done
done
