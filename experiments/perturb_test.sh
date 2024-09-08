#!/bin/bash

# Paths to models for different environments
declare -A models
models["vpush"]="results/paper/vpush"
models["catch"]="results/paper/catch"
models["dlr"]="results/paper/dlr"
models["panda"]="results/paper/panda"

# # Environment IDs
# declare -A env_ids
# env_ids["vpush"]="VPushPbSimulationEnv-v0"
# env_ids["catch"]="UCatchSimulationEnv-v0"
# env_ids["dlr"]="DLRSimulationEnv-v0"
# env_ids["panda"]="PandaUPushEnv-v0"

# Perturbation levels for each environment
declare -A perturb_levels
perturb_levels["vpush"]="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
perturb_levels["catch"]="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
perturb_levels["panda"]="0.0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2"
perturb_levels["dlr"]="0.0 0.3 0.6 0.9 1.2 1.5 1.8 2.1 2.4"

# Random seeds
random_seeds=(42 43 44 45 46)
num_episodes=50
render_mode="rgb_array" # "human"

# # Iterate over environments
# for env in "catch" "vpush" "panda" "dlr"; do
#     # env_id=${env_ids[$env]}

#     # Iterate over perturbation levels
#     for perturb in ${perturb_levels[$env]}; do
#         csv_file_a="results/paper/perturb/${env}_perturb_${perturb}_combination_1_robust_1_perturb_1.csv"
#         csv_file_b="results/paper/perturb/${env}_perturb_${perturb}_combination_2_robust_0_perturb_1.csv"
#         csv_file_c="results/paper/perturb/${env}_perturb_${perturb}_combination_3_robust_1_perturb_0.csv"
#         csv_file_d="results/paper/perturb/${env}_perturb_${perturb}_combination_4_robust_0_perturb_0.csv"
        
#         # Iterate over 5 random seeds
#         for i in {1..5}; do
#             model_files=($(ls ${models[$env]}/$i/*.zip | sort))
#             random_seed=${random_seeds[$i-1]}

#             # Run the 4 different combinations
#             python3 experiments/perturb_ablation_test.py --num_episodes "$num_episodes"  --render_mode  "$render_mode" --model_path "${model_files[0]}" --env_id "$env" --perturb_sigma "$perturb" --perturb 1 --random_seed "$random_seed" --output_file "$csv_file_a" 
#             python3 experiments/perturb_ablation_test.py --num_episodes "$num_episodes"  --render_mode  "$render_mode" --model_path "${model_files[1]}" --env_id "$env" --perturb_sigma "$perturb" --perturb 1 --random_seed "$random_seed" --output_file "$csv_file_b"
#             python3 experiments/perturb_ablation_test.py --num_episodes "$num_episodes"  --render_mode  "$render_mode" --model_path "${model_files[2]}" --env_id "$env" --perturb_sigma "$perturb" --perturb 1 --random_seed "$random_seed" --output_file "$csv_file_c"
#             python3 experiments/perturb_ablation_test.py --num_episodes "$num_episodes"  --render_mode  "$render_mode" --model_path "${model_files[3]}" --env_id "$env" --perturb_sigma "$perturb" --perturb 1 --random_seed "$random_seed" --output_file "$csv_file_d"
#         done
#     done
# done

# Meta-data generation (mean and std across 500 trajectories)
output_meta="results/paper/perturb/0_meta_results.csv"
echo "env,perturb_sigma,combination_id,mean_success_rate,std_success_rate" > "$output_meta"

# Iterate through all environments and perturb levels to gather meta-data
for env in "vpush" "catch" "dlr" "panda"; do
    for combination_id in {1..4}; do
        for perturb in ${perturb_levels[$env]}; do
            # Combine all csv files for the current {env, combination_id, perturb, }
            success_rates=()
            csv_files=$(ls results/paper/perturb/${env}_perturb_${perturb}_combination_${combination_id}_*.csv)
            
            # Extract the success rates from each file
            for csv_file in $csv_files; do
                success_rate=$(awk -F, 'NR==2 {print $3}' "$csv_file")
                success_rates+=($success_rate)
            done

            # Calculate mean and std
            mean_success_rate=$(echo "${success_rates[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')
            std_success_rate=$(echo "${success_rates[@]}" | awk '{sum=0; sumsq=0; for (i=1;i<=NF;i++) {sum+=$i; sumsq+=$i*$i} print sqrt(sumsq/NF - (sum/NF)*(sum/NF))}')

            # Append the results to the meta CSV
            echo "$env,$perturb,$combination_id,$mean_success_rate,$std_success_rate" >> "$output_meta"
        done
    done
done

echo "Meta-data results saved to $output_meta"
