#!/bin/zsh

# Define values
algos=("PPO" "PPOReg")
modes=("from_scratch" "finetune")
seeds=(123 456)

# Optional: limit number of parallel jobs
MAX_JOBS=4
job_count=0

for algo in $algos; do
  for mode in $modes; do
    for seed in $seeds; do
      echo "Running with --algo $algo --train_mode $mode --seed $seed"
      python experiments/panda_RL_train.py --algo "$algo" --train_mode "$mode" --seed "$seed" &

      ((++job_count))
      if [[ $job_count -ge $MAX_JOBS ]]; then
        wait  # Wait for current batch of jobs to finish
        job_count=0
      fi
    done
  done
done

wait  # Wait for any remaining background jobs
echo "All runs completed."
