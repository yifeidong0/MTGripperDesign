#!/bin/zsh

# Define values
algos=("PPO" "PPOReg")
modes=("finetune" "from_scratch")
seeds=(123 456)

SESSION_NAME="rl_train"
tmux new-session -d -s $SESSION_NAME  # Create tmux session in detached mode
window_index=1

for algo in $algos; do
  for mode in $modes; do
    for seed in $seeds; do
      WINDOW_NAME="${algo}_${mode}_${seed}"
      CMD="python experiments/panda_RL_train.py --algo ${algo} --train_mode ${mode} --seed ${seed}"

      # Create a new tmux window for each job
      tmux new-window -t $SESSION_NAME:$window_index -n $WINDOW_NAME "$CMD"
      ((window_index++))
    done
  done
done

echo "All tmux windows created. Attach with: tmux attach -t $SESSION_NAME"



