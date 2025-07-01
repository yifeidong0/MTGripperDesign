#!/bin/zsh

# Define values
algos=("PPO" "PPOReg")
train_modes=("finetune" "from_scratch")
test_modes=("test_before_training" "test_after_training")

SESSION_NAME="rl_test"
tmux new-session -d -s $SESSION_NAME  # Create tmux session in detached mode
window_index=1

for algo in $algos; do
  for train_mode in $train_modes; do
    for test_mode in $test_modes; do
      WINDOW_NAME="${algo}_${train_mode}_${test_mode}"
      
      CMD="
        echo 'Running with algo=${algo}, train_mode=${train_mode}, test_mode=${test_mode}';
        python3 experiments/panda_RL_test.py --algo ${algo} --train_mode ${train_mode} --test_mode ${test_mode} --seed 456 --n_eval_episodes 100;
        echo 'Done. Press any key to close.';
        read
      "

      # Create a new tmux window for each job
      tmux new-window -t $SESSION_NAME:$window_index -n $WINDOW_NAME "zsh -c \"$CMD\""
      ((window_index++))
    done
  done
done

echo "All tmux windows created. Attach with: tmux attach -t $SESSION_NAME"


# IL average return: -40.91802177518737, success rate: 0.65
# PPO finetune average return: -28.137411540560315, success rate: 0.34
# PPO scratch average return: -12.77407824106363, success rate: 0.4
# PPOReg finetune average -22.250268204846993, success rate: 0.81
