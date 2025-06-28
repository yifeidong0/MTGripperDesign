import os
import sys
from pathlib import Path

# Add parent directory to path more robustly
sys.path.append(str(Path(__file__).parent.parent))

import envs
import torch
import numpy as np
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from policy.ppo_reg import PPOReg
from panda_IL import PandaEnvWrapper, DeepPolicy


def main(args):
    algo_name = args.algo  # "PPO" or "PPOReg"
    train_mode = args.train_mode  # "finetune" or "from_scratch"
    test_mode = args.test_mode  # "train" or "test_before_finetune" or "test_after_finetune"
    seed = args.seed  # Random seed for reproducibility
    n_eval_episodes = args.n_eval_episodes  # Number of episodes for evaluation
    render_mode = "human"  # "human" or "rgb_array"
    env_id = "PandaUPushEnv-v0"
    env = gym.make(env_id, render_mode=render_mode, max_episode_steps=1000)
    env = PandaEnvWrapper(env, stat_file="data/obs_statistics.pkl")
    
    if test_mode == "test_before_training":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrained_policy_path = "data/panda_bc_policy_rand_morph.pth"
        saved_variables = torch.load(pretrained_policy_path, map_location=device, weights_only=False)
        policy = DeepPolicy.load_from_state_dict(saved_variables['state_dict'], device=device)
        policy.eval()  # Set the policy to evaluation mode
        mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=n_eval_episodes, deterministic=True)
        print(f"Before training, over {n_eval_episodes}, average reward: {mean_reward}, std: {std_reward}")
        
    else:
        if algo_name == "PPO":
            model = PPO.load(f"data/models/{env_id}_{algo_name}_{train_mode}_{seed}", env=env, policy=DeepPolicy)
        elif algo_name == "PPOReg":
            model = PPOReg.load(f"data/models/{env_id}_{algo_name}_{train_mode}_{seed}", env=env, policy=DeepPolicy)
        model.policy.eval()  # Set the policy to evaluation mode
        mean_reward, std_reward = evaluate_policy(model.policy, env, n_eval_episodes=n_eval_episodes, deterministic=True)
        print(f"After training, over {n_eval_episodes}, average reward: {mean_reward}, std: {std_reward}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panda RL Test Script")
    parser.add_argument("--algo", type=str, default="PPOReg", choices=["PPO", "PPOReg"], help="Algorithm to use")
    parser.add_argument("--train_mode", type=str, default="finetune", choices=["finetune", "from_scratch"], help="Training mode")
    parser.add_argument("--test_mode", type=str, default="test_after_finetune", choices=["train", "test_before_finetune", "test_after_finetune"], help="Mode of operation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n_eval_episodes", type=int, default=10, help="Number of episodes for evaluation")
    
    args = parser.parse_args()
    main(args)