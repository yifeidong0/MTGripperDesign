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
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy
from policy.ppo_reg import PPOReg
from panda_IL import PandaEnvWrapper, DeepPolicy


def evaluate_policy(model, env, n_eval_episodes=10, deterministic=True):
    returns, successes = [], []
    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        eps_return = 0
        eps_success = 0
        while not done:
            actions, states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            eps_return += reward
            eps_success = info["is_success"]
        returns.append(eps_return)
        successes.append(eps_success)
    return np.mean(returns), np.mean(successes)

def main(args):
    algo_name = args.algo  # "PPO" or "PPOReg"
    train_mode = args.train_mode  # "finetune" or "from_scratch"
    test_mode = args.test_mode  # "train" or "test_before_finetune" or "test_after_training"
    seed = args.seed  # Random seed for reproducibility
    n_eval_episodes = args.n_eval_episodes  # Number of episodes for evaluation
    set_random_seed(seed, using_cuda=False)  # leave the CUDA randomness for performance
    render_mode = "rgb_array"  # "human" or "rgb_array"
    env_id = "PandaUPushEnv-v0"
    env = gym.make(env_id, render_mode=render_mode, run_id=f"{algo_name}_{train_mode}_{test_mode}_{seed}")
    env = PandaEnvWrapper(env, stat_file="data/obs_statistics.pkl")
    env.reset(seed=seed)
    
    if test_mode == "test_before_training":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrained_policy_path = "data/panda_bc_policy_rand_morph.pth"
        saved_variables = torch.load(pretrained_policy_path, map_location=device, weights_only=False)
        policy = ActorCriticPolicy(**saved_variables["data"])
        policy.load_state_dict(saved_variables['state_dict'])
        policy.to(device)
        policy.eval()  # Set the policy to evaluation mode
        mean_return, mean_success = evaluate_policy(policy, env, n_eval_episodes=n_eval_episodes, deterministic=True)
        print(f"Before training, over {n_eval_episodes}, average return: {mean_return}, success rate: {mean_success}")
        
    else:
        model = None
        if algo_name == "PPO":
            model = PPO.load(f"data/models/{env_id}_{algo_name}_{train_mode}_{seed}", env=env, policy=DeepPolicy)
        elif algo_name == "PPOReg" and train_mode == "finetune":
            model = PPOReg.load(f"data/models/{env_id}_{algo_name}_{train_mode}_{seed}", env=env, policy=DeepPolicy)
        if model is not None:
            model.policy.eval()  # Set the policy to evaluation mode
            mean_return, mean_success = evaluate_policy(model.policy, env, n_eval_episodes=n_eval_episodes, deterministic=True)
            print(f"After training, over {n_eval_episodes}, average return: {mean_return}, success rate: {mean_success}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panda RL Test Script")
    parser.add_argument("--algo", type=str, default="PPOReg", choices=["PPO", "PPOReg"], help="Algorithm to use")
    parser.add_argument("--train_mode", type=str, default="finetune", choices=["finetune", "from_scratch"], help="Training mode")
    parser.add_argument("--test_mode", type=str, default="test_after_training", choices=["test_before_training", "test_after_training"], help="Mode of operation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n_eval_episodes", type=int, default=10, help="Number of episodes for evaluation")
    
    args = parser.parse_args()
    main(args)