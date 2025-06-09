import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envs

import torch
import numpy as np

import gymnasium as gym
# from stable_baselines3 import PPO
from policy.ppo_reg import PPOReg

from stable_baselines3.common.evaluation import evaluate_policy


if __name__ == "__main__":
    env_id = "PandaUPushEnv-v0"
    env = gym.make(env_id)
    # model = PPOReg("MultiInputPolicy", env, verbose=1, 
    #                 tensorboard_log=f"data/runs/{env_id}/",
    #                 max_grad_norm=0.01,
    #                 clip_range=0.01,
    #                 n_steps=1024,
    #                 policy_kwargs={
    #                     "net_arch": [32, 32],
    #                     "log_std_init": -10.0,
    #                 },
    #             )
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # pretrained_policy_path = "data/panda_bc_policy.pth"
    # saved_variables = torch.load(pretrained_policy_path, map_location=device, weights_only=False)

    # model.set_pretrained_policy(saved_variables, log_std_init=-6.0, reg_coef=0.5)
        
    # model.policy.eval()  # Set the policy to evaluation mode
    # n_episodes=1
    # reward, _ = evaluate_policy(model.policy, env, n_eval_episodes=n_episodes, deterministic=True)
    # print(f"Before fine-tuning, average reward over {n_episodes} episodes: {reward}")

    # try:
    #     model.learn(total_timesteps=int(1e6), log_interval=5, progress_bar=True)
    # except KeyboardInterrupt:
    #     print("Training interrupted")
    # finally:
    #     model.save(f"data/models/{env_id}_final_model.zip")
    #     pass
    model = PPOReg.load("data/models/PandaUPushEnv-v0_final_model.zip", env=env)
    
    model.policy.eval()  # Set the policy to evaluation mode
    n_episodes = 10
    mean_reward, std_reward = evaluate_policy(model.policy, env, n_eval_episodes=n_episodes, deterministic=True)
    env.close()
    print(f"After fine-tuning, average reward over {n_episodes} episodes: {mean_reward}")
