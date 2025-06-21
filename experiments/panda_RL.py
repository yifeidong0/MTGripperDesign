import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envs

import torch
import numpy as np

import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3 import PPO
from policy.ppo_reg import PPOReg

from stable_baselines3.common.evaluation import evaluate_policy


def train(env_id: str, model_path: str, env_path: str) -> None:
    env = make_vec_env(env_id, env_kwargs={
        "render_mode": "rgb_array",  
    })
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1,
        tensorboard_log=f"data/runs/{env_id}/",
    )
    try:
        model.learn(total_timesteps=int(2e6), log_interval=10, progress_bar=True)
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        model.save(model_path)
        env.save(env_path)
    del model, env  # Clean up to free memory

def fine_tune(env_id: str, model_path: str, env_path: str, n_timesteps: int = int(1e6)) -> None:
    env = make_vec_env(env_id, env_kwargs={
        "render_mode": "rgb_array",  
    })
    env = VecNormalize.load(env_path, env)
    env.training = True  # Enable training mode for fine-tuning
    env.norm_reward = True  # Enable reward normalization for fine-tuning
    model = PPO.load(model_path, env=env)
    model.policy.train()  # Set the policy to training mode
    model.learn(total_timesteps=n_timesteps, log_interval=10, progress_bar=True)
    model.save(model_path)
    del model, env  # Clean up to free memory

def evaluate(env_id: str, model_path: str, env_path: str, n_episodes: int = 10) -> None:
    env = make_vec_env(env_id, env_kwargs={
            "render_mode": "human",  
            })
    env = VecNormalize.load(env_path, env)
    env.training = False  # Disable training mode for evaluation
    env.norm_reward = False  # Disable reward normalization for evaluation
    model = PPO.load(model_path, env=env)
    model.policy.eval()  # Set the policy to evaluation mode
    mean_reward, std_reward = evaluate_policy(model.policy, env, n_eval_episodes=n_episodes, deterministic=True)
    print(f"After fine-tuning, average reward over {n_episodes} episodes: {mean_reward}")
    del model, env  # Clean up to free memory

if __name__ == "__main__":
    env_id = "PandaUPushEnv-v0"
    model_path = f"data/models/{env_id}"
    env_path = model_path + "_env.pkl"
    # train(env_id, model_path, env_path)
    fine_tune(env_id, model_path, env_path, n_timesteps=int(5e5))
    # evaluate(env_id, model_path, env_path, n_episodes=10)
    

    # model = PPOReg("MultiInputPolicy", env, verbose=1,
    #                tensorboard_log=f"data/runs/{env_id}/",
    #                max_grad_norm=0.01,
    #                clip_range=0.01,
    #                 n_steps=1024,
    #                 policy_kwargs={
    #                     "net_arch": [32, 32],
    #                     "log_std_init": -10.0,
    #                 },
    #             )
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # pretrained_policy_path = "data/panda_bc_policy.pth"
    # saved_variables = torch.load(pretrained_policy_path, map_location=device, weights_only=False)
    # model.set_pretrained_policy(saved_variables, log_std_init=-6.0, reg_coef=0.1, reset_policy=True)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = PPOReg.load(model_path, env=env, device=device)
    # pretrained_policy_path = "data/panda_bc_policy.pth"
    # saved_variables = torch.load(pretrained_policy_path, map_location=device, weights_only=False)
    # model.set_pretrained_policy(saved_variables, log_std_init=-6.0, reg_coef=0.5, reset_policy=False)
    # model.learn(total_timesteps=int(1e6), log_interval=5, progress_bar=True)
    
    # for episode in range(n_episodes):
    #     obs = env.reset()
    #     done, truncated = False, False
    #     total_reward = 0.0
    #     while not (done or truncated):
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, dones, info = env.step(action)
    #         total_reward += reward
    #     print(f"Episode {episode + 1} finished with total reward: {total_reward}")

