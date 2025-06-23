import os
import sys
from pathlib import Path

# Add parent directory to path more robustly
sys.path.append(str(Path(__file__).parent.parent))

import envs
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from policy.ppo_reg import PPOReg
from panda_IL import PandaEnvWrapper, DeepPolicy



if __name__ == "__main__":
    mode = "test"  # or "test"
    env_id = "PandaUPushEnv-v0"
    render_mode = "human" if mode == "test" else "rgb_array"
    # render_mode = "rgb_array"
    env = gym.make(env_id, render_mode=render_mode, max_episode_steps=1000)
    env = PandaEnvWrapper(env)
    
    if mode == "train":
        # model = PPOReg(policy=DeepPolicy,
        #             env=env,
        #             verbose=1, 
        #             tensorboard_log=f"data/runs/{env_id}/",
        #             max_grad_norm=0.01,
        #             clip_range=0.01,
        #             n_steps=1024,
        #             )
        model = PPO(policy=DeepPolicy,
                    env=env,
                    verbose=1, 
                    tensorboard_log=f"data/runs/{env_id}/",
                    max_grad_norm=0.01,
                    clip_range=0.01,
                    )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrained_policy_path = "data/panda_bc_policy.pth"
        saved_variables = torch.load(pretrained_policy_path, map_location=device, weights_only=False)
        if isinstance(model, PPOReg):
            model.set_pretrained_policy(saved_variables, log_std_init=-3.0, reg_coef=0.1)
        else:
            model.policy.load_state_dict(saved_variables['state_dict'])
        
        # model.policy.eval()  # Set the policy to evaluation mode
        # n_episodes = 10
        # reward, _ = evaluate_policy(model.policy, env, n_eval_episodes=n_episodes, deterministic=True)
        # print(f"Before fine-tuning, average reward over {n_episodes} episodes: {reward}")
        
        try:
            model.learn(total_timesteps=int(1e6), log_interval=5, progress_bar=True)
        except KeyboardInterrupt:
            print("Training interrupted")
        finally:
            model.save(f"data/models/{env_id}_final_model.zip")
        env.close()
        
    else:
        model = PPOReg.load(f"data/models/{env_id}_final_model.zip", env=env, policy=DeepPolicy)
        model.policy.eval()  # Set the policy to evaluation mode
        n_episodes = 10
        mean_reward, std_reward = evaluate_policy(model.policy, env, n_eval_episodes=n_episodes, deterministic=True)
        env.close()
        print(f"After fine-tuning, average reward over {n_episodes} episodes: {mean_reward}")
