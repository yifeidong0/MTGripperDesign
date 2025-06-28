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
    mode = "test_after_finetune"  # "train" or "test_before_finetune" or "test_after_finetune"
    n_eval_episodes = 10
    render_mode = "human" # "human" or "rgb_array"
    env_id = "PandaUPushEnv-v0"
    env = gym.make(env_id, render_mode=render_mode, perturb=1, max_episode_steps=1000)
    env = PandaEnvWrapper(env)
    
    if mode != "test_after_finetune":
        model = PPOReg(policy=DeepPolicy,
                    env=env,
                    verbose=1,
                    tensorboard_log=f"data/runs/{env_id}/",
                    max_grad_norm=0.01,
                    clip_range=0.01,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    )
        # model = PPO(policy=DeepPolicy,
        #             env=env,
        #             verbose=1,
        #             tensorboard_log=f"data/runs/{env_id}/",
        #             max_grad_norm=0.01,
        #             clip_range=0.01,
        #             )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrained_policy_path = "data/panda_bc_policy_rand_morph.pth"
        saved_variables = torch.load(pretrained_policy_path, map_location=device, weights_only=False)
        if isinstance(model, PPOReg):
            model.set_pretrained_policy(saved_variables, log_std_init=-1.0, reg_coef=0.1)
        else:
            model.policy.load_state_dict(saved_variables['state_dict'])
        
        if mode == "test_before_finetune":
            model.policy.eval()  # Set the policy to evaluation mode
            mean_reward, std_reward = evaluate_policy(model.policy, env, n_eval_episodes=n_eval_episodes, deterministic=True)
            print(f"Before fine-tuning, over {n_eval_episodes}, average reward: {mean_reward}, std: {std_reward}")
        else:
            try:
                model.learn(total_timesteps=int(1e6), log_interval=5, progress_bar=True)
            except KeyboardInterrupt:
                print("Training interrupted")
            finally:
                model.save(f"data/models/{env_id}_final_model_perturb.zip")
        env.close()
        
    else:
        model = PPOReg.load(f"data/models/{env_id}_final_model_perturb.zip", env=env, policy=DeepPolicy)
        model.policy.eval()  # Set the policy to evaluation mode
        mean_reward, std_reward = evaluate_policy(model.policy, env, n_eval_episodes=n_eval_episodes, deterministic=True)
        print(f"After fine-tuning, over {n_eval_episodes}, average reward: {mean_reward}, std: {std_reward}")
        env.close()

