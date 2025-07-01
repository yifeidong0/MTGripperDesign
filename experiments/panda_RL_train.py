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
from policy.ppo_reg import PPOReg
from panda_IL import PandaEnvWrapper, DeepPolicy

def main(args):
    algo_name = args.algo  
    train_mode = args.train_mode  
    seed = args.seed 
    set_random_seed(seed, using_cuda=False)  # leave the CUDA randomness for performance
    render_mode = "rgb_array"  # "human" or "rgb_array"
    training_steps = int(1.5e6)  # Total training steps
    env_id = "PandaUPushEnv-v0"
    env = gym.make(env_id, render_mode=render_mode, run_id=f"{algo_name}_{train_mode}_{seed}")
    env = PandaEnvWrapper(env, stat_file="data/obs_statistics.pkl")

    if algo_name == "PPO":
        model = PPO(policy=DeepPolicy,
                    env=env,
                    verbose=1,
                    tensorboard_log=f"data/runs/panda",
                    max_grad_norm=0.01,
                    clip_range=0.01,
                    seed=seed,
                    )
    elif algo_name == "PPOReg":
        model = PPOReg(policy=DeepPolicy,
                        env=env,
                        verbose=1,
                        tensorboard_log=f"data/runs/panda",
                        max_grad_norm=0.01,
                        clip_range=0.01,
                        seed=seed,
                        )
        
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")
    
    if train_mode == "finetune":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrained_policy_path = "data/panda_bc_policy_rand_morph.pth"
        saved_variables = torch.load(pretrained_policy_path, map_location=device, weights_only=False)
        if isinstance(model, PPOReg):
            model.set_pretrained_policy(saved_variables, log_std_init=-1.0, reg_coef=0.1)
        else:
            model.policy.load_state_dict(saved_variables['state_dict'])
            
    elif train_mode == "from_scratch" and algo_name == "PPOReg":
        env.close()
        exit(0)
        
    try:
        model.learn(total_timesteps=training_steps, log_interval=5, progress_bar=True, tb_log_name=f"{algo_name}_{train_mode}_{seed}")
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        model.save(f"data/models/{env_id}_{algo_name}_{train_mode}_{seed}.zip")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panda RL Training Script")
    parser.add_argument("--algo", type=str, default="PPOReg", choices=["PPO", "PPOReg"], help="RL algorithm to use")
    parser.add_argument("--train_mode", type=str, default="fine_tune", choices=["from_scratch", "finetune"], help="Training mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    main(args)