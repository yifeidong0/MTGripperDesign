import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import envs
import datetime

def main():
    env_id = 'ScoopSimulationEnv-v0' # VPushSimulationEnv-v0, VPushPbSimulationEnv-v0, UCatchSimulationEnv-v0, ScoopSimulationEnv-v0
    device = 'auto' # 'cpu', 'cuda', 'auto'
    obs_type = 'pose' # image, pose
    env = gym.make(env_id, gui=1, obs_type=obs_type)
    check_env(env)
    # env = make_vec_env(env_id, n_envs=4)    
    
    total_timesteps = int(2e6)
    # current time in yyyy-mm-dd-hh-mm-ss format
    curr_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model = PPO("CnnPolicy" if obs_type == 'image' else "MlpPolicy", 
                env, 
                verbose=1, 
                tensorboard_log=f"results/logs/ppo_{env_id}_tensorboard_{curr_time}/", # TODO: wandb
                device=device)
    model.learn(total_timesteps=total_timesteps, progress_bar=True, log_interval=5)
    model.save(f"results/models/ppo_{env_id}_{total_timesteps}_{curr_time}")    

if __name__ == "__main__":
    main()