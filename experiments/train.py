import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import envs

def main():
    env_id = 'VPushPbSimulationEnv-v0' # VPushSimulationEnv-v0, VPushPbSimulationEnv-v0
    device = 'auto' # 'cpu', 'cuda', 'auto'
    obs_type = 'pose' # image, pose
    env = gym.make(env_id, gui=True, obs_type=obs_type)
    check_env(env)
    # env = make_vec_env(env_id, n_envs=4)    
    
    total_timesteps = int(1e6)
    model = PPO("CnnPolicy" if obs_type == 'image' else "MlpPolicy", 
                env, 
                verbose=1, 
                tensorboard_log="results/logs/ppo_box2d_tensorboard/",
                device=device)
    model.learn(total_timesteps=total_timesteps, progress_bar=True, log_interval=5)
    model.save(f"results/models/ppo_box2d_{total_timesteps}")    

if __name__ == "__main__":
    main()
