import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

import envs
from experiments.args_utils import get_args

def main():
    args = get_args()
    env_ids = {'vpush':'VPushPbSimulationEnv-v0', 
              'catch':'UCatchSimulationEnv-v0',
              'dlr':'DLRSimulationEnv-v0',
              'panda':'PandaUPushEnv-v0'}
    env_id = env_ids[args.env_id]
    env_kwargs = {'obs_type': args.obs_type, 
                  'using_robustness_reward': args.using_robustness_reward, 
                  'render_mode': args.render_mode,
                  'time_stamp': args.time_stamp,
                  }
    env = gym.make(env_id, **env_kwargs)
    # check_env(env)
    
    model = None
    if env_id == 'UCatchSimulationEnv-v0':
        model = PPO.load("results/models/UCatchSimulationEnv-v0/UCatchSimulationEnv-v0_2024-08-23_18-19-42_1000_steps.zip")
    elif env_id == 'VPushPbSimulationEnv-v0':
        model = PPO.load("results/models/VPushPbSimulationEnv-v0/VPushPbSimulationEnv-v0_2024-08-23_17-56-58_1000_steps.zip")
    elif env_id == 'PandaUPushEnv-v0':
        model = PPO.load("results/models/PandaUPushEnv-v0/PandaUPushEnv-v0_2024-08-23_20-15-08_3000_steps.zip")
    elif env_id == 'DLRSimulationEnv-v0':
        model = PPO.load("results/models/DLRSimulationEnv-v0/DLRSimulationEnv-v0_2024-08-23_09-38-01_4000_steps.zip")
                         
    for episode in range(3):
        obs, _ = env.reset(seed=0)
        print(f"Episode {episode + 1} begins")
        done, truncated = False, False
        while not (done or truncated):
            if model is not None:
                action, state = model.predict(obs)
            else:
                action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            env.render()

        print("Done!" if done else "Truncated.")
        print(f"Episode {episode + 1} finished")

    env.close()
    os.system(f"rm -rf asset/202*")

if __name__ == "__main__":
    main()