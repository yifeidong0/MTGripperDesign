import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import envs

def main():    
    env_id = 'VPushPbSimulationEnv-v0' # VPushSimulationEnv-v0, VPushPbSimulationEnv-v0, UCatchSimulationEnv-v0
    env = gym.make(env_id, gui=True, obs_type='pose')
    model = PPO.load("results/models/ppo_VPushPbSimulationEnv-v0_2000000_2024-07-17-11-00-39.zip")

    for episode in range(12):
        obs, _ = env.reset(seed=0)
        print(f"Episode {episode + 1} begins")
        done, truncated = False, False
        while not (done or truncated):
            action = model.predict(obs)[0]
            obs, reward, done, truncated, _ = env.step(action)
            env.render()

        print("Done!" if done else "Truncated.")
        print(f"Episode {episode + 1} finished")

    env.close()

if __name__ == "__main__":
    main()
    