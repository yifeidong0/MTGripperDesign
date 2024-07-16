import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import envs

            
def main():    
    env_id = 'UCatchSimulationEnv-v0' # VPushSimulationEnv-v0, VPushPbSimulationEnv-v0, UCatchSimulationEnv-v0
    env = gym.make(env_id, gui=True, obs_type='pose')
    
    model = PPO.load("results/models/ppo_box2dUCatchSimulationEnv-v0_500000_2024-07-16-10-57-45.zip")

    obs, _ = env.reset(seed=0)
    for episode in range(1):
        print(f"Episode {episode + 1} begins")
        done, truncated = False, False
        while not (done or truncated):
            # action = env.action_space.sample()
            action = model.predict(obs)[0]
            obs, reward, done, truncated, _ = env.step(action)
            env.render()
                
        print("Done!" if done else "Truncated")
        print(f"Episode {episode + 1} finished")

    env.close()

if __name__ == "__main__":
    main()
    