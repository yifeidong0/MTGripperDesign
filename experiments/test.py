import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import envs
import time

def main():
    env_name = 'panda_vpush' # 'vpush', 'ucatch'
    if env_name == 'ucatch':
        model = PPO.load("results/models/best_model_ucatch_w_robustness_reward.zip")
        env_id = 'UCatchSimulationEnv-v0' # VPushPbSimulationEnv-v0, UCatchSimulationEnv-v0
    elif env_name == 'vpush':
        model = PPO.load("results/models/ppo_VPushPbSimulationEnv-v0_3000000_2024-07-22-16-17-10_with_robustness_reward.zip")
        env_id = "VPushPbSimulationEnv-v0"
    elif env_name == 'panda_vpush':
        env_id = "PandaPushEnv-v0"
    env = gym.make(env_id, gui=True, obs_type='pose')
    for episode in range(10):
        obs, _ = env.reset(seed=0)
        print(f"Episode {episode + 1} begins")
        i = 0
        done, truncated = False, False
        while not (done or truncated):
            i += 1
            if i > 20:
                break
            # action = model.predict(obs)[0]
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            env.render()
            time.sleep(1/500)

        print("Done!" if done else "Truncated.")
        print(f"Episode {episode + 1} finished")
        time.sleep(1)

    env.close()

if __name__ == "__main__":
    main()