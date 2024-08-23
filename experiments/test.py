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
    env_kwargs = {'obs_type': args.obs_type, 'using_robustness_reward': args.using_robustness_reward, 'render_mode': args.render_mode}
    env = gym.make(args.env_id, **env_kwargs)
    # check_env(env)
    
    model = None
    if args.env_id == 'UCatchSimulationEnv-v0':
        model = PPO.load("results/models/best_model_ucatch_w_robustness_reward.zip")
    elif args.env_id == 'VPushPbSimulationEnv-v0':
        model = PPO.load("results/models/ppo_VPushPbSimulationEnv-v0_3000000_2024-07-22-16-17-10_with_robustness_reward.zip")
    elif args.env_id == 'PandaPushEnv-v0':
        model = PPO.load("results/models/PandaPushEnv-v0_1000000_2024-08-14_16-52-37_final")
    elif args.env_id == 'DLRSimulationEnv-v0':
        model = PPO.load("results/models/DLRSimulationEnv-v0/saved/DLRSimulationEnv-v0_2024-08-19_22-57-35_2191000_steps.zip")
                         
    for episode in range(10):
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

if __name__ == "__main__":
    main()