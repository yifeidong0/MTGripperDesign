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
                  'perturb': args.perturb,
                  }
    env = gym.make(env_id, **env_kwargs)
    # check_env(env)
    
    model = None
    if env_id == 'UCatchSimulationEnv-v0':
        model = PPO.load("/home/yif/Documents/git/MTGripperDesign/results/paper/catch/lowlevel/with_robustness_reward/1/UCatchSimulationEnv-v0_2024-08-26_13-52-39_747000_steps.zip")
        # model = PPO.load("/home/yif/Documents/git/MTGripperDesign/results/paper/catch/lowlevel/without_robustness_reward/1/UCatchSimulationEnv-v0_2024-08-26_13-52-44_717000_steps.zip")
    elif env_id == 'VPushPbSimulationEnv-v0':
        model = PPO.load("results/models/VPushPbSimulationEnv-v0/VPushPbSimulationEnv-v0_2024-08-23_17-56-58_1000_steps.zip")
    elif env_id == 'PandaUPushEnv-v0':
        model = PPO.load("results/models/PandaUPushEnv-v0/PandaUPushEnv-v0_2024-08-26_10-38-59_809000_steps.zip")
    elif env_id == 'DLRSimulationEnv-v0':
        model = PPO.load("results/models/DLRSimulationEnv-v0/DLRSimulationEnv-v0_2024-08-23_09-38-01_4000_steps.zip")
    
    success_rate = 0
    for episode in range(10):
        obs, _ = env.reset(seed=0)
        print(f"Episode {episode + 1} begins")
        done, truncated = False, False
        while not (done or truncated):
            time.sleep(0.1)
            if model is not None:
                action, state = model.predict(obs)
            else:
                action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            env.render()

        print("Done!" if done else "Truncated.")
        print(f"Episode {episode + 1} finished")
        success_rate += int(done)

    env.close()
    os.system(f"rm -rf asset/202*")
    print(f"Success rate: {success_rate}/{episode + 1}")

if __name__ == "__main__":
    main()