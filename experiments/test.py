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
              'panda':'PandaUPushEnv-v0',}
    env_id = env_ids[args.env_id]
    env_kwargs = {'obs_type': args.obs_type, 
                  'using_robustness_reward': args.using_robustness_reward, 
                  'render_mode': args.render_mode,
                  'perturb': args.perturb,
                  'reward_type': args.reward_type,
                  }
    env = gym.make(env_id, **env_kwargs)
    # check_env(env)
    
    model = None
    if env_id == 'UCatchSimulationEnv-v0':
        model = PPO.load("results/paper/catch/1/UCatchSimulationEnv-v0_5000000_2024-08-28_07-45-29_final.zip")
    elif env_id == 'VPushPbSimulationEnv-v0':
        model = PPO.load("results/paper/vpush/1/VPushPbSimulationEnv-v0_2024-08-29_20-24-50_1833000_steps.zip")
    elif env_id == 'PandaUPushEnv-v0':
        # model = PPO.load("results/models/PandaUPushEnv-v0/2024-09-11_01-10-53_pkiqsoum_42_True_False/pkiqsoum_1750000_steps.zip")
        model = PPO.load("wandb/run-20240912_171137-eciuk3p8/files/results/models/PandaUPushEnv-v0/2024-09-12_17-11-37_eciuk3p8_1_True_False/best_model_506118_steps_0.5700.zip")
        # model = None
    elif env_id == 'DLRSimulationEnv-v0':
        model = PPO.load("results/models/DLRSimulationEnv-v0/2024-09-04_15-47-56_z0v28fzm_2_False_True/z0v28fzm_1720000_steps.zip")
    
    success_rate = 0
    for episode in range(10):
        obs, _ = env.reset(seed=1)
        print(f"Episode {episode + 1} begins")
        done, truncated = False, False
        while not (done or truncated):
            if model is not None:
                action, state = model.predict(obs)
            else:
                action = env.action_space.sample()
            time.sleep(0.1)
            obs, reward, done, truncated, _ = env.step(action)
            time.sleep(0.1)
            env.render()

        print("Done!" if done else "Truncated.")
        print(f"Episode {episode + 1} finished")
        success_rate += int(done)

    env.close()
    os.system(f"rm -rf asset/202*")
    print(f"Success rate: {success_rate}/{episode + 1}")

if __name__ == "__main__":
    main()