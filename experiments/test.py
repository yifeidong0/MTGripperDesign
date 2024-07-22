import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import envs

def main(design=[1,]):
    num_episodes = 3
    env_id = 'VPushPbSimulationEnv-v0' # VPushSimulationEnv-v0, VPushPbSimulationEnv-v0, UCatchSimulationEnv-v0
    env = gym.make(env_id, gui=True, obs_type='pose')
    if env_id == 'UCatchSimulationEnv-v0':
        model = PPO.load("results/models/ppo_UCatchSimulationEnv-v0_1000000_2024-07-16-14-51-42.zip")
        robustness_score_weight = 0.1
        num_task = 2
    elif env_id == 'VPushPbSimulationEnv-v0':
        model = PPO.load("results/models/ppo_VPushPbSimulationEnv-v0_2000000_2024-07-17-11-00-39.zip")
        robustness_score_weight = 1.0
        num_task = 2
    
    avg_score = 0
    avg_success_score = 0
    avg_robustness_score = 0
    obs, _ = env.reset(seed=0)
    for task in range(num_task):
        for episode in range(num_episodes):
            obs, _ = env.reset_task_and_design(task, design, seed=0) # design: a list of coefficients
            # print(f"Episode {episode + 1} begins")
            done, truncated = False, False
            avg_robustness = 0
            num_robustness_step = 0
            while not (done or truncated):
                action = model.predict(obs)[0]
                obs, reward, done, truncated, info = env.step(action)
                if info['robustness'] is not None and info['robustness'] > 0:
                    num_robustness_step += 1
                    avg_robustness += info['robustness'] * robustness_score_weight
                env.render()
            success_score = 0.5 if done else 0.1
            robustness_score = avg_robustness / num_robustness_step if num_robustness_step > 0 else 0
            # print(f"Success: {success_score}, Robustness: {robustness_score}")
            score = success_score + robustness_score
            avg_score += score
            avg_success_score += success_score
            avg_robustness_score += robustness_score
            print("Done!" if done else "Truncated.")
            # print(f"Episode {episode + 1} finished")
    env.close()

    avg_score /= (num_episodes * num_task)
    avg_success_score /= (num_episodes * num_task)
    avg_robustness_score /= (num_episodes * num_task)
    print(f"Average score: {avg_score}", 
          f"Average success score: {avg_success_score}", 
          f"Average robustness score: {avg_robustness_score}", 
          sep='\n')
    return avg_score, avg_success_score, avg_robustness_score

if __name__ == "__main__":
    main()
    