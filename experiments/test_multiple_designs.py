import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import envs
import csv
import ast
import numpy as np
import matplotlib.pyplot as plt

def evaluate_design(design=[1,], num_episodes=10):
    env_id = 'VPushPbSimulationEnv-v0' # VPushSimulationEnv-v0, VPushPbSimulationEnv-v0, UCatchSimulationEnv-v0
    env = gym.make(env_id, gui=0, obs_type='pose')
    if env_id == 'UCatchSimulationEnv-v0':
        model = PPO.load("results/models/ppo_UCatchSimulationEnv-v0_1000000_2024-07-16-14-51-42.zip")
        robustness_score_weight = 0.1
        num_task = 2
    elif env_id == 'VPushPbSimulationEnv-v0':
        model = PPO.load("results/models/ppo_VPushPbSimulationEnv-v0_3000000_2024-07-22-16-17-10_with_robustness_reward.zip")
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

def read_designs(file_path):
    """
    Read num_iter, best_design, best_score from a CSV file.
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        designs = []
        for row in reader:
            # Evaluate the string as a list
            design = ast.literal_eval(row[2])
            designs.append(design)
    return designs

def write_scores(file_path, scores):
    """
    Write scores to an existing CSV file. 
    Add 3 new columns with headers 'score_true', 'success_score_true', 'robustness_score_true'.
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)  # Read all rows into memory

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers + ['score_true', 'success_score_true', 'robustness_score_true'])
        for i, row in enumerate(rows):
            writer.writerow(row + list(scores[i]))
    

def main(env_name, num_runs, num_episodes):
    # Load designs from CSV files
    file_paths = []
    for r in range(num_runs):
        file_path = f"results/csv/{env_name}_mtbo_results_{r}.csv"
        file_paths.append(file_path)
    
    # Evaluate designs
    for file_path in file_paths:
        designs = read_designs(file_path)
        print("Designs:", designs)
        scores = []
        for design in designs:
            scores.append(evaluate_design(design, num_episodes=num_episodes))
        print("scores:", scores)

        # Write scores to a CSV file
        write_scores(file_path, scores)

def main_plot(env_name, num_runs):
    """
    Plot scores v.s. number of episodes so far
    """
    # Load data from CSV files
    file_paths = []
    for r in range(num_runs):
        file_path = f"results/csv/{env_name}_mtbo_results_{r}.csv"
        file_paths.append(file_path)

    score_true_runs = []
    score_estimated_runs = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            num_episodes_so_far = []
            score_true = []
            scores_estimated = []

            for row in reader:
                # Evaluate the string as a list
                epi = float(row[1])
                score_estimated = float(row[3])
                score = float(row[4])
                num_episodes_so_far.append(epi)
                scores_estimated.append(score_estimated)
                score_true.append(score)

            score_estimated_runs.append(scores_estimated)
            score_true_runs.append(score_true) # num_runs * num_mtbo_iter

    # Compute mean and std of scores using np
    score_true_mean = np.mean(score_true_runs, axis=0)
    score_true_std = np.std(score_true_runs, axis=0)
    score_estimated_mean = np.mean(score_estimated_runs, axis=0)
    score_estimated_std = np.std(score_estimated_runs, axis=0)

    # Plot scores (x-log scale)
    plt.figure()
    plt.plot(num_episodes_so_far, score_true_mean, label='True total score')
    plt.fill_between(num_episodes_so_far, score_true_mean - score_true_std, score_true_mean + score_true_std, alpha=0.2)
    plt.plot(num_episodes_so_far, score_estimated_mean, label='Estimated total score')
    plt.fill_between(num_episodes_so_far, score_estimated_mean - score_estimated_std, score_estimated_mean + score_estimated_std, alpha=0.2)
    plt.yscale('linear')
    plt.xscale('log')
    plt.xlabel('Number of episodes')
    plt.ylabel('Total score')
    plt.legend()
    plt.savefig(f"results/plots/{env_name}_mtbo_total_scores.png")


if __name__ == "__main__":
    num_runs = 10
    num_episodes = 10
    env_name = 'vpush' # 'vpush', 'ucatch'

    # main(env_name, num_runs, num_episodes)

    main_plot(env_name, num_runs)