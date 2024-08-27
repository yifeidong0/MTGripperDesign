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
import time
import imageio
from stable_baselines3 import PPO
import pygame
from experiments.args_utils import get_args

def evaluate_design(env_name, design=[1,], num_episodes=10, gui=0, 
                    create_new_env=1, close_env=1, env=None, writer=None, video_write_freq=4):
    # if env_name == 'vpush':
    #     env_id = 'VPushPbSimulationEnv-v0'
    # elif env_name == 'catch':
    #     env_id = 'UCatchSimulationEnv-v0'
    if create_new_env:
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

    model_with_robustness_reward = 0
    if model_with_robustness_reward:
        model_path = "results/models/UCatchSimulationEnv-v0/UCatchSimulationEnv-v0_2024-08-27_07-00-59_809000_steps.zip" # with robustness reward
    else:
        model_path = "results/models/UCatchSimulationEnv-v0/UCatchSimulationEnv-v0_2024-08-27_07-01-23_737000_steps.zip" # without robustness reward

    if env_id == 'UCatchSimulationEnv-v0':
        model = PPO.load(model_path)
        robustness_score_weight = 0.0
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
            done, truncated = False, False
            avg_robustness = 0
            num_robustness_step = 0
            step_count = 0
            while not (done or truncated):
                action = model.predict(obs)[0]
                obs, reward, done, truncated, info = env.step(action)
                if info['robustness'] is not None and info['robustness'] > 0:
                    num_robustness_step += 1
                    avg_robustness += info['robustness']
                
                # Render the environment and capture the frame
                env.render()
                if writer is not None and step_count % video_write_freq == 0:
                    frame = pygame.surfarray.array3d(pygame.display.get_surface())
                    frame = frame.swapaxes(0, 1)
                    writer.append_data(frame)
                step_count += 1

            success_score = 1.0 if done else 0.0
            robustness_score = avg_robustness / num_robustness_step if num_robustness_step > 0 else 0
            score = success_score + robustness_score * robustness_score_weight
            avg_score += score
            avg_success_score += success_score
            avg_robustness_score += robustness_score
            print("Done!" if done else "Truncated.")
    
    if close_env:
        env.close() 
    
    avg_score /= (num_episodes * num_task)
    avg_success_score /= (num_episodes * num_task)
    avg_robustness_score /= (num_episodes * num_task)
    print(f"Average score: {avg_score}", 
          f"Average success score: {avg_success_score}", 
          f"Average robustness score: {avg_robustness_score}", 
          sep='\n')
    return avg_score, avg_success_score, avg_robustness_score, env


def read_designs(file_path, env_name):
    """
    Read num_iter, best_design, best_score from a CSV file.
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        designs = []
        for row in reader:
            if env_name == 'vpush':
                # Evaluate the string as a list
                # design = ast.literal_eval(row[2])
                design = [float(row[2]),]
            elif env_name == 'catch':
                design = [float(row[k]) for k in range(2, 7)]
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
    
def main(env_name, optimizer, num_runs, num_episodes):
    # Load designs from CSV files
    import glob
    file_paths = []
    for r in range(num_runs):
        file_path = glob.glob(f"results/csv/{env_name}_{optimizer}_results_{r}_*.csv")[0]
        # file_path = "results/csv/catch_mtbo_results_usingrob1_2024-08-27_10-39-49.csv"
        file_path = "results/csv/catch_mtbo_results_usingrob0_2024-08-27_10-40-10.csv"
        file_paths.append(file_path)
    
    # Evaluate designs
    for file_path in file_paths:
        designs = read_designs(file_path, env_name)
        print("Designs:", designs)
        scores = []
        for design in designs:
            scores.append(evaluate_design(env_name, design, num_episodes=num_episodes)[:-1])
        print("scores:", scores)

        # Write scores to a CSV file
        write_scores(file_path, scores)

def qualitative_design_iteration_demo(env_name, optimizer, num_episodes=1, save_video=1):
    # Load designs from CSV files
    id_run = 0
    file_path = f"results/csv/{env_name}_{optimizer}_results_{id_run}.csv"
    
    # Evaluate designs
    designs = read_designs(file_path, env_name)
    create_new_env = 1
    close_env = 0
    env = None

    writer = None
    if save_video:
        writer = imageio.get_writer('simulation.mp4', fps=30)

    for i, design in enumerate(designs):
        if i == len(designs) - 1:
            close_env = 1
        _, _, _, env = evaluate_design(env_name, design, num_episodes=num_episodes, gui=1, 
                                       create_new_env=create_new_env, close_env=close_env, env=env, writer=writer)
        create_new_env = 0
        # time.sleep(1)

    if save_video:
        writer.close()

def main_plot(env_name, optimizer, num_runs, plot_type='test_score_composition'):
    """
    Plot scores v.s. number of episodes so far
    plot_type: 'estimation_accuracy' or 'test_score_composition'
    """
    # Load data from CSV files
    file_paths = []
    for r in range(num_runs):
        file_path = f"results/csv/{env_name}_{optimizer}_results_{r}.csv"
        file_paths.append(file_path)

    score_estimated_runs = []
    score_test_runs = []
    success_score_test_runs = []
    robustness_score_test_runs = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            num_episodes_so_far = []
            scores_estimated = []
            score_test = []
            success_score_test = []
            robustness_score_test = []
            for row in reader:
                # Evaluate the string as a list
                epi = float(row[1])
                if env_name == 'vpush':
                    score_estimated = float(row[3])
                    score = float(row[4])
                    success_score = float(row[5])
                    robustness_score = float(row[6])
                elif env_name == 'ucatch':
                    score_estimated = float(row[7])
                    score = float(row[8])
                    success_score = float(row[9])
                    robustness_score = float(row[10])
                num_episodes_so_far.append(epi)
                scores_estimated.append(score_estimated)
                score_test.append(score)
                success_score_test.append(success_score)
                robustness_score_test.append(robustness_score)

            score_estimated_runs.append(scores_estimated)
            score_test_runs.append(score_test) # num_runs * num_mtbo_iter
            success_score_test_runs.append(success_score_test)
            robustness_score_test_runs.append(robustness_score_test)

    # Compute mean and std of scores using np
    score_estimated_mean = np.mean(score_estimated_runs, axis=0)
    score_estimated_std = np.std(score_estimated_runs, axis=0)
    score_test_mean = np.mean(score_test_runs, axis=0)
    score_test_std = np.std(score_test_runs, axis=0)
    success_score_test_mean = np.mean(success_score_test_runs, axis=0)
    success_score_test_std = np.std(success_score_test_runs, axis=0)
    robustness_score_test_mean = np.mean(robustness_score_test_runs, axis=0)
    robustness_score_test_std = np.std(robustness_score_test_runs, axis=0)

    # Plot scores (x-log scale)
    if plot_type == 'estimation_accuracy':
        plt.figure()
        plt.plot(num_episodes_so_far, score_test_mean, label='design test score')
        plt.fill_between(num_episodes_so_far, score_test_mean - score_test_std, score_test_mean + score_test_std, alpha=0.2)
        plt.plot(num_episodes_so_far, score_estimated_mean, label=f"{optimizer} estimated score")
        plt.fill_between(num_episodes_so_far, score_estimated_mean - score_estimated_std, score_estimated_mean + score_estimated_std, alpha=0.2)
        plt.yscale('linear')
        plt.xscale('log')
        plt.xlabel('Number of episodes')
        plt.ylabel('Total score')
        plt.title(f'Total scores of {env_name} with {optimizer}')
        plt.legend()
        plt.savefig(f"results/plots/{env_name}_{optimizer}_estimation_accuracy.png")
    elif plot_type == 'test_score_composition':
        plt.figure()
        plt.plot(num_episodes_so_far, success_score_test_mean, label='success score')
        plt.fill_between(num_episodes_so_far, success_score_test_mean - success_score_test_std, success_score_test_mean + success_score_test_std, alpha=0.2)
        plt.plot(num_episodes_so_far, robustness_score_test_mean, label='robustness score')
        plt.fill_between(num_episodes_so_far, robustness_score_test_mean - robustness_score_test_std, robustness_score_test_mean + robustness_score_test_std, alpha=0.2)
        plt.plot(num_episodes_so_far, score_test_mean, label='total score')
        plt.fill_between(num_episodes_so_far, score_test_mean - score_test_std, score_test_mean + score_test_std, alpha=0.2)
        plt.yscale('linear')
        plt.xscale('log')
        plt.xlabel('Number of episodes')
        plt.ylabel('Score')
        plt.title(f'Test scores of the best design in {env_name} with {optimizer}')
        plt.legend()
        plt.savefig(f"results/plots/{env_name}_{optimizer}_test_scores_composition.png")
    
def plot_multi_task_sample_efficiency(env_name, optimizers, num_runs):
    score_test_mean_all = []
    score_test_std_all = []
    num_episodes_so_far_all = []
    for optimizer in optimizers:
        # Load data from CSV files
        file_paths = []
        for r in range(num_runs):
            file_path = f"results/csv/{env_name}_{optimizer}_results_{r}.csv"
            file_paths.append(file_path)

        score_test_runs = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                num_episodes_so_far = []
                score_test = []
                for row in reader:
                    # Evaluate the string as a list
                    epi = float(row[1])
                    if env_name == 'vpush':
                        score = float(row[4])
                    elif env_name == 'ucatch':
                        score = float(row[8])
                    num_episodes_so_far.append(epi)
                    score_test.append(score)

                score_test_runs.append(score_test) # num_runs * num_mtbo_iter

        # Compute mean and std of scores using np
        score_test_mean = np.mean(score_test_runs, axis=0)
        score_test_std = np.std(score_test_runs, axis=0)
        score_test_mean_all.append(score_test_mean)
        score_test_std_all.append(score_test_std)
        num_episodes_so_far_all.append(num_episodes_so_far)

    # Plot scores (x-log scale)
    plt.figure()
    for i in range(len(optimizers)):
        plt.plot(num_episodes_so_far_all[i], score_test_mean_all[i], label=f'{optimizers[i]}')
        plt.fill_between(num_episodes_so_far_all[i], score_test_mean_all[i] - score_test_std_all[i], 
                         score_test_mean_all[i] + score_test_std_all[i], alpha=0.2)
    plt.yscale('linear')
    plt.xscale('log')
    plt.xlabel('Number of episodes')
    plt.ylabel('Total score')
    plt.title(f'Multi-task sample efficiency of {env_name} over algorithms')
    plt.legend()
    plt.savefig(f"results/plots/{env_name}_multi_task_sample_efficiency.png")


if __name__ == "__main__":
    num_runs = 1
    num_episodes = 10
    env_name = 'catch' # 'vpush', 'catch'
    optimizer = 'mtbo' # 'mtbo', 'ga', 'bo'

    main(env_name, optimizer, num_runs, num_episodes)

    # plot_type = 'test_score_composition' # 'estimation_accuracy', 'test_score_composition'
    # main_plot(env_name, optimizer, num_runs, plot_type)

    # plot_multi_task_sample_efficiency(env_name, ['mtbo', 'ga', 'bo'], num_runs)

    # num_episodes = 1
    # qualitative_design_iteration_demo(env_name, optimizer, num_episodes)