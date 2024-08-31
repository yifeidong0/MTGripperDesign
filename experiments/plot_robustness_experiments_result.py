import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob

class TrainingCurvePlotter:
    def __init__(self, base_path):
        self.base_path = base_path
        self.labels = [
            "robustness=1, perturb=1",
            "robustness=0, perturb=1",
            "robustness=1, perturb=0",
            "robustness=0, perturb=0"
        ]
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colors suitable for academic papers

    def extract_data(self, event_file):
        steps = []
        success_rates = []
        rewards = []

        for e in tf.compat.v1.train.summary_iterator(event_file):
            for v in e.summary.value:
                if v.tag == 'rollout/success_rate':
                    steps.append(e.step)
                    success_rates.append(v.simple_value)
                elif v.tag == 'rollout/ep_rew_mean':
                    rewards.append(v.simple_value)

        return np.array(steps), np.array(success_rates), np.array(rewards)

    def load_all_data(self):
        data = {}

        for i in range(1, 3):
            model_folders = sorted(glob(f"{self.base_path}/{i}/PPO_*"))
            data[i] = {}

            for j, model_folder in enumerate(model_folders):
                event_file = glob(f"{model_folder}/events.out.tfevents.*")[0]
                steps, success_rates, rewards = self.extract_data(event_file)
                data[i][self.labels[j]] = {
                    'steps': steps,
                    'success_rates': success_rates,
                    'rewards': rewards
                }

        return data

    def plot_with_mean_std(self, metric, ylabel, save_path):
        data = self.load_all_data()

        plt.figure(figsize=(10, 6))
        k = 0
        for label, color in zip(self.labels[::-1], self.colors[::-1]):
            all_steps = []
            all_values = []

            for i in range(1, 3):
                steps = data[i][label]['steps']
                values = data[i][label][metric]

                if len(all_steps) == 0:
                    all_steps = steps
                all_values.append(np.interp(all_steps, steps, values))

            all_values = np.array(all_values)
            mean_values = np.mean(all_values, axis=0)
            std_values = np.std(all_values, axis=0)

            plt.plot(all_steps, mean_values, label=label, color=color)
            if k == 2 or k == 3:
                plt.fill_between(all_steps, mean_values - std_values, mean_values + std_values, color=color, alpha=0.3)
            k += 1

        plt.xlabel('Training Steps')
        plt.ylabel(ylabel)
        plt.xlim(0, 3e6)
        plt.ylim(0.1, 0.72) if metric == 'success_rates' else plt.ylim(0, 80)
        plt.legend()
        plt.title(f'{ylabel} vs Training Steps')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def plot_all(self):
        self.plot_with_mean_std('success_rates', 'Success Rate', f'{self.base_path}/success_rate.png')
        self.plot_with_mean_std('rewards', 'Episode Reward Mean', f'{self.base_path}/ep_rew_mean.png')

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SuccessScorePlotter:
    def __init__(self, base_path):
        self.base_path = base_path
        self.labels = [
            "robustness=1, perturb=1",
            "robustness=0, perturb=1",
            "robustness=1, perturb=0",
            "robustness=0, perturb=0"
        ]
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colors suitable for academic papers

    def load_csv_data(self):
        data = {}

        for i in range(1, 7):  # For folders 1 to 6
            csv_files = sorted(glob(f"{self.base_path}/{i}/*run1.csv"))
            data[i] = {}

            for j, csv_file in enumerate(csv_files):
                df = pd.read_csv(csv_file)
                steps = df['num_iter'].values
                success_scores = df['success_score_true'].values
                data[i][self.labels[j]] = {
                    'steps': steps,
                    'success_scores': success_scores
                }

        return data

    def plot_with_mean_std(self, save_path):
        data = self.load_csv_data()

        plt.figure(figsize=(10, 6))
        k = 0
        for label, color in zip(self.labels, self.colors):
            all_steps = []
            all_success_scores = []

            for i in range(1, 7):
                steps = data[i][label]['steps']
                success_scores = data[i][label]['success_scores']

                if len(all_steps) == 0:
                    all_steps = steps
                all_success_scores.append(np.interp(all_steps, steps, success_scores))

            all_success_scores = np.array(all_success_scores)
            mean_values = np.mean(all_success_scores, axis=0)
            std_values = np.std(all_success_scores, axis=0)

            plt.plot(all_steps, mean_values, label=label, color=color)
            if k == 0 or k == 1:
                plt.fill_between(all_steps, mean_values - std_values, mean_values + std_values, color=color, alpha=0.3)
            k += 1

        plt.xlabel('Iteration')
        plt.ylabel('Test Success Score')
        plt.xlim(1, 50)
        plt.ylim(0, 1)
        plt.legend()
        plt.title(f'Test Performance of Optimal Design over BO Iterations')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def plot_all(self):
        self.plot_with_mean_std(f'{self.base_path}/catch_bo_success_score_run1.png')



if __name__ == "__main__":
    base_path = 'results/paper/panda'
    plotter = TrainingCurvePlotter(base_path)
    # plotter = SuccessScorePlotter(base_path)
    plotter.plot_all()

