import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd

class SuccessScorePlotter:
    def __init__(self, env, optimizer):
        self.env = env
        self.optimizer = optimizer
        self.base_path = f"results/paper/{env}"
        self.labels = [
            # "robustness=0, perturb=0",
            "robustness=0, perturb=1",
            # "robustness=1, perturb=0",
            "robustness=1, perturb=1",
        ]
        # self.colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4',]  # Colors suitable for academic papers
        self.colors = ['#ff7f0e', '#1f77b4',]  # Colors suitable for academic papers
        self.num_seeds = {'catch': 5, 'vpush': 5, 'panda': 4, 'dlr': 5}
        self.num_seed = self.num_seeds[env]
        if self.optimizer == 'mtbo' or self.optimizer == 'bo':
            self.success_rate_name = 'success_score_true'
        elif self.optimizer == 'ga':
            self.success_rate_name = 'success_rate'

    def load_csv_data(self):
        data = {}

        for i in range(1, self.num_seed+1):
            csv_files = sorted(glob(f"{self.base_path}/{i}/{self.env}_{self.optimizer}*1_10simga_0.3robreward.csv"))
            data[i] = {}

            for j, csv_file in enumerate(csv_files):
                df = pd.read_csv(csv_file)
                steps = df['num_iter'].values
                success_scores = df[self.success_rate_name].values
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

            for i in range(1, self.num_seed+1):
                steps = data[i][label]['steps']
                success_scores = data[i][label]['success_scores']

                if len(all_steps) == 0:
                    all_steps = steps
                all_success_scores.append(np.interp(all_steps, steps, success_scores))

            all_success_scores = np.array(all_success_scores)
            mean_values = np.mean(all_success_scores, axis=0)
            std_values = np.std(all_success_scores, axis=0)

            plt.plot(all_steps, mean_values, label=label, color=color)
            # if k == 1 or k == 3:
            print(f"label: {label}")
            print(f"first 5 mean_values: {np.mean(mean_values[:5])}")
            print(f"first 5 std_values: {np.mean(std_values[:5])}") 
            print(f"last 5 mean_values: {np.mean(mean_values[-5:])}")
            print(f"last 5 std_values: {np.mean(std_values[-5:])}")
            plt.fill_between(all_steps, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2)
            k += 1

        plt.xlabel('Iteration')
        plt.ylabel('Test Success Score')
        # plt.xlim(1, 35)
        plt.ylim(0, 1)
        plt.legend()
        plt.title(f'Test Performance of Optimal Design over GA Iterations')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")

    def plot_all(self):
        self.plot_with_mean_std(f'{self.base_path}/{self.optimizer}_design_iterations_1.png')
        # self.plot_with_mean_std(f'{self.optimizer}_design_iterations.png')


if __name__ == "__main__":
    env = 'dlr' # ['catch', 'vpush', 'panda', 'dlr']
    optimizer = 'mtbo' # ['mtbo', 'ga', 'bo']
    plotter = SuccessScorePlotter(env, optimizer)
    plotter.plot_all()
