import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
import tikzplotlib  # Import tikzplotlib for saving TikZ plots
import pandas as pd

class TrainingCurvePlotter:
    def __init__(self, env):
        self.env = env
        self.base_path = f"results/paper/{env}"
        self.labels = [
            "robustness=1, perturb=1",
            "robustness=0, perturb=1",
            "robustness=1, perturb=0",
            "robustness=0, perturb=0"
        ]
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        self.num_seeds = {'catch': 6, 'vpush': 5, 'panda': 6, 'dlr': 8}
        self.num_seed = self.num_seeds[env]
        self.xlims = {'catch': [0.25e6, 4.9e6], 'vpush': [0, 2e6], 'panda': [0, 1.25e6], 'dlr': [0, 1.25e6]}
        self.xlim = self.xlims[env]
        self.ylim_srs = {'catch': [0.0, 0.95], 'vpush': [0.1, 0.8], 'panda': [0.1, 0.75], 'dlr': [0.0, 1]}
        self.ylim_sr = self.ylim_srs[env]
        self.ylim_rews = {'catch': [0, 280], 'vpush': [0, 120], 'panda': [0, 90], 'dlr': [-50, 450]}
        self.ylim_rew = self.ylim_rews[env]

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

        for i in range(1, self.num_seed):
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

    def plot_with_mean_std(self, ax, metric, ylabel=None):
        data = self.load_all_data()
        k = 0
        for label, color in zip(self.labels, self.colors):
            all_steps = []
            all_values = []

            for i in range(1, self.num_seed):
                steps = data[i][label]['steps']
                values = data[i][label][metric]

                if len(all_steps) == 0:
                    all_steps = steps
                all_values.append(np.interp(all_steps, steps, values))

            all_values = np.array(all_values)
            mean_values = np.mean(all_values, axis=0)
            std_values = np.std(all_values, axis=0)

            ax.plot(all_steps, mean_values, label=label, color=color)
            if k == 0 or k == 1:
                ax.fill_between(all_steps, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2)
            k += 1

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim_sr) if metric == 'success_rates' else ax.set_ylim(self.ylim_rew)
        
        if ylabel:
            ax.set_ylabel(ylabel)
        # ax.set_xlabel('Training Steps')

    def plot_envs_in_subfigures(self, envs, save_path):
        fig, axs = plt.subplots(1, 4, figsize=(24, 5), sharey=True)  # Increase figure width to 24 for more stretch
        plt.subplots_adjust(wspace=0.1)  # Reduce horizontal space between subplots

        for i, env in enumerate(envs):
            # Re-initialize environment settings for each subplot
            self.env = env  # Set the environment for each subplot
            self.base_path = f"results/paper/{env}"
            self.num_seed = self.num_seeds[env]
            self.xlim = self.xlims[env]
            self.ylim_sr = self.ylim_srs[env]  # Use specific ylim for each environment
            self.ylim_rew = self.ylim_rews[env]

            # Plot the success rates for each environment
            if i == 0:
                self.plot_with_mean_std(axs[i], 'success_rates', ylabel='Success Rate')  # Only show ylabel on the leftmost plot
            else:
                self.plot_with_mean_std(axs[i], 'success_rates')

            # Show the legend only on the rightmost subplot
            # if i == len(envs) - 1:
            #     axs[i].legend(loc='lower right')

        # Make the layout tight and save as PNG
        plt.tight_layout()
        plt.savefig(save_path)

        # Save the figure as TikZ .tex file with wider subplots
        tikz_save_path = save_path.replace('.png', '.tex')
        tikzplotlib.save(tikz_save_path, axis_height="5cm", axis_width="5.5cm", standalone=False)  # Increase axis_width for clarity

        plt.close()


    def plot_all(self):
        self.plot_with_mean_std('success_rates', 'Success Rate', f'results/paper/figures/robustness/{self.env}_success_rate.png')
        self.plot_with_mean_std('rewards', 'Episode Reward Mean', f'results/paper/figures/robustness/{self.env}_ep_rew_mean.png')

    
if __name__ == "__main__":
    envs = ['catch', 'vpush', 'panda', 'dlr']
    plotter = TrainingCurvePlotter(envs[0])  # Initialize with the first environment
    plotter.plot_envs_in_subfigures(envs, 'results/paper/figures/robustness/envs_success_rate.png')
    # plotter.plot_all()
