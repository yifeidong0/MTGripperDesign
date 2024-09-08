import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Define environments and combination IDs
environments = ["catch", "vpush", "panda", "dlr",]

# Perturbation levels for each environment
perturb_levels = {
    "catch": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "vpush": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "panda": [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2],
    "dlr": [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4]
}

# Labels and colors for combinations
labels = [
    "robustness=1, perturb=1",
    "robustness=0, perturb=1",
    "robustness=1, perturb=0",
    "robustness=0, perturb=0",
]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ]  # Colors suitable for academic papers

# Function to extract success rates from individual CSV files
def extract_success_rates(csv_files):
    success_rates = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        success_rates.extend(df['Success'].values)
    return np.array(success_rates)

# Plot setup
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Success Rate vs Perturbation Level for Different Environments and Combinations', fontsize=16)

# Plot for each environment
for i, env in enumerate(environments):
    ax = axs[i // 2, i % 2]
    
    # Plot 4 curves for each combination_id
    for combination_id in range(1, 5):
        mean_success_rates = []
        std_success_rates = []

        # For each perturbation level, extract success rates from the corresponding CSV files
        for perturb in perturb_levels[env]:
            # Get all the relevant CSV files for the given environment, perturbation level, and combination ID
            csv_pattern = f"results/paper/perturb/{env}_perturb_{perturb}_combination_{combination_id}_*.csv"
            csv_files = glob.glob(csv_pattern)

            # Extract success rates from individual CSV files
            success_rates = extract_success_rates(csv_files)

            # Calculate mean and 0.5*std deviation
            mean_success_rate = np.mean(success_rates) if len(success_rates) > 0 else 0
            std_success_rate = 0.5 * np.std(success_rates) if len(success_rates) > 0 else 0

            mean_success_rates.append(mean_success_rate)
            std_success_rates.append(std_success_rate)

        # Plot the curve with shaded area for std deviation (0.5*std)
        mean_success_rates = np.array(mean_success_rates)
        std_success_rates = np.array(std_success_rates)
        ax.plot(perturb_levels[env], mean_success_rates, label=labels[combination_id - 1], color=colors[combination_id - 1])
        # ax.fill_between(perturb_levels[env], mean_success_rates - std_success_rates, mean_success_rates + std_success_rates, color=colors[combination_id - 1], alpha=0.2)

    # Set title and labels
    ax.set_title(f'Success Rate for {env.capitalize()}')
    ax.set_xlabel('Perturbation Level')
    ax.set_ylabel('Mean Success Rate')
    ax.set_ylim([0, 1])
    ax.legend(loc='lower left')
    ax.grid(True)

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('results/paper/perturb/0_plot.png')
plt.show()
