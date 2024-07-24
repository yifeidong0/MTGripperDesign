import numpy as np
import matplotlib.pyplot as plt
from GPyOpt.plotting.plots_bo import plot_acquisition as plot_acq

"""
Multi-dimensional design space plotting for multi-task Bayesian optimization.
"""

def plot_marginalized_results(grid_points, means, vars, tasks=[0, 1], optimizer='mtbo'):
    num_dims = grid_points.shape[1]
    
    for i in range(num_dims):
        plt.figure(figsize=(14, 5))
        
        task_ids = tasks if optimizer=='mtbo' else [0,]
        for j, task in enumerate(task_ids):
            plt.subplot(1, 2, j + 1)
            
            x_i = grid_points[:, i]
            if optimizer=='mtbo':
                mean_i = means[task].flatten()
                var_i = vars[task].flatten()
            elif optimizer=='bo':
                mean_i = means.flatten()
                var_i = vars.flatten()
            
            # Compute marginal scores
            marginalized_scores = []
            marginalized_vars = []
            for val in np.unique(x_i):
                idxs = x_i == val
                marginalized_scores.append(np.mean(mean_i[idxs]))
                marginalized_vars.append(np.mean(var_i[idxs]))
            
            plt.plot(np.unique(x_i), marginalized_scores, label=f'Task {task} Mean')
            plt.fill_between(np.unique(x_i), 
                             np.array(marginalized_scores) - np.array(marginalized_vars), 
                             np.array(marginalized_scores) + np.array(marginalized_vars), 
                             alpha=0.2, label=f'Task {task} Variance')
            
            plt.xlabel(f'Dimension {i}')
            plt.ylabel('Score')
            plt.title(f'Task {task}: Dimension {i} vs Score')
            plt.legend()
            plt.grid()
        
        plt.tight_layout()
        plt.savefig(f"dim_{i}.png")
        