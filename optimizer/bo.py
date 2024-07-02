import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import GPy
import GPyOpt
import os
import sys
from utils.plot_bo import *
from utils.plot_bo_multi import *

class BayesianOptimization:
    def __init__(self, env_type="push", initial_iter=3, max_iter=10, gui=False):
        """
        Initialize the Bayesian Optimization Pipeline with the given parameters.

        Args:
            env_type (str): The type of environment ('push', 'vpush', or 'ucatch').
            initial_iter (int): The number of initial iterations to run.
            max_iter (int): The maximum number of iterations to run.
            gui (bool): Whether to use GUI for the simulations.
        """
        self.env_type = env_type
        self.gui = gui

        # Set bounds and number of tasks based on environment type
        if self.env_type == "push":
            self.x_scale = np.arange(0.1, 1.2, 0.04)
            self.bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.1, 1.2)}]
            self.num_tasks = 2 # number of tasks
        elif self.env_type == "vpush":
            self.x_scale = np.arange(0, np.pi, np.pi/64)
            self.bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, np.pi)}]
            self.num_tasks = 2
        elif self.env_type == "ucatch":
            self.bounds = [{'name': 'd0', 'type': 'continuous', 'domain': (5, 10)}, # design space bounds
                        {'name': 'd1', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'd2', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'alpha0', 'type': 'continuous', 'domain': (np.pi/2, np.pi)},
                        {'name': 'alpha1', 'type': 'continuous', 'domain': (np.pi/2, np.pi)}]
            self.num_tasks = 2

        self.initial_iter = initial_iter
        self.max_iter = max_iter
        self.kernel = self.get_kernel_basic(input_dim=len(self.bounds))
        self.task_counter = {}
        self.setup_bo()

    def get_kernel_basic(self, input_dim=1):
        """
        Create a basic kernel for Gaussian Process.

        Args:
            input_dim (int): The dimensionality of the input space.

        Returns:
            GPy.kern: The basic kernel for GP.
        """
        return GPy.kern.Matern52(input_dim, ARD=1) + GPy.kern.Bias(input_dim)

    def objective(self, x, num_episodes=1):
        """
        Evaluate the objective function for a given design by averaging the scores across all tasks.

        Args:
            x (np.array): The input design.
            num_episodes (int): The number of episodes to run the simulation.

        Returns:
            float: The average score for the given design across all tasks.
        """
        x = x[0]
        scores = []
        for t in range(self.num_tasks):
            if self.env_type == "push":
                from sim.push_sim import ForwardSimulationPlanePush as Simulation
                task = 'ball' if t == 0 else 'box'
                sim = Simulation(task, x, self.gui)
                score = sim.run()
            elif self.env_type == "vpush":
                from sim.vpush_sim import VPushSimulation as Simulation
                task = 'circle' if t == 0 else 'polygon'
                sim = Simulation(task, x, self.gui)
                score = sim.run(num_episodes)
            elif self.env_type == "ucatch":
                from sim.ucatch_sim import UCatchSimulation as Simulation
                task = 'circle' if t == 0 else 'polygon'
                sim = Simulation(task, x, self.gui)
                score = sim.run(num_episodes)
            scores.append(score)
        return np.mean(scores)

    def setup_bo(self):
        """
        Set up the Bayesian Optimization pipeline.
        """
        self.objective = GPyOpt.core.task.SingleObjective(self.objective)
        self.model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False, kernel=self.kernel, noise_var=None)
        self.space = GPyOpt.Design_space(space=self.bounds)
        self.acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(self.space)
        self.initial_design = GPyOpt.experiment_design.initial_design('random', self.space, self.initial_iter)

        self.acquisition = GPyOpt.acquisitions.AcquisitionEI(self.model, self.space, optimizer=self.acquisition_optimizer)
        self.evaluator = GPyOpt.core.evaluators.Sequential(self.acquisition)
        self.bo = GPyOpt.methods.ModularBayesianOptimization(self.model, self.space, self.objective, self.acquisition, self.evaluator, self.initial_design, normalize_Y=False)

    def find_optimal_design(self, resolution=10):
        """
        Find the optimal design by evaluating a grid of points.

        Args:
            resolution (int): The resolution of the grid.

        Returns:
            tuple: The best design, its score, the grid points, the means, and the variances.
        """
        # Extract the bounds for the design space
        bounds = self.space.get_bounds()
        
        # Generate a grid of points across the design space
        grids = [np.linspace(bound[0], bound[1], resolution) for bound in bounds]
        grid_points = np.array(np.meshgrid(*grids)).T.reshape(-1, len(bounds)) # 10000*5
        
        # Predict the mean for each task at all grid points
        means, vars = self.model.predict(grid_points) # 10000*1, 10000*1
        
        # Calculate the average score over all tasks for each design point
        avg_scores = means.flatten()
        
        # Find the design with the highest average score
        best_idx = np.argmax(avg_scores)
        best_design = grid_points[best_idx]
        best_score = avg_scores[best_idx]
        
        return best_design, best_score, grid_points, means, vars

    def run(self):
        """
        Run the Bayesian Optimization pipeline.
        """
        for i in range(1, self.max_iter + 1):
            print("-------------- Iteration: --------------", i)
            self.bo.run_optimization(1)
            print('next_locations: ', self.bo.suggest_next_locations())
            # if self.env_type in ["push", "vpush"]:
            #     plot_bo(self.bo, self.x_scale, i) # TODO: plot bo
    
        # Find the optimal design after the optimization loop
        best_design, best_score, grid_points, means, vars = self.find_optimal_design()
        print(f"Optimal Design: {best_design}, Score: {best_score}")

        # Plot the results
        if self.env_type in ["ucatch"]:
            plot_marginalized_results(grid_points, means, vars, tasks=list(range(self.num_tasks)), optimizer='bo')

if __name__ == "__main__":
    pipeline = BayesianOptimization(env_type="ucatch", initial_iter=10, max_iter=1, gui=0) # vpush, push, ucatch
    pipeline.run()
