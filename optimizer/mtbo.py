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
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import envs
import time

class BayesianOptimizationMultiTask:
    def __init__(self, env_type="push", initial_iter=3, max_iter=10, policy='rl', gui=False):
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
        self.policy = policy # "heuristic" or "rl"

        # Set bounds and number of tasks based on environment type
        if self.env_type == "push":
            self.x_scale = np.arange(0.1, 1.2, 0.04)
            self.bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.1, 1.2)},
                        {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2 # number of tasks
        elif self.env_type == "vpush":
            self.x_scale = np.arange(0, np.pi, np.pi/64)
            self.bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, np.pi)},
                        {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2
            self.robustness_score_weight = 1.0
            self.env_id = 'VPushPbSimulationEnv-v0'
            self.model_path = "results/models/ppo_VPushPbSimulationEnv-v0_2000000_2024-07-17-11-00-39.zip"
        elif self.env_type == "vpush-frictionless":
            self.x_scale = np.arange(0, np.pi, np.pi/64)
            self.bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, np.pi)},
                        {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2
        elif self.env_type == "ucatch":
            self.bounds = [{'name': 'd0', 'type': 'continuous', 'domain': (5, 10)}, # design space bounds
                        {'name': 'd1', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'd2', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'alpha0', 'type': 'continuous', 'domain': (np.pi/2, np.pi)},
                        {'name': 'alpha1', 'type': 'continuous', 'domain': (np.pi/2, np.pi)},
                        {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2
            self.robustness_score_weight = 0.1
            self.env_id = 'UCatchSimulationEnv-v0'
            self.model_path = "results/models/ppo_UCatchSimulationEnv-v0_1000000_2024-07-16-14-51-42.zip"
        elif self.env_type == "scoop":
            self.bounds = [{'name': 'c0', 'type': 'continuous', 'domain': (0.5, 2)},
                        {'name': 'c1', 'type': 'continuous', 'domain': (0.2, 1.3)},
                        {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2

        if self.policy == "heuristic":
            if self.env_type == "vpush":
                from sim.vpush_pb_sim import VPushPbSimulation as Simulation
                self.sim = Simulation('circle', 1, self.gui)
            elif self.env_type == "scoop":
                from sim.scoop_sim import ScoopingSimulation as Simulation
                self.sim = Simulation('pillow', [1,1], self.gui)
        elif self.policy == "rl":
            self.env = gym.make(self.env_id, gui=self.gui, obs_type='pose')
            self.rl_model = PPO.load(self.model_path)
        
        self.initial_iter = initial_iter
        self.max_iter = max_iter
        self.kernel = self.get_kernel_mt(input_dim=len(self.bounds)-1, num_outputs=self.num_outputs)
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

    def get_kernel_mt(self, input_dim=1, num_outputs=2):
        """
        Create a multi-task kernel for Gaussian Process.

        Args:
            input_dim (int): The dimensionality of the input space.
            num_outputs (int): The number of outputs (tasks).

        Returns:
            GPy.util.multioutput.ICM: The multi-task kernel for GP.
        """
        return GPy.util.multioutput.ICM(input_dim, num_outputs, self.get_kernel_basic(input_dim), W_rank=1, name='ICM')

    def mt_objective(self, xt, num_episodes=1):
        """
        Evaluate the multi-task objective function.

        Args:
            xt (list): The input design and task.
            num_episodes (int): The number of episodes to run the simulation.

        Returns:
            float: The score for the given design and task.
        """
        print('xt: ', xt)
        assert len(xt) == 1
        xt = xt[0]
        assert xt[-1] == 0.0 or xt[-1] == 1.0
        x, t = xt[:-1], int(xt[-1])
        self.task_counter[t] = self.task_counter.get(t, 0) + 1
        
        if self.policy == "heuristic":
            if self.env_type == "push":
                from sim.push_sim import ForwardSimulationPlanePush as Simulation
                task = 'ball' if int(t) == 0 else 'box'
                sim = Simulation(task, x, self.gui)
                score = sim.run()
            elif self.env_type == "vpush":
                task = 'circle' if int(t) == 0 else 'polygon'
                self.sim.reset_task_and_design(task, x[0])
                score = self.sim.run(num_episodes)
            elif self.env_type == "vpush-frictionless":
                from sim.vpush_sim import VPushSimulation as Simulation
                task = 'circle' if int(t) == 0 else 'polygon'
                sim = Simulation(task, x, self.gui)
                score = sim.run(num_episodes)
            elif self.env_type == "ucatch":
                from sim.ucatch_sim import UCatchSimulation as Simulation
                task = 'circle' if int(t) == 0 else 'polygon'
                sim = Simulation(task, x, self.gui)
                score = sim.run(num_episodes)
            elif self.env_type == "scoop":
                task = 'bread' if int(t) == 0 else 'pillow'
                self.sim.reset_task_and_design(task, x)
                score = self.sim.run(num_episodes)
        elif self.policy == "rl":
            avg_score = 0
            obs, _ = self.env.reset(seed=0)
            for episode in range(num_episodes):
                obs, _ = self.env.reset_task_and_design(t, x, seed=0)
                done, truncated = False, False
                avg_robustness = 0
                num_robustness_step = 0
                while not (done or truncated):
                    action = self.rl_model.predict(obs)[0]
                    obs, reward, done, truncated, info = self.env.step(action)
                    if info['robustness'] is not None and info['robustness'] > 0:
                        num_robustness_step += 1
                        avg_robustness += info['robustness'] * self.robustness_score_weight
                    self.env.render()
                success_score = 1 if done else 0
                robustness_score = avg_robustness / num_robustness_step if num_robustness_step > 0 else 0
                print(f"Success: {success_score}, Robustness: {robustness_score}")
                score = success_score + robustness_score
                avg_score += score
                print("Done!" if done else "Truncated.")
            score = avg_score / num_episodes
            # time.sleep(1)

        return score

    def cost_mt(self, xt):
        """
        Calculate the cost and gradients for the given input.

        Args:
            xt (np.array): The input design and task.

        Returns:
            tuple: The costs and gradients for the input.
        """
        C0, C1 = 1.0, 1.0
        t2c = {0: C0, 1: C1}
        t2g = {0: [0.0, (C1 - C0)], 1: [0.0, (C1 - C0)]}
        costs = np.array([[t2c[int(t)]] for t in xt[:, -1]])
        gradients = np.array([t2g[int(t)] for t in xt[:, -1]])
        return costs, gradients

    def setup_bo(self):
        """
        Set up the Bayesian Optimization pipeline.
        """
        self.objective = GPyOpt.core.task.SingleObjective(self.mt_objective)
        self.model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False, kernel=self.kernel, noise_var=None)
        self.space = GPyOpt.Design_space(space=self.bounds)
        self.acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(self.space)
        self.initial_design = GPyOpt.experiment_design.initial_design('random', self.space, self.initial_iter)

        from GPyOpt.acquisitions.base import AcquisitionBase
        class MyAcquisition(AcquisitionBase):
            analytical_gradient_prediction = False

            def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2):
                """
                Initialize the custom acquisition function.

                Args:
                    model (GPyOpt.models.GPModel): The GP model.
                    space (GPyOpt.Design_space): The design space.
                    optimizer (GPyOpt.optimization.AcquisitionOptimizer): The optimizer.
                    cost_withGradients (function): The cost function with gradients.
                    exploration_weight (float): The weight for exploration in acquisition.
                """
                self.optimizer = optimizer
                super(MyAcquisition, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
                self.exploration_weight = exploration_weight

            def _compute_acq(self, x):
                """
                Compute the acquisition function.

                Args:
                    x (np.array): The input points for acquisition.

                Returns:
                    np.array: The acquisition values for the input points.
                """
                def eval_scale(task_no):
                    X1 = self.model.model.X
                    X1 = X1[X1[:, -1] == task_no]
                    try:
                        mean, var = self.model.model.predict(X1)
                        std = np.sqrt(var)
                        return (mean.max() - mean.min()) + 1e-6
                    except:
                        return 1.0
                m, s = self.model.predict(x)
                f_acqu = self.exploration_weight * s
                c0 = eval_scale(task_no=0)
                c1 = eval_scale(task_no=1)
                try:
                    f_acqu[x[:, -1] == 1] = f_acqu[x[:, -1] == 1] / c1
                except:
                    pass
                try:
                    f_acqu[x[:, -1] == 0] = f_acqu[x[:, -1] == 0] / c0
                except:
                    pass
                return f_acqu

        self.acquisition = MyAcquisition(self.model, self.space, optimizer=self.acquisition_optimizer, cost_withGradients=self.cost_mt)
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
        grids = [np.linspace(bound[0], bound[1], resolution) for bound in bounds[:-1]]
        grid_points = np.array(np.meshgrid(*grids)).T.reshape(-1, len(bounds) - 1) # 10000*5
        
        # Create the input array for all grid points and all tasks
        xt = np.vstack([np.hstack([grid_points, task * np.ones((grid_points.shape[0], 1))]) 
                        for task in range(self.num_outputs)]) # 20000*6, first 1e4 for task 0, next 1e4 for task 1
        
        # Predict the mean for each task at all grid points
        means, vars = self.model.predict(xt) # 20000*1, 20000*1
        
        # Reshape the means to separate tasks
        means = means.reshape(self.num_outputs, -1, 1) # 2*10000*1 (reshape: cut from middle of the matrix block)
        vars = vars.reshape(self.num_outputs, -1, 1)

        # Calculate the average score over all tasks for each design point
        avg_scores = np.mean(means, axis=0).flatten()
        
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
            self.bo.run_optimization(1) # TODO: slow..2-8 sec
            print('next_locations: ', self.bo.suggest_next_locations()) # 1-4 sec
            if self.env_type in ["push", "vpush"]:
                plot_bo(self.bo, self.x_scale, i)

        # Find the optimal design after the optimization loop
        best_design, best_score, grid_points, means, vars = self.find_optimal_design()
        print(f"Optimal Design: {best_design}, Score: {best_score}")

        # Plot the results
        if self.env_type in ["ucatch", "scoop"]:
            plot_marginalized_results(grid_points, means, vars, tasks=list(range(self.num_outputs)), optimizer='mtbo')


if __name__ == "__main__":
    pipeline = BayesianOptimizationMultiTask(env_type="ucatch", # vpush, (vpush-frictionless, push), ucatch, scoop
                                             initial_iter=10, 
                                             max_iter=5, 
                                             policy='rl',
                                             gui=1) 
    pipeline.run()
