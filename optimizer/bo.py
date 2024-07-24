import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import GPy
import GPyOpt
import os
import sys
from utils.plot_bo import *
import gymnasium as gym
from stable_baselines3 import PPO
import envs

class BayesianOptimization:
    def __init__(self, env_type="push", initial_iter=3, max_iter=10, policy='rl', num_episodes=4, gui=False):
        """
        Initialize the Bayesian Optimization Pipeline with the given parameters.

        Args:
            env_type (str): The type of environment ('push', 'vpush', or 'ucatch').
            initial_iter (int): The number of initial iterations to run.
            policy (str): The policy to use for the design evaluation.
            max_iter (int): The maximum number of iterations to run.
            gui (bool): Whether to use GUI for the simulations.
        """
        self.env_type = env_type
        self.policy = policy
        self.num_episodes = num_episodes
        self.gui = gui

        # Set bounds and number of tasks based on environment type
        if self.env_type == "vpush":
            self.x_scale = np.arange(np.pi/12, np.pi*11/12, np.pi/64)
            self.bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (np.pi/12, np.pi*11/12)}]
            self.env_id = 'VPushPbSimulationEnv-v0'
            self.num_tasks = 2
            self.robustness_score_weight = 1.0
            self.model_path = "results/models/ppo_VPushPbSimulationEnv-v0_3000000_2024-07-22-16-17-10_with_robustness_reward.zip"
        elif self.env_type == "ucatch":
            self.bounds = [{'name': 'd0', 'type': 'continuous', 'domain': (5, 10)}, # design space bounds
                        {'name': 'd1', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'd2', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'alpha0', 'type': 'continuous', 'domain': (np.pi/2, np.pi)},
                        {'name': 'alpha1', 'type': 'continuous', 'domain': (np.pi/2, np.pi)}]
            self.num_tasks = 2
            self.robustness_score_weight = 0.1
            self.env_id = 'UCatchSimulationEnv-v0'
            self.model_path = "results/models/best_model_ucatch_w_robustness_reward.zip"

        if self.policy == "heuristic":
            if self.env_type == "vpush":
                from sim.vpush_pb_sim import VPushPbSimulation as Simulation
                self.sim = Simulation('circle', 1, self.gui)
            # elif self.env_type == "scoop":
            #     from sim.scoop_sim import ScoopingSimulation as Simulation
            #     self.sim = Simulation('pillow', [1,1], self.gui)
        elif self.policy == "rl":
            self.env = gym.make(self.env_id, gui=self.gui, obs_type='pose')
            self.rl_model = PPO.load(self.model_path)

        self.initial_iter = initial_iter
        self.max_iter = max_iter
        self.kernel = self.get_kernel_basic(input_dim=len(self.bounds))
        # self.task_counter = {}
        self.setup_bo()

    def get_kernel_basic(self, input_dim=1):
        """
        Create a basic kernel for Gaussian Process.

        Args:
            input_dim (int): The dimensionality of the input space.

        Returns:
            GPy.kern: The basic kernel for GP.
        """
        # return GPy.kern.Matern52(input_dim, ARD=1) + GPy.kern.Bias(input_dim)
        return GPy.kern.Matern52(input_dim, ARD=1, lengthscale=10.0) + GPy.kern.White(input_dim, variance=1.0)

    def objective(self, x):
        """
        Evaluate the objective function for a given design by averaging the scores across all tasks.

        Args:
            x (np.array): The input design.
            num_episodes (int): The number of episodes to run the simulation.

        Returns:
            float: The average score for the given design across all tasks.
        """
        print(f"!!!!Objective: {x}")
        x = x[0]
        scores = []
        for t in range(self.num_tasks):
            if self.policy == "heuristic":
                if self.env_type == "vpush":
                    task = 'circle' if int(t) == 0 else 'polygon'
                    self.sim.reset_task_and_design(task, x[0])
                    score = self.sim.run(self.num_episodes)
                elif self.env_type == "ucatch":
                    from sim.ucatch_sim import UCatchSimulation as Simulation
                    task = 'circle' if int(t) == 0 else 'polygon'
                    sim = Simulation(task, x, self.gui)
                    score = sim.run(self.num_episodes)

            elif self.policy == "rl":
                avg_score = 0
                obs, _ = self.env.reset(seed=0)
                for episode in range(self.num_episodes):
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
                    success_score = 0.5 if done else 0.1
                    robustness_score = avg_robustness / num_robustness_step if num_robustness_step > 0 else 0
                    # print(f"Success: {success_score}, Robustness: {robustness_score}")
                    score = success_score + robustness_score
                    avg_score += score
                    print("Done!" if done else "Truncated.")
                score = avg_score / self.num_episodes

            scores.append(score)
        return np.mean(scores)

    def setup_bo(self):
        """
        Set up the Bayesian Optimization pipeline.
        """
        self.objective = GPyOpt.core.task.SingleObjective(self.objective)
        self.model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False, kernel=self.kernel, noise_var=1e-1)
        self.space = GPyOpt.Design_space(space=self.bounds)
        self.acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(self.space)
        self.initial_design = GPyOpt.experiment_design.initial_design('random', self.space, self.initial_iter)

        # self.acquisition = GPyOpt.acquisitions.AcquisitionEI(self.model, self.space, optimizer=self.acquisition_optimizer, jitter=.1)
        self.acquisition = GPyOpt.acquisitions.AcquisitionLCB(self.model, self.space, optimizer=self.acquisition_optimizer, exploration_weight=200)
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

    def save_to_csv(self, filename, csv_buffer):
        """
        Save the results of num_iter, best_design, best_score to a csv file.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            if self.env_type in ["vpush"]:
                f.write("num_iter,num_episodes_so_far,best_design,best_score_estimated\n")
                for line in csv_buffer:
                    f.write(f"{line[0]},{line[1]},{line[2][0]},{line[3]}\n")
            elif self.env_type in ["ucatch",]:
                f.write("num_iter,num_episodes_so_far,best_design_0,best_design_1,best_design_2,best_design_3,best_design_4,best_score_estimated\n")
                for line in csv_buffer:
                    f.write(f"{line[0]},{line[1]},{line[2][0]},{line[2][1]},{line[2][2]},{line[2][3]},{line[2][4]},{line[3]}\n")

    def run(self):
        """
        Run the Bayesian Optimization pipeline.
        """
        # Create a file with unique name
        k = 0
        while True:     
            csv_filename = f"results/csv/{self.env_type}_bo_results_{k}.csv"
            if os.path.exists(csv_filename):
                k += 1
            else:
                break

        csv_buffer = []
        for i in range(1, self.max_iter+1):
            print("-------------- Iteration: --------------", i)
            self.bo.run_optimization(1, eps=-1) # eps=-1: avoid early stopping
            print('next_locations: ', self.bo.suggest_next_locations())
            if self.env_type in ["vpush"]:
                plot_bo(self.bo, self.x_scale, i, id_run=k)
    
            # Find the optimal design after the optimization loop
            best_design, best_score, grid_points, means, vars = self.find_optimal_design()
            print(f"Optimal Design: {best_design}, Score: {best_score}")
            num_episodes_so_far = self.num_episodes * (i + self.initial_iter) * self.num_tasks
            csv_buffer.append([i, num_episodes_so_far, best_design, best_score])
        
        # Save intermediate designs to a csv file
        self.save_to_csv(csv_filename, csv_buffer)

        # Plot the results
        # if self.env_type in ["ucatch"]:
        #     plot_marginalized_results(grid_points, means, vars, tasks=list(range(self.num_tasks)), optimizer='bo')

if __name__ == "__main__":
    num_run = 10
    for r in range(num_run):
        pipeline = BayesianOptimization(env_type="ucatch", # vpush, ucatch
                                        initial_iter=1, 
                                        max_iter=50, 
                                        policy='rl', 
                                        num_episodes=4, 
                                        gui=0)
        pipeline.run()
        pipeline.env.close()
