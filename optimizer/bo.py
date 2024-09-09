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
from experiments.args_utils import get_args_bo
import wandb
import csv

class BayesianOptimization:
    def __init__(self, initial_iter=1,):
        """
        Initialize the Bayesian Optimization Pipeline with the given parameters.

        Args:
            env_type (str): The type of environment ('push', 'vpush', or 'ucatch').
            initial_iter (int): The number of initial iterations to run.
            policy (str): The policy to use for the design evaluation.
            max_iter (int): The maximum number of iterations to run.
            gui (bool): Whether to use GUI for the simulations.
        """
        self.initial_iter = initial_iter
        self.args = get_args_bo()
        self.env_type = self.args.env
        self.max_iter = self.args.max_iterations
        self.num_episodes = self.args.num_episodes_eval
        self.model_path = self.args.model_path

        # Set bounds and number of tasks based on environment type
        if self.env_type == "vpush":
            self.x_scale = np.arange(np.pi/12, np.pi*11/12, np.pi/64)
            self.bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (np.pi/12, np.pi*11/12)},]
            self.num_outputs = 2
            self.robustness_score_weight = 1.0
        elif self.env_type == "catch":
            self.bounds = [{'name': 'd0', 'type': 'continuous', 'domain': (5, 10)}, # design space bounds
                        {'name': 'd1', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'd2', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'alpha0', 'type': 'continuous', 'domain': (np.pi/2, np.pi)},
                        {'name': 'alpha1', 'type': 'continuous', 'domain': (np.pi/2, np.pi)},]
            self.num_outputs = 2
            self.robustness_score_weight = 0.1
        elif self.env_type == "panda":
            self.num_outputs = 5
            self.bounds = [{'name': 'v_angle', 'type': 'continuous', 'domain': (np.pi/6, np.pi*5/6)}, # design space bounds, make sure it aligns with reset in panda_push_env
                        {'name': 'finger_length', 'type': 'continuous', 'domain': (0.05, 0.12)}, 
                        {'name': 'finger_angle', 'type': 'continuous', 'domain': (-np.pi/3, np.pi/3)},
                        {'name': 'distal_phalanx_length', 'type': 'continuous', 'domain': (0.00, 0.08)},]
            self.robustness_score_weight = 0.3
        elif self.env_type == "dlr":
            self.num_outputs = 11
            self.bounds = [{'name': 'l', 'type': 'discrete', 'domain': list(np.arange(30, 65, 5))},
                           {'name': 'c', 'type': 'discrete', 'domain': list(np.arange(2, 10, 2))},]
            self.robustness_score_weight = 1.0 / 2000.0

        # Create the environment and RL model
        env_ids = {'vpush':'VPushPbSimulationEnv-v0', 
                    'catch':'UCatchSimulationEnv-v0',
                    'dlr':'DLRSimulationEnv-v0',
                    'panda':'PandaUPushEnv-v0'}
        self.env_id = env_ids[self.args.env]
        run_id = wandb.util.generate_id()
        env_kwargs = {'obs_type': self.args.obs_type, 
                        'render_mode': self.args.render_mode,
                        'perturb': self.args.perturb,
                        'run_id': run_id,
                        }
        self.env = gym.make(self.env_id, **env_kwargs)
        self.rl_model = PPO.load(self.model_path)

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
        x = x[0]
        scores = []
        for t in range(self.num_outputs):
            avg_score = 0
            obs, _ = self.env.reset(seed=self.args.random_seed)
            for episode in range(self.num_episodes):
                if self.env_type == "dlr":
                    self.env.num_discrete_tasks = self.num_outputs
                obs, _ = self.env.reset_task_and_design(t, x, seed=self.args.random_seed)
                if self.args.env == "panda" and self.env.is_invalid_design:
                    print(f"Invalid design: {x}")
                    return 0                
                
                done, truncated = False, False
                avg_robustness = 0
                num_robustness_step = 0
                while not (done or truncated):
                    action = self.rl_model.predict(obs)[0]
                    obs, reward, done, truncated, info = self.env.step(action)
                    if info['robustness'] is not None and info['robustness'] > 0:
                        num_robustness_step += 1
                        weight = self.robustness_score_weight if self.args.model_with_robustness_reward else 0.0
                        avg_robustness += info['robustness'] * weight
                    # self.env.render()
                success_score = 1.0 if done else 0.0
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

    def save_to_csv(self, row_data, mode='a'):
        """
        Save the results of num_iter, best_design, best_score, and evaluation scores to a CSV file.

        Args:
            row_data (list): Data row to be saved.
            mode (str): Mode to open the file. Default is 'a' (append).
        """
        filename = self.args.save_filename
        if not os.path.exists(filename):
            if self.args.env == "catch":
                headers = ["num_iter", "num_episodes_so_far", "best_design_0", "best_design_1", "best_design_2", 
                        "best_design_3", "best_design_4", "best_score_estimated", "score_true", 
                        "success_score_true", "robustness_score_true"]
            elif self.args.env == "vpush":
                headers = ["num_iter", "num_episodes_so_far", "best_design_0", "best_score_estimated", 
                        "score_true", "success_score_true", "robustness_score_true"]
            elif self.args.env == "panda":
                headers = ["num_iter", "num_episodes_so_far", "best_design_0", "best_design_1", "best_design_2", 
                        "best_design_3", "best_score_estimated", "score_true", "success_score_true", "robustness_score_true"]
            elif self.args.env == "dlr":
                headers = ["num_iter", "num_episodes_so_far", "best_design_0", "best_design_1", "best_score_estimated", 
                        "score_true", "success_score_true", "robustness_score_true"]
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        
        with open(filename, mode, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

    def evaluate_optimal_design(self, design):
        """
        Evaluate the optimal design.

        Args:
            design (list): Optimal design parameters.
        Returns:
            tuple: Average score, success score, robustness score.
        """
        avg_score, avg_success_score, avg_robustness_score = 0, 0, 0
        obs, _ = self.env.reset(seed=self.args.random_seed)

        for episode in range(self.args.num_episodes_eval_best):
            if self.env_type == "dlr":
                self.env.num_discrete_tasks = self.num_outputs
            task = np.random.choice(range(self.num_outputs))
            obs, _ = self.env.reset_task_and_design(task, design, seed=self.args.random_seed)
            done, truncated = False, False
            avg_robustness = 0
            num_robustness_step = 0

            while not (done or truncated):
                action = self.rl_model.predict(obs)[0]
                obs, reward, done, truncated, info = self.env.step(action)
                if info.get('robustness') is not None and info['robustness'] > 0:
                    num_robustness_step += 1
                    avg_robustness += info['robustness'] * self.robustness_score_weight

            print("Done!" if done else "Truncated.")
            success_score = 1.0 if done else 0.0
            robustness_score = avg_robustness / num_robustness_step if num_robustness_step > 0 else 0
            score = success_score + robustness_score
            avg_score += score
            avg_success_score += success_score
            avg_robustness_score += robustness_score

        avg_score /= self.args.num_episodes_eval_best
        avg_success_score /= self.args.num_episodes_eval_best
        avg_robustness_score /= self.args.num_episodes_eval_best
        print(f"Design: {design}, Score: {avg_score}, Success: {avg_success_score}, Robustness: {avg_robustness_score}")

        return avg_score, avg_success_score, avg_robustness_score
    
    def run(self):
        """Run the Bayesian Optimization pipeline."""
        # csv_filename = f"results/csv/{self.env_type}_bo_results.csv"
        for i in range(1, self.max_iter + 1):
            print("-------------- Iteration: --------------", i)
            self.bo.run_optimization(1, eps=-1)
            print('next_locations: ', self.bo.suggest_next_locations())
            best_design, best_score, _, _, _ = self.find_optimal_design()
            print(f"Optimal Design: {best_design}, Score: {best_score}")
            evaluated_score, success_rate, robustness_score = self.evaluate_optimal_design(best_design)
            num_episodes_so_far = self.num_episodes * i * self.num_outputs
            row_data = [i, num_episodes_so_far] + best_design.tolist() + [best_score, evaluated_score, success_rate, robustness_score]
            self.save_to_csv(row_data)

if __name__ == "__main__":
    pipeline = BayesianOptimization(initial_iter=1,)
    pipeline.run()
    pipeline.env.close()
