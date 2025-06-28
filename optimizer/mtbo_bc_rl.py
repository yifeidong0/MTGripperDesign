import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import GPy
import GPyOpt
import os
import sys
from utils.plot_mtbo import *
from utils.plot_mtbo_multi import *
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import envs
import time
import csv
import wandb
from experiments.args_utils import get_args_bo
from GPyOpt.acquisitions.base import AcquisitionBase
from experiments.panda_IL import PandaEnvWrapper, DeepPolicy
from policy.ppo_reg import PPOReg
import sys
sys.path.append('/home/yif/Documents/KTH/git/MTGripperDesign/experiments')  # Adjust if panda_IL is in a subdirectory
import experiments.panda_IL  # Make sure this succeeds
from stable_baselines3.common.evaluation import evaluate_policy

class MyAcquisition(AcquisitionBase):
    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=1, num_outputs=2):
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
        self.num_outputs = num_outputs
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
        f_acqu = m + self.exploration_weight * s

        # Adjust acquisition function for an arbitrary number of tasks
        for t in range(self.num_outputs):
            try:
                scale_factor = eval_scale(task_no=t)
                f_acqu[x[:, -1] == t] = f_acqu[x[:, -1] == t] / scale_factor
            except:
                pass
            
        # Penalize high variance (potential outliers)
        penalized_acq = f_acqu - (s / 2.0)
        return penalized_acq


class BayesianOptimizationMultiTask:
    def __init__(self, initial_iter=1):
        """
        Initialize the Bayesian Optimization Pipeline with the given parameters.

        Args:
            env_type (str): The type of environment ('vpush', or 'catch').
            initial_iter (int): The number of initial iterations to run.
            max_iter (int): The maximum number of iterations to run.
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
            self.bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (np.pi/12, np.pi*11/12)},
                        {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2
            self.robustness_score_weight = 1.0
        elif self.env_type == "catch":
            self.bounds = [{'name': 'd0', 'type': 'continuous', 'domain': (5, 10)}, # design space bounds
                        {'name': 'd1', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'd2', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'alpha0', 'type': 'continuous', 'domain': (np.pi/2, np.pi)},
                        {'name': 'alpha1', 'type': 'continuous', 'domain': (np.pi/2, np.pi)},
                        {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2
            self.robustness_score_weight = 0.1
        elif self.env_type == "panda":
            self.num_outputs = 5
            self.bounds = [{'name': 'v_angle', 'type': 'continuous', 'domain': (np.pi/6, np.pi*5/6)}, # design space bounds, make sure it aligns with reset in panda_push_env
                        {'name': 'finger_length', 'type': 'continuous', 'domain': (0.05, 0.12)}, 
                        {'name': 'finger_angle', 'type': 'continuous', 'domain': (-np.pi/3, np.pi/3)},
                        {'name': 'distal_phalanx_length', 'type': 'continuous', 'domain': (0.00, 0.08)},
                        {'name': 'task', 'type': 'discrete', 'domain': tuple(range(self.num_outputs))}]
            self.robustness_score_weight = 0.15
        elif self.env_type == "dlr":
            self.num_outputs = 11
            self.bounds = [{'name': 'l', 'type': 'discrete', 'domain': list(np.arange(30, 65, 5))},
                           {'name': 'c', 'type': 'discrete', 'domain': list(np.arange(2, 10, 2))},
                           {'name': 'task', 'type': 'discrete', 'domain': list(range(self.num_outputs))}]
            self.robustness_score_weight = 1.0 / 2000.0

        # Create the environment and RL model
        env_ids = {'vpush':'VPushPbSimulationEnv-v0', 
                    'catch':'UCatchSimulationEnv-v0',
                    'dlr':'DLRSimulationEnv-v0',
                    'panda':'PandaUPushEnv-v0'}
        self.env_id = env_ids[self.args.env]
        run_id = wandb.util.generate_id()

        env_kwargs = {
                        # 'obs_type': self.args.obs_type, 
                        'render_mode': self.args.render_mode,
                        'perturb': self.args.perturb,
                        # 'run_id': run_id,
                        'max_episode_steps': 1000,
                        }
        self.env = gym.make(self.env_id, **env_kwargs)
        self.env = PandaEnvWrapper(self.env)
        self.rl_model = PPOReg.load(self.model_path, env=self.env, policy=DeepPolicy)
        # self.env = gym.make(self.env_id, **env_kwargs)
        # self.rl_model = PPO.load(self.model_path)

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
        # length scale helps to smooth the function and avoid overfitting
        return GPy.kern.Matern52(input_dim, ARD=1, lengthscale=10.0) + GPy.kern.White(input_dim, variance=1.0) 

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

    def evaluate_policy_fixed_task_and_design(self, policy, t, x, n_eval_episodes, deterministic=True):
        """
        Evaluate the policy for a fixed task and design across multiple episodes.

        Args:
            policy: The policy object with a `.predict(obs)` method.
            t: Task index (int).
            x: Design vector (list or np.ndarray).
            n_eval_episodes: Number of episodes to evaluate.
            deterministic: Whether to use deterministic actions.

        Returns:
            mean_reward: Average reward over episodes.
            std_reward: Standard deviation of reward.
        """
        episode_rewards = []
        dones = []

        for episode in range(n_eval_episodes):
            obs, _ = self.env.reset()
            obs, _ = self.env.reset_task_and_design(t, x, seed=self.args.random_seed)

            # Flatten dict observation if needed
            if isinstance(obs, dict):
                obs = np.concatenate([
                    obs["observation"],
                    obs["achieved_goal"],
                    obs["desired_goal"]
                ], axis=-1)

            done, truncated = False, False
            episode_reward = 0.0

            while not (done or truncated):
                action = policy.predict(obs, deterministic=deterministic)[0]
                obs, reward, done, truncated, info = self.env.step(action)

                if isinstance(obs, dict):
                    obs = np.concatenate([
                        obs["observation"],
                        obs["achieved_goal"],
                        obs["desired_goal"]
                    ], axis=-1)

                episode_reward += reward

            dones.append(1) if done else dones.append(0)
            episode_rewards.append(episode_reward)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        success_rate = np.mean(dones)

        return mean_reward, std_reward, success_rate

    def mt_objective(self, xt):
        """
        Evaluate the multi-task objective function.

        Args:
            xt (list): The input design and task.
            num_episodes (int): The number of episodes to run the simulation.

        Returns:
            float: The score for the given design and task.
        """
        print("Evaluating the BO candidate...")
        assert len(xt) == 1
        xt = xt[0]
        # assert xt[-1] == 0.0 or xt[-1] == 1.0
        x, t = xt[:-1], int(xt[-1])
        self.task_counter[t] = self.task_counter.get(t, 0) + 1

        self.rl_model.policy.eval()  # Set the policy to evaluation mode
        # mean_reward, std_reward = evaluate_policy(self.rl_model.policy, self.env, 
        #                                           n_eval_episodes=1, deterministic=True)
        mean_reward, std_reward, success_rate = self.evaluate_policy_fixed_task_and_design(
            self.rl_model.policy,
            t=t,
            x=x,
            n_eval_episodes=self.num_episodes,
            deterministic=True
        )
        score = mean_reward + 100/std_reward + 50*success_rate
        print(f"Task: {t}, Design: {x}, Score: {score}, Mean Reward: {mean_reward}, Std Reward: {std_reward}, Success Rate: {success_rate}")

        return score

    def cost_mt(self, xt):
        """
        Calculate the cost and gradients for the given input.

        Args:
            xt (np.array): The input design and task.

        Returns:
            tuple: The costs and gradients for the input.
        """
        costs = np.zeros((xt.shape[0], 1))
        gradients = np.zeros((xt.shape[0], len(self.bounds)))

        # Assign costs and gradients for each task dynamically
        for t in range(self.num_outputs):
            costs[xt[:, -1] == t] = 1.0  # You can modify the cost function for each task if necessary
            gradients[xt[:, -1] == t] = [0.0] * len(self.bounds)  # Modify if task-specific gradients are needed

        return costs, gradients

    def setup_bo(self):
        """
        Set up the Bayesian Optimization pipeline.
        """
        self.objective = GPyOpt.core.task.SingleObjective(self.mt_objective)
        self.model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False, kernel=self.kernel, noise_var=1e-1) # noise helps to avoid overfitting
        self.space = GPyOpt.Design_space(space=self.bounds)
        self.acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(self.space)
        self.initial_design = GPyOpt.experiment_design.initial_design('random', self.space, self.initial_iter)

        self.acquisition = MyAcquisition(self.model, self.space, optimizer=self.acquisition_optimizer, cost_withGradients=self.cost_mt, num_outputs=self.num_outputs)
        self.evaluator = GPyOpt.core.evaluators.Sequential(self.acquisition)
        self.bo = GPyOpt.methods.ModularBayesianOptimization(self.model, self.space, self.objective, self.acquisition, self.evaluator, self.initial_design, normalize_Y=False)

    def find_optimal_design(self, resolution=10, mean_max=0):
        """
        Find the optimal design by evaluating a grid of points.

        Args:
            resolution (int): The resolution of the grid.

        Returns:
            tuple: The best design, its score, the grid points, the means, and the variances.
        """
        print("Finding optimal design...")
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
        if mean_max:
            avg_scores = np.mean(means, axis=0).flatten()
        else: # min-max
            avg_scores = np.min(means, axis=0).flatten()

        # Find the design with the highest average score
        best_idx = np.argmax(avg_scores)
        best_design = grid_points[best_idx]
        best_score = avg_scores[best_idx]

        # make best_design to be the closest grid point
        if self.args.env == "dlr":
            best_design[0] = np.round(best_design[0] / 5) * 5
            best_design[1] = np.round(best_design[1] / 2) * 2
        
        return best_design, best_score, grid_points, means, vars

    def evaluate_optimal_design(self, design):
        """
        Evaluate the given design across multiple tasks using a fixed task and design per run.

        Returns:
            tuple: mean_reward, std_reward, success_rate
        """
        total_rewards = []
        total_success = []

        print("Evaluating optimal design...")
        print("@@@@@@@@@@@@@@@self.args.num_episodes_eval_best", self.args.num_episodes_eval_best)
        for i in range(self.args.num_episodes_eval_best):
            if self.env_type == "dlr":
                self.env.num_discrete_tasks = self.num_outputs
            task = np.random.choice(range(self.num_outputs))

            self.rl_model.policy.eval()
            mean_reward, std_reward, success_rate = self.evaluate_policy_fixed_task_and_design(
                policy=self.rl_model.policy,
                t=task,
                x=design,
                n_eval_episodes=1,
                deterministic=True
            )

            total_rewards.append(mean_reward)
            total_success.append(success_rate)

        final_mean_reward = np.mean(total_rewards)
        final_success_rate = np.mean(total_success)

        print(f"Design: {design}, Mean Reward: {final_mean_reward}, Success Rate: {final_success_rate}")
        return final_mean_reward, final_success_rate

    def run(self):
        """
        Run the Bayesian Optimization pipeline.
        """
        for i in range(1, self.max_iter + 1):
            print("-------------- Iteration: --------------", i)
            self.bo.run_optimization(1, eps=-1)
            print('next_locations: ', self.bo.suggest_next_locations())

            # Find the optimal design after the optimization loop
            best_design, best_score, grid_points, means, vars = self.find_optimal_design()
            print(f"Optimal Design: {best_design}, Score: {best_score}")
            
            # Evaluate the best design using the optimal policy
            opt_mean_reward, opt_success_rate = self.evaluate_optimal_design(best_design)
            
            num_episodes_so_far = self.num_episodes * (i + self.initial_iter)
            row_data = [i, num_episodes_so_far] + best_design.tolist() + [best_score, opt_mean_reward, opt_success_rate]

            # Save the result to CSV immediately
            self.save_to_csv(row_data)

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
                        "best_design_3", "best_score_estimated", "opt_mean_reward", "opt_success_rate"]
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

if __name__ == "__main__":
    pipeline = BayesianOptimizationMultiTask(initial_iter=1) # initial_iter has to be more than 0 
    pipeline.run()
    pipeline.env.close()
