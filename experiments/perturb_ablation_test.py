import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import csv
from stable_baselines3.common.env_checker import check_env
import envs

class RLAgentEvaluator:
    def __init__(self, model_path, env_id, perturb_sigma, perturb, num_episodes=3, random_seed=42, render_mode='rgb_array'):
        """
        Initialize the RL Agent evaluator.

        Args:
            model_path (str): Path to the saved RL model (zip file).
            env_id (str): Gym environment ID.
            perturb_sigma (float): Perturbation level.
            perturb (bool): Whether to apply perturbation.
            num_episodes (int): Number of episodes to run.
            random_seed (int): Random seed for environment initialization.
        """
        self.model_path = model_path
        self.env_id = env_id
        self.perturb_sigma = perturb_sigma
        self.perturb = perturb
        self.num_episodes = num_episodes
        self.random_seed = random_seed
        self.render_mode = render_mode
        self.env = self._create_env()
        self.model = PPO.load(self.model_path)

        if self.env_id == "catch":
            self.robustness_score_weight = 0.1
        elif self.env_id == "vpush":
            self.robustness_score_weight = 1.0
        elif self.env_id == "panda":
            self.robustness_score_weight = 0.3
        elif self.env_id == "dlr":
            self.robustness_score_weight = 1/2000

    def _create_env(self):
        """Create a gym environment with perturbation."""
        env_kwargs = {
            'perturb': self.perturb,
            'perturb_sigma': self.perturb_sigma if self.perturb else 0.0,
            'using_robustness_reward': 1,
            'render_mode': self.render_mode,
        }
        env_ids = {'vpush':'VPushPbSimulationEnv-v0', 
                    'catch':'UCatchSimulationEnv-v0',
                    'dlr':'DLRSimulationEnv-v0',
                    'panda':'PandaUPushEnv-v0'}
        self.env = env_ids[self.env_id]

        env = gym.make(self.env, **env_kwargs)
        return env

    def run_episodes(self):
        """Run the RL agent for a specified number of episodes and record metrics."""
        scores = []
        successes = []
        robustness_scores = []
        
        for episode in range(self.num_episodes):
            obs, _ = self.env.reset(seed=self.random_seed)
            done, truncated = False, False
            total_score = 0
            success = 0
            robustness = 0
            step_count = 0

            while not (done or truncated):
                action, _ = self.model.predict(obs)
                obs, reward, done, truncated, info = self.env.step(action)

                if done:
                    success = 1
                if info.get('robustness') is not None and info['robustness'] > 0:
                    robustness_score = info['robustness'] * self.robustness_score_weight
                    robustness += robustness_score
                    step_count += 1
            
            robustness = robustness / step_count if step_count > 0 else 0
            total_score = success + robustness
            scores.append(total_score)
            successes.append(success)
            robustness_scores.append(robustness)

        return scores, successes, robustness_scores

    def save_results(self, scores, successes, robustness_scores, filepath):
        """Save the results of the evaluation to a CSV file, appending data if the file exists."""
        os.makedirs('results/paper/perturb', exist_ok=True)

        # Check if the file exists
        file_exists = os.path.isfile(filepath)
        
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write headers only if the file does not exist
            if not file_exists:
                writer.writerow(['Episode', 'Score', 'Success', 'Robustness'])
            
            # Write data for each episode
            for i, (score, success, robustness) in enumerate(zip(scores, successes, robustness_scores)):
                writer.writerow([i+1, score, success, robustness])
            
            # Write mean and std of the metrics
            # writer.writerow(['Mean', np.mean(scores), np.mean(successes), np.mean(robustness_scores)])
            # writer.writerow(['Std', np.std(scores), np.std(successes), np.std(robustness_scores)])

    def evaluate(self, filename):
        """Run the evaluation and save results."""
        scores, successes, robustness_scores = self.run_episodes()
        self.save_results(scores, successes, robustness_scores, filename)
        print(f"Evaluation completed for {self.env_id} with perturb_sigma={self.perturb_sigma} perturb={self.perturb}")


import argparse
import datetime
def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_args():
    parser = argparse.ArgumentParser(description="RL co-design project")
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility. Default is 42.')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes to run')
    parser.add_argument('--env_id', type=str, choices=['vpush', 'catch', 'dlr', 'panda',], default='panda', help='Environment ID for the simulation')
    parser.add_argument('--time_stamp', type=str, default=get_timestamp(), help='Current time of the script execution')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'], default='cpu', help='Computational device to use (auto, cuda, cpu)')
    parser.add_argument('--obs_type', type=str, choices=['pose', 'image'], default='pose', help='Type of observations for the training')
    parser.add_argument('--using_robustness_reward', type=str2bool, nargs='?', const=True, default=True, help='Enable or disable the robustness reward')
    parser.add_argument('--perturb', type=str2bool, nargs='?', const=False, default=1, help='Add random perturbations to the target object')
    parser.add_argument('--perturb_sigma', type=float, default=1.8, help='Random perturbations sigma')
    parser.add_argument('--render_mode', type=str, choices=['rgb_array', 'human'], default='rgb_array', help='Rendering mode for the simulation')
    parser.add_argument('--model_path', type=str, default='results/paper/catch/1/UCatchSimulationEnv-v0_5000000_2024-08-28_07-45-29_final.zip', help='model for evaluation')
    parser.add_argument('--output_file', type=str, default='results/paper/catch/1/mtbo_eval.csv', help='model for evaluation')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    evaluator = RLAgentEvaluator(
        model_path=args.model_path,
        env_id=args.env_id,
        perturb_sigma=args.perturb_sigma,
        perturb=bool(args.perturb),
        num_episodes=args.num_episodes,
        random_seed=args.random_seed,
        render_mode=args.render_mode,
    )
    evaluator.evaluate(args.output_file)
