import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import envs
import gymnasium as gym
import time
import wandb
from experiments.args_utils import get_args_ga
import csv

class GeneticAlgorithmPipeline:
    def __init__(self, ):
        """
        Initialize the Genetic Algorithm Pipeline with the given parameters.

        Args:
            env_type (str): The type of environment ('vpush', or 'catch').
            population_size (int): The size of the population for the genetic algorithm.
            generations (int): The number of generations to run the genetic algorithm.
            mutation_rate (float): The mutation rate for the genetic algorithm.
            num_episodes (int): The number of episodes to run for each evaluation.
        """
        self.args = get_args_ga()
        self.env_type = self.args.env
        self.model_path = self.args.model_path

        if self.env_type == "vpush":
            self.bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, np.pi)},
                        {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2
            self.robustness_score_weight = 1.0
        elif self.env_type == "catch":
            self.bounds = [{'name': 'd0', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'd1', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'd2', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'alpha0', 'type': 'continuous', 'domain': (np.pi / 2, np.pi)},
                        {'name': 'alpha1', 'type': 'continuous', 'domain': (np.pi / 2, np.pi)},
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
            self.robustness_score_weight = 0.3
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
        self.env_id = env_ids[self.env_type]
        run_id = wandb.util.generate_id()

        env_kwargs = {'obs_type': self.args.obs_type, 
                        'render_mode': self.args.render_mode,
                        'perturb': self.args.perturb,
                        'run_id': run_id,
                        }
        self.env = gym.make(self.env_id, **env_kwargs)
        self.model = PPO.load(self.model_path)

        self.population_size = self.args.population_size
        self.num_generations = self.args.num_generations
        self.mutation_rate = self.args.mutation_rate
        self.num_episodes = self.args.num_episodes_eval
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        """
        Generate the initial population for the genetic algorithm.

        Returns:
            np.array: The initial population of designs. Each element is a list of design parameters, excluding task space.
        """
        population = []
        for _ in range(self.population_size):
            individual = []
            for bound in self.bounds[:-1]:  # Do not include the task dimension
                if bound['type'] == 'continuous':
                    individual.append(random.uniform(*bound['domain']))
                elif bound['type'] == 'discrete':
                    individual.append(random.choice(bound['domain']))
            population.append(individual)
        return np.array(population)

    def mt_objective(self, xt):
        """
        Evaluate the objective function for a given design and task.

        Args:
            xt (list): The design and task to evaluate.

        Returns:
            float: The score for the given design and task.
        """
        x, t = xt[:-1], int(xt[-1])
        avg_score = 0
        obs, _ = self.env.reset(seed=0)
        for episode in range(self.num_episodes):
            obs, _ = self.env.reset_task_and_design(t, x, seed=0)
            # print(f"Episode {episode + 1} begins")
            done, truncated = False, False
            avg_robustness = 0
            num_robustness_step = 0
            while not (done or truncated):
                action = self.model.predict(obs)[0]
                obs, reward, done, truncated, info = self.env.step(action)
                if info['robustness'] is not None and info['robustness'] > 0:
                    num_robustness_step += 1
                    weight = self.robustness_score_weight if self.args.model_with_robustness_reward else 0.0
                    avg_robustness += info['robustness'] * weight
                # self.env.render()
            success_score = 1 if done else 0
            robustness_score = avg_robustness / num_robustness_step if num_robustness_step > 0 else 0
            # print(f"Success: {success_score}, Robustness: {robustness_score}")
            score = success_score + robustness_score
            avg_score += score
            print("Done!" if done else "Truncated.")
            # print(f"Episode {episode + 1} finished")
        score = avg_score / self.num_episodes
        # time.sleep(1)
        # self.env.close()

        return score

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual design.

        Args:
            individual (list): The design to evaluate.

        Returns:
            float: The average score across tasks for the given design.
        """
        scores = []
        for task in range(self.num_outputs):
            xt = np.append(individual, task)
            score = self.mt_objective(xt)
            scores.append(score)
        return np.mean(scores)

    def select_parents(self, fitness):
        """
        Select parents for the next generation based on fitness.

        Args:
            fitness (np.array): The fitness of each individual in the population.

        Returns:
            np.array: The selected parents.
        """
        total_fitness = np.sum(fitness)
        probabilities = fitness / total_fitness
        parents_indices = np.random.choice(range(self.population_size), size=self.population_size, p=probabilities)
        return self.population[parents_indices]

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to generate two children.

        Args:
            parent1 (list): The first parent.
            parent2 (list): The second parent.

        Returns:
            tuple: The two children generated from the crossover.
        """
        crossover_point = random.randint(1, len(parent1) - 1) if len(parent1)>1 else 0
        child1 = np.hstack((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.hstack((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(self, individual):
        """
        Mutate an individual design.

        Args:
            individual (list): The design to mutate.

        Returns:
            list: The mutated design.
        """
        for i in range(len(individual)):  # Do not mutate the task dimension
            if random.random() < self.mutation_rate:
                bound = self.bounds[i]['domain']
                individual[i] = random.uniform(*bound)
        return individual

    def save_to_csv(self, filename, data):
        """
        Save results to a CSV file after each iteration, appending to the file.
        Args:
            filename (str): Path to the CSV file.
            data (list): Row data to be written (single row).
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Check if the file already exists
        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write the header only if the file does not exist
            if not file_exists:
                if self.env_type in ["vpush"]:
                    writer.writerow(["num_iter", "num_episodes_so_far", "best_design_0", "best_score_estimated", "evaluated_score", "success_rate", "robustness_score"])
                elif self.env_type in ["catch"]:
                    writer.writerow(["num_iter", "num_episodes_so_far", "best_design_0", "best_design_1", "best_design_2", "best_design_3", "best_design_4", "best_score_estimated", "evaluated_score", "success_rate", "robustness_score"])
                elif self.env_type in ["panda"]:
                    writer.writerow(["num_iter", "num_episodes_so_far", "best_design_0", "best_design_1", "best_design_2", "best_design_3", "best_score_estimated", "evaluated_score", "success_rate", "robustness_score"])
                elif self.env_type in ["dlr"]:
                    writer.writerow(["num_iter", "num_episodes_so_far", "best_design_0", "best_design_1", "best_score_estimated", "evaluated_score", "success_rate", "robustness_score"])

            # Write the row data
            writer.writerow(data)
            
    def evaluate_optimal_design(self, design):
        """
        Evaluate the optimal design found by the GA algorithm.

        Args:
            design (list): The design parameters to evaluate.

        Returns:
            tuple: The average score, success score, and robustness score over the evaluation episodes.
        """
        avg_score = 0
        avg_success_score = 0
        avg_robustness_score = 0
        obs, _ = self.env.reset(seed=0)
        print('Evaluating optimal design...')

        for episode in range(self.args.num_episodes_eval_best):
            task = np.random.choice(list(range(self.num_outputs)))  # Randomly select a task
            obs, _ = self.env.reset_task_and_design(task, design, seed=0)
            done, truncated = False, False
            avg_robustness = 0
            num_robustness_step = 0

            while not (done or truncated):
                action = self.model.predict(obs)[0]
                obs, reward, done, truncated, info = self.env.step(action)
                if info.get('robustness') is not None and info['robustness'] > 0:
                    num_robustness_step += 1
                    avg_robustness += info['robustness'] * self.robustness_score_weight

            success_score = 1.0 if done else 0.0
            robustness_score = avg_robustness / num_robustness_step if num_robustness_step > 0 else 0
            score = success_score + robustness_score
            avg_score += score
            avg_success_score += success_score
            avg_robustness_score += robustness_score

        avg_score /= self.args.num_episodes_eval_best
        avg_success_score /= self.args.num_episodes_eval_best
        avg_robustness_score /= self.args.num_episodes_eval_best
        print(f"Evaluated Design: {design}, Score: {avg_score}, Success: {avg_success_score}, Robustness: {avg_robustness_score}")

        return avg_score, avg_success_score, avg_robustness_score
    
    def find_optimal_design(self):
        """
        Run the genetic algorithm to find the optimal design.

        Returns:
            tuple: The best design and its score.
        """
        best_design = None
        best_score = -np.inf

        # Create a unique file name
        csv_filename = self.args.save_filename

        for generation in range(self.num_generations):
            fitness = np.array([self.evaluate_fitness(individual) for individual in self.population])
            best_gen_design = self.population[np.argmax(fitness)]
            best_gen_score = np.max(fitness)

            if best_gen_score > best_score:
                best_score = best_gen_score
                best_design = best_gen_design

            print(f"Generation {generation}, Best Score: {best_gen_score}, Best Design: {best_gen_design}")

            parents = self.select_parents(fitness)
            next_population = []

            for i in range(0, self.population_size, 2):
                parent1, parent2 = parents[i], parents[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                next_population.append(self.mutate(child1))
                next_population.append(self.mutate(child2))

            self.population = np.array(next_population)

            # Evaluate the optimal design after each generation
            evaluated_score, success_rate, robustness_score = self.evaluate_optimal_design(best_design)

            # Write results to the CSV after each iteration
            num_episodes_so_far = self.population_size * self.num_episodes * self.num_outputs * (generation + 1)
            row_data = [generation, num_episodes_so_far] + best_design.tolist() + [best_gen_score, evaluated_score, success_rate, robustness_score]
            self.save_to_csv(csv_filename, row_data)

        return best_design, best_score

    def run(self):
        """
        Run the genetic algorithm pipeline to find and print the optimal design.
        """
        best_design, best_score = self.find_optimal_design()
        print(f"Optimal Design: {best_design}, Score: {best_score}")


if __name__ == "__main__":
    num_run = 1
    for r in range(num_run):
        pipeline = GeneticAlgorithmPipeline()
        pipeline.run()
        pipeline.env.close()
