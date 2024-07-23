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

class GeneticAlgorithmPipeline:
    def __init__(self, env_type="push", population_size=20, generations=50, mutation_rate=0.1, num_episodes=1, gui=False, policy="heuristic"):
        """
        Initialize the Genetic Algorithm Pipeline with the given parameters.

        Args:
            env_type (str): The type of environment ('push', 'vpush', or 'ucatch').
            population_size (int): The size of the population for the genetic algorithm.
            generations (int): The number of generations to run the genetic algorithm.
            mutation_rate (float): The mutation rate for the genetic algorithm.
            num_episodes (int): The number of episodes to run for each evaluation.
            gui (bool): Whether to use GUI for the simulations.
        """
        self.env_type = env_type
        self.gui = gui
        self.policy = policy  # "heuristic" or "rl"

        if self.env_type == "push":
            self.bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.1, 1.2)},
                        {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2
        elif self.env_type == "vpush":
            self.bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, np.pi)},
                        {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2
            self.robustness_score_weight = 1.0
            self.env_id = 'VPushPbSimulationEnv-v0'
            self.model_path = "results/models/ppo_VPushPbSimulationEnv-v0_3000000_2024-07-22-16-17-10_with_robustness_reward.zip"
        elif self.env_type == "ucatch":
            self.bounds = [{'name': 'd0', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'd1', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'd2', 'type': 'continuous', 'domain': (5, 10)},
                        {'name': 'alpha0', 'type': 'continuous', 'domain': (np.pi / 2, np.pi)},
                        {'name': 'alpha1', 'type': 'continuous', 'domain': (np.pi / 2, np.pi)},
                        {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2
            self.robustness_score_weight = 0.1
            self.env_id = 'UCatchSimulationEnv-v0'
            self.model_path = "results/models/best_model_ucatch_w_robustness_reward.zip"

        if self.policy == "heuristic":
            if self.env_type == "vpush":
                from sim.vpush_pb_sim import VPushPbSimulation as Simulation
                self.sim = Simulation('circle', 1, self.gui)
        elif self.policy == "rl":
            self.env = gym.make(self.env_id, gui=self.gui, obs_type='pose')
            self.model = PPO.load(self.model_path)

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.num_episodes = num_episodes
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
            for bound in self.bounds:
                if bound['type'] == 'continuous':
                    individual.append(random.uniform(*bound['domain']))
                # elif bound['type'] == 'discrete':
                #     individual.append(random.choice(bound['domain']))
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
        
        if self.policy == "heuristic":
            if self.env_type == "push":
                from sim.push_sim import ForwardSimulationPlanePush as Simulation
                task = 'ball' if t == 0 else 'box'
                sim = Simulation(task, x, self.gui)
                score = sim.run()
            elif self.env_type == "vpush":
                # from sim.vpush_sim import VPushSimulation as Simulation
                task = 'circle' if int(t) == 0 else 'polygon'
                self.sim.reset_task_and_design(task, x[0])
                score = self.sim.run(self.num_episodes)
            elif self.env_type == "ucatch":
                from sim.ucatch_sim import UCatchSimulation as Simulation
                task = 'circle' if t == 0 else 'polygon'
                sim = Simulation(task, x, self.gui)
                score = sim.run(self.num_episodes)
        elif self.policy == "rl":
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
                        avg_robustness += info['robustness'] * self.robustness_score_weight
                    self.env.render()
                success_score = 0.5 if done else 0.1
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

    def find_optimal_design(self):
        """
        Run the genetic algorithm to find the optimal design.

        Returns:
            tuple: The best design and its score.
        """
        best_design = None
        best_score = -np.inf

        # Create a file with unique name
        k = 0
        while True:     
            csv_filename = f"results/csv/{self.env_type}_ga_results_{k}.csv"
            if os.path.exists(csv_filename):
                k += 1
            else:
                break

        csv_buffer = []
        for generation in range(self.generations):
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

            # Write to CSV file
            num_episodes_so_far = self.population_size * self.num_episodes * self.num_outputs * (generation + 1)
            csv_buffer.append([i, num_episodes_so_far, best_design, best_score])

        # Save intermediate designs to a csv file
        self.save_to_csv(csv_filename, csv_buffer)

        return best_design, best_score

    def run(self):
        """
        Run the genetic algorithm pipeline to find and print the optimal design.
        """
        best_design, best_score = self.find_optimal_design()
        print(f"Optimal Design: {best_design}, Score: {best_score}")


if __name__ == "__main__":
    num_run = 10
    for r in range(num_run):
        pipeline = GeneticAlgorithmPipeline(env_type="vpush",  # ucatch, vpush
                                            population_size=8, # has to be even
                                            generations=20, 
                                            mutation_rate=0.1, 
                                            num_episodes=4, 
                                            gui=0,
                                            policy="rl")  # heuristic, rl
        pipeline.run()
        pipeline.env.close()
