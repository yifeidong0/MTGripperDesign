import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random

class GeneticAlgorithmPipeline:
    def __init__(self, env_type="push", population_size=20, generations=50, mutation_rate=0.1, num_episodes=1, gui=False):
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

        if self.env_type == "push":
            self.bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.1, 1.2)},
                           {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2
        elif self.env_type == "vpush":
            self.bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, np.pi)},
                           {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2
        elif self.env_type == "ucatch":
            self.bounds = [{'name': 'd0', 'type': 'continuous', 'domain': (5, 10)},
                           {'name': 'd1', 'type': 'continuous', 'domain': (5, 10)},
                           {'name': 'd2', 'type': 'continuous', 'domain': (5, 10)},
                           {'name': 'alpha0', 'type': 'continuous', 'domain': (np.pi / 2, np.pi)},
                           {'name': 'alpha1', 'type': 'continuous', 'domain': (np.pi / 2, np.pi)},
                           {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2

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
        
        if self.env_type == "push":
            from sim.push_sim import ForwardSimulationPlanePush as Simulation
            task = 'ball' if t == 0 else 'box'
            sim = Simulation(task, x, self.gui)
            score = sim.run()
        elif self.env_type == "vpush":
            from sim.vpush_sim import VPushSimulation as Simulation
            task = 'circle' if t == 0 else 'polygon'
            sim = Simulation(task, x, self.gui)
            score = sim.run(self.num_episodes)
        elif self.env_type == "ucatch":
            from sim.ucatch_sim import UCatchSimulation as Simulation
            task = 'circle' if t == 0 else 'polygon'
            sim = Simulation(task, x, self.gui)
            score = sim.run(self.num_episodes)

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

    def find_optimal_design(self):
        """
        Run the genetic algorithm to find the optimal design.

        Returns:
            tuple: The best design and its score.
        """
        best_design = None
        best_score = -np.inf

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

        return best_design, best_score

    def run(self):
        """
        Run the genetic algorithm pipeline to find and print the optimal design.
        """
        best_design, best_score = self.find_optimal_design()
        print(f"Optimal Design: {best_design}, Score: {best_score}")

if __name__ == "__main__":
    pipeline = GeneticAlgorithmPipeline(env_type="ucatch", population_size=20, generations=50, mutation_rate=0.1, num_episodes=1, gui=False) # ucatch, vpush
    pipeline.run()
