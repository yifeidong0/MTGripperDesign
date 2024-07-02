import numpy as np
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RandomSearchPipeline:
    def __init__(self, env_type="push", num_samples=100, num_episodes=1, gui=False):
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

        self.num_samples = num_samples
        self.num_episodes = num_episodes
        self.samples = self.generate_random_samples()

    def generate_random_samples(self):
        samples = []
        for _ in range(self.num_samples):
            sample = []
            for bound in self.bounds:
                if bound['type'] == 'continuous':
                    sample.append(random.uniform(*bound['domain']))
            samples.append(sample)
        return np.array(samples) # shape: (num_samples, dim_design)

    def mt_objective(self, xt):
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

    def find_optimal_design(self):
        best_design = None
        best_score = -np.inf
        
        for sample in self.samples:
            scores = []
            for task in range(self.num_outputs):
                xt = np.append(sample, task)
                score = self.mt_objective(xt)
                scores.append(score)
            avg_score = np.mean(scores)
            # print(f"Design: {sample}, Score: {avg_score}")
            if avg_score > best_score:
                best_score = avg_score
                best_design = sample
        
        return best_design, best_score

    def run(self):
        best_design, best_score = self.find_optimal_design()
        print(f"Optimal Design: {best_design}, Score: {best_score}")

if __name__ == "__main__":
    pipeline = RandomSearchPipeline(env_type="ucatch", num_samples=5, num_episodes=1, gui=False)
    pipeline.run()

