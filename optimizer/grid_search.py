import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class GridSearchPipeline:
    def __init__(self, env_type="push", resolution=10, num_episodes=1, gui=False):
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

        self.resolution = resolution
        self.num_episodes = num_episodes
        self.grid_points = self.generate_grid_points()

    def generate_grid_points(self):
        grids = [np.linspace(bound['domain'][0], bound['domain'][1], self.resolution) 
                 for bound in self.bounds if bound['type'] == 'continuous']
        return np.array(np.meshgrid(*grids)).T.reshape(-1, len(self.bounds)-1)

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
        
        for sample in self.grid_points:
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
    pipeline = GridSearchPipeline(env_type="ucatch", resolution=2, num_episodes=1, gui=False)
    pipeline.run()
