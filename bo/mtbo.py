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

class BayesianOptimizationPipeline:
    def __init__(self, env_type="push", initial_iter=3, max_iter=10, gui=False):
        self.env_type = env_type
        self.gui = gui

        if self.env_type == "push":
            self.x_scale = np.arange(0.1, 1.2, 0.04)
            self.bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.1, 1.2)},
                        {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}]
            self.num_outputs = 2 # no. of tasks
        elif self.env_type == "vpush":
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

        self.initial_iter = initial_iter
        self.max_iter = max_iter
        self.kernel = self.get_kernel_mt(input_dim=len(self.bounds)-1, num_outputs=self.num_outputs)
        self.task_counter = {}
        self.setup_bo()

    def get_kernel_basic(self, input_dim=1):
        return GPy.kern.Matern52(input_dim, ARD=1) + GPy.kern.Bias(input_dim)

    def get_kernel_mt(self, input_dim=1, num_outputs=2):
        return GPy.util.multioutput.ICM(input_dim, num_outputs, self.get_kernel_basic(input_dim), W_rank=1, name='ICM')

    def mt_objective(self, xt, num_episodes=1):
        assert len(xt) == 1
        xt = xt[0]
        assert xt[-1] == 0.0 or xt[-1] == 1.0
        x, t = xt[:-1], int(xt[-1])
        self.task_counter[t] = self.task_counter.get(t, 0) + 1
        
        if self.env_type == "push":
            from sim.push_sim import ForwardSimulationPlanePush as Simulation
            task = 'ball' if t == 0 else 'box'
            sim = Simulation(task, x, self.gui)
            score = sim.run()
        elif self.env_type == "vpush":
            from sim.vpush_sim import VPushSimulation as Simulation
            task = 'circle' if t == 0 else 'polygon'
            sim = Simulation(task, x, self.gui)
            score = sim.run(num_episodes)
        elif self.env_type == "ucatch":
            from sim.ucatch_sim import UCatchSimulation as Simulation
            task = 'circle' if t == 0 else 'polygon'
            sim = Simulation(task, x, self.gui)
            score = sim.run(num_episodes)

        return score

    def cost_mt(self, xt):
        C0, C1 = 1.0, 1.0
        t2c = {0: C0, 1: C1}
        t2g = {0: [0.0, (C1 - C0)], 1: [0.0, (C1 - C0)]}
        costs = np.array([[t2c[int(t)]] for t in xt[:, -1]])
        gradients = np.array([t2g[int(t)] for t in xt[:, -1]])
        return costs, gradients

    def setup_bo(self):
        self.objective = GPyOpt.core.task.SingleObjective(self.mt_objective)
        self.model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False, kernel=self.kernel, noise_var=None)
        self.space = GPyOpt.Design_space(space=self.bounds)
        self.acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(self.space)
        self.initial_design = GPyOpt.experiment_design.initial_design('random', self.space, self.initial_iter)

        from GPyOpt.acquisitions.base import AcquisitionBase
        class MyAcquisition(AcquisitionBase):
            analytical_gradient_prediction = False

            def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2):
                self.optimizer = optimizer
                super(MyAcquisition, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
                self.exploration_weight = exploration_weight

            def _compute_acq(self, x):
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
        # Extract the bounds for the design space
        bounds = self.space.get_bounds()
        
        # Generate a grid of points across the design space
        grids = [np.linspace(bound[0], bound[1], resolution) for bound in bounds[:-1]]
        grid_points = np.array(np.meshgrid(*grids)).T.reshape(-1, len(bounds) - 1) # 100000*5
        
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
        for i in range(1, self.max_iter + 1):
            print("-------------- Iteration: --------------", i)
            self.bo.run_optimization(1)
            print('next_locations: ', self.bo.suggest_next_locations())
            if self.env_type in ["push", "vpush"]:
                plot_bo(self.bo, self.x_scale, i)
    
        # Find the optimal design after the optimization loop
        best_design, best_score, grid_points, means, vars = self.find_optimal_design()
        print(f"Optimal Design: {best_design}, Score: {best_score}")

        # Plot the results
        if self.env_type in ["ucatch"]:
            plot_marginalized_results(grid_points, means, vars, )


if __name__ == "__main__":
    pipeline = BayesianOptimizationPipeline(env_type="ucatch", initial_iter=5, max_iter=25, gui=0) # vpush, push, ucatch
    pipeline.run()
