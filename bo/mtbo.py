import numpy as np
import GPy
import GPyOpt
import matplotlib.pyplot as plt
import pylab as pl
from IPython import display
import time
from matplotlib import pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.push_sim import ForwardSimulationPlanePush

x_scale = np.arange(0.1, 1.2, 0.04)

def plot_err(x, y, yerr, color=None, alpha_fill=0.2, ax=None, label="", lw=1, ls="-"):
    y, yerr = y.reshape(-1), yerr.reshape(-1)
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label, lw=lw, ls=ls)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill, linewidth=0.0)

RENAME = {"Gaussian_noise.variance": "noise.var"}
def plot_params(bo):
    plt.xlim(-1,1); plt.ylim(-1,1); plt.axis('off');

    for i, (n, v) in enumerate(zip(bo.model.get_model_parameters_names(), bo.model.get_model_parameters()[0])):
        n = ".".join(n.split(".")[-2:])
        n = RENAME.get(n, n)
        plt.text(-0.9, 0.9-i*0.2, "%s:" % (n), ha="left", va="top", fontsize=13)
        plt.text(0.5, 0.9-i*0.2, "%.3f" % (v), ha="left", va="top", fontsize=13)

def plot_next_acquisitions(bo, DIM_NO=0, tasks_to_be_plotted=[0,1]):
    for xt in bo.suggest_next_locations():
        x, t = xt[:-1], int(xt[-1])
        assert t==0 or t==1 # only two tasks allowed for now
        if t not in tasks_to_be_plotted: continue
        plt.axvline(x=x[DIM_NO], color=("salmon" if t==0 else "limegreen"),
                    ls="--", lw=2, label="next acquisition")
        
def plot_acquisition(bo, DIM_NO=0, grid=0.001):

    bounds = bo.acquisition.space.get_bounds()

    x_grid = np.arange(bounds[DIM_NO][0], bounds[DIM_NO][1], grid)
    x_grid = x_grid.reshape(len(x_grid),1)

    xt = np.hstack([x_grid, np.zeros((x_grid.shape[0],1))]) # task 0
    acqu0 = -bo.acquisition.acquisition_function(xt)
    plt.plot(x_grid, acqu0, label="task 0", color="red", lw=2)

    xt = np.hstack([x_grid, np.ones((x_grid.shape[0],1))]) # task 1
    acqu1 = -acquisition.acquisition_function(xt)
    plt.plot(x_grid, acqu1, label="task 1", color="green", lw=2)

    plot_next_acquisitions(bo, DIM_NO, [0, 1])
    plt.xlim(-0.0,1.3); plt.legend(loc=2); plt.title("acquisition function"); plt.grid();

def plot_task(bo, TASK_NO): # TODO cover multi-dimensional inputs
    color = {0: "red", 1: "green"}[TASK_NO]
    # plt.plot(x_scale, f(x_scale), color=color, label="f0")

    task_column = TASK_NO*np.ones((x_scale.shape[0],1))
    xt = np.hstack([x_scale.reshape(-1,1), task_column])
    means0mt, stds0mt = bo.model.predict(xt)
    plot_err(x_scale, means0mt, stds0mt, lw=1, ls="-.", color="blue", label="BO fit%i" % TASK_NO, alpha_fill=0.05)

    X, Y = bo.get_evaluations()
    if len(X)>0 and X[-1,1]==TASK_NO:
        plt.scatter(X[-1,0],Y[-1,0], color=color, marker="o", s=60, label="prev acquisition")
    Y, X  = Y[X[:,1]==TASK_NO], X[X[:,1]==TASK_NO,0]
    plt.scatter(X,Y, color="k", marker="x", label="acquisition pts")

    plot_next_acquisitions(bo, 0, [TASK_NO])
    plt.xlim(-0.0,1.3); plt.ylim(0,1.2); plt.legend(loc=2); plt.title("task %i" % TASK_NO); plt.grid();

def plot_bo(bo):
    plt.figure(figsize=(22,4))
    plt.subplot(1,4,1)
    plot_task(bo, 0);
    plt.subplot(1,4,2)
    plot_task(bo, 1)
    plt.subplot(1,4,3)
    plot_acquisition(bo)
    plt.subplot(1,4,4)
    plot_params(bo)

# MT optimization objective (Run low-level planner and get behavior measures)
    
task_counter = {} # counter of how many times each task was evaluated
def mt_objective(xt):
    assert len(xt)==1 # one sample per batch
    xt = xt[0]
    assert xt[-1]==0.0 or xt[-1]==1.0 # for now only two tasks supported
    x, t = xt[:-1], int(xt[-1]) # extract coordinates and task number
    task_counter[t] = task_counter.get(t, 0) + 1 # update counts

    task = 'ball' if t==0 else 'box'
    sim = ForwardSimulationPlanePush(task_type=task, gripper_length=x, gui=1)

    return sim.run()

# Evaluation costs

def cost_const(xt):
    """constant cost baseline"""
    return np.ones(xt.shape[0])[:,None], np.zeros(xt.shape)

C0 = 1.0 # task 0 evaluation cost
C1 = 1.0 # task 1 evaluation cost

def cost_mt(xt):
    """different cost for different task no"""
    t2c = {0: C0, 1: C1} # task to cost
    t2g = {0: [0.0, (C1-C0)], 1: [0.0, (C1-C0)]} # task to gradient # TODO Correct???

    costs = np.array([[t2c[int(t)]] for t in xt[:,-1]])
    gradients = np.array([t2g[int(t)] for t in xt[:,-1]])
    return costs, gradients

bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.1, 1.2)},
          {'name': 'task', 'type': 'discrete', 'domain': (0, 1)}] #second variable denotes task number

def get_kernel_basic(input_dim=1):
    #return GPy.kern.RBF(1, active_dims=[0], lengthscale=1.0, ARD=1) + GPy.kern.Bias(input_dim)
    return GPy.kern.Matern52(input_dim, ARD=1) + GPy.kern.Bias(input_dim) #+ GPy.kern.White(input_dim)

def get_kernel_mt(input_dim=1, num_outputs=2):
#     kern_corg = GPy.kern.Coregionalize(1, output_dim=2, rank=1)
#     kern_mt = get_kernel_basic()**kern_corg
#     return kern_mt

    return GPy.util.multioutput.ICM(input_dim, num_outputs, get_kernel_basic(input_dim), W_rank=1, name='ICM')

initial_iter = 3
max_iter  = 10
kernel = get_kernel_mt()

cost = cost_mt # with this cost, acquisition function for task 1 is downweighted (flattened)

objective = GPyOpt.core.task.SingleObjective(mt_objective)

model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False, kernel=kernel, noise_var=None)
space = GPyOpt.Design_space(space=bounds)
acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
initial_design = GPyOpt.experiment_design.initial_design('random', space, initial_iter)

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
          X1 = X1[X1[:,1]==task_no]
          try:
            mean, var = self.model.model.predict(X1)
            std = np.sqrt(var)
            return (mean.max()-mean.min())+1e-6
          except:
            return 1.0

        m, s = self.model.predict(x)
        f_acqu = self.exploration_weight * s

        c0 = eval_scale(task_no=0)
        c1 = eval_scale(task_no=1)

        try: f_acqu[x[:,1]==1] = f_acqu[x[:,1]==1]/c1
        except: pass
        try: f_acqu[x[:,1]==0] = f_acqu[x[:,1]==0]/c0
        except: pass

        #print(x,"->",f_acqu)
        return f_acqu
    
acquisition = MyAcquisition(model, space, optimizer=acquisition_optimizer, cost_withGradients=cost)

evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
bo = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition, evaluator, initial_design,
                                                normalize_Y=False)

task_counter = {} # reset counter
for i in range(1, max_iter+1):
    bo.run_optimization(1)
    print('next_locations: ', bo.suggest_next_locations())
    # time.sleep(1.0);
    plot_bo(bo)
    plt.gcf().suptitle("[iter. %i/%i] [#task evaluations: %s]" % (i, max_iter, task_counter), fontsize=14)
    display.display(pl.gcf())
    plt.show() 
    # display.clear_output(wait=True);