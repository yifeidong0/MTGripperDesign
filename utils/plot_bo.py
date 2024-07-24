import numpy as np
import matplotlib.pyplot as plt
import os

def plot_err(x, y, yerr, color=None, alpha_fill=0.2, ax=None, label="", lw=1, ls="-"):
    y, yerr = y.reshape(-1), yerr.reshape(-1)
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label, lw=lw, ls=ls)
    # ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill, linewidth=0.0)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

# RENAME = {"Gaussian_noise.variance": "noise.var"}
# def plot_params(bo):
#     plt.xlim(-1,1)
#     plt.ylim(-1,1)
#     plt.axis('off')
#     for i, (n, v) in enumerate(zip(bo.model.get_model_parameters_names(), bo.model.get_model_parameters()[0])):
#         n = ".".join(n.split(".")[-2:])
#         n = RENAME.get(n, n)
#         plt.text(-0.9, 0.9-i*0.2, "%s:" % (n), ha="left", va="top", fontsize=13)
#         plt.text(0.5, 0.9-i*0.2, "%.3f" % (v), ha="left", va="top", fontsize=13)

# def plot_next_acquisitions(bo, DIM_NO=0, tasks_to_be_plotted=[0,1]):
    # for xt in bo.suggest_next_locations():
    #     x, t = xt[:-1], int(xt[-1])
    #     assert t==0 or t==1
    #     if t not in tasks_to_be_plotted:
    #         continue
    #     plt.axvline(x=x[DIM_NO], color=("salmon" if t==0 else "limegreen"), ls="--", lw=2, label="next acquisition")

def plot_next_acquisitions(bo, DIM_NO=0):
    for x in bo.suggest_next_locations():
        plt.axvline(x=x[DIM_NO], color=("salmon"), ls="--", lw=2, label="next acquisition")

# def plot_acquisition(bo, x_scale, DIM_NO=0, grid=0.001):
#     bounds = bo.acquisition.space.get_bounds()
#     x_grid = np.arange(bounds[DIM_NO][0], bounds[DIM_NO][1], grid)
#     x_grid = x_grid.reshape(len(x_grid), 1)
#     xt = np.hstack([x_grid, np.zeros((x_grid.shape[0], 1))])
#     acqu0 = -bo.acquisition.acquisition_function(xt)
#     plt.plot(x_grid, acqu0, label="task 0", color="red", lw=2)
#     xt = np.hstack([x_grid, np.ones((x_grid.shape[0], 1))])
#     acqu1 = -bo.acquisition.acquisition_function(xt)
#     plt.plot(x_grid, acqu1, label="task 1", color="green", lw=2)
#     plot_next_acquisitions(bo, DIM_NO)
#     plt.xlim(x_scale[0]-0.1, x_scale[-1]+0.1)
#     plt.legend(loc=2)
#     plt.title("acquisition function")
#     plt.grid()

def plot_acquisition(bo, x_scale, DIM_NO=0, grid=0.001):
    bounds = bo.acquisition.space.get_bounds()
    x_grid = np.arange(bounds[DIM_NO][0], bounds[DIM_NO][1], grid)
    x_grid = x_grid.reshape(len(x_grid), 1)
    acqu = -bo.acquisition.acquisition_function(x_grid)
    plt.plot(x_grid, acqu, color="red", lw=2)
    plot_next_acquisitions(bo, DIM_NO)
    plt.xlim(x_scale[0]-0.1, x_scale[-1]+0.1)
    plt.legend(loc=2)
    plt.title("acquisition function")
    plt.grid()

def plot_task(bo, TASK_NO, x_scale):
    # color = {0: "red", 1: "green"}[TASK_NO]
    # task_column = TASK_NO * np.ones((x_scale.shape[0], 1))
    # xt = np.hstack([x_scale.reshape(-1, 1), task_column])
    # means0mt, stds0mt = bo.model.predict(xt)
    x = x_scale.reshape(-1, 1)
    means0mt, stds0mt = bo.model.predict(x)
    plot_err(x_scale, means0mt, stds0mt, lw=1, ls="-.", color="blue", label="BO fit", alpha_fill=0.05)
    X, Y = bo.get_evaluations()
    # if len(X) > 0 and X[-1, 1] == TASK_NO:
    #     plt.scatter(X[-1, 0], Y[-1, 0], color=color, marker="o", s=60, label="prev acquisition")
    plt.scatter(X[-1, 0], Y[-1, 0], color="red", marker="o", s=60, label="prev acquisition")


    # Y, X = Y[X[:, 1] == TASK_NO], X[X[:, 1] == TASK_NO, 0]
    plt.scatter(X, Y, color="k", marker="x", label="acquisition pts")
    plot_next_acquisitions(bo, 0)
    plt.xlim(x_scale[0]-0.1, x_scale[-1]+0.1)
    # plt.ylim(0, 1.2)
    plt.legend(loc=2)
    # plt.title("task %i" % TASK_NO)
    plt.title("task")
    plt.grid()

def plot_bo(bo, x_scale, iter, id_run=0):
    """Plot the BO iteration. Two subplots"""
    plt.figure(figsize=(22, 4))
    plt.subplot(1, 2, 1)
    plot_task(bo, 0, x_scale,)
    plt.subplot(1, 2, 2)
    plot_acquisition(bo, x_scale, DIM_NO=0)
     
    # Create folder if not exists
    if not os.path.exists(f"results/png/{id_run}"):
        os.makedirs(f"results/png/{id_run}")
    plt.savefig(f"results/png/{id_run}/bo_vpush_iter_{iter}.png")
