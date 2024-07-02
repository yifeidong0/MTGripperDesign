# Multi-task Gripper Co-Design

## Folder Descriptions

### optimizer
Contains the three optimization algorithms:
- `grid_search.py`: Implements the grid search optimization algorithm.
- `mtbo.py`: Implements the multi-task Bayesian optimization algorithm.
- `random_search.py`: Implements the random search optimization algorithm.

### sim
Contains the three simulation environments:
- `push_sim.py`: Simulation environment for the "push" task.
- `vpush_sim.py`: Simulation environment for the "vpush" task.
- `ucatch_sim.py`: Simulation environment for the "ucatch" task.

### utils
Contains utility functions for the project:
- `plot_bo.py`: Contains the plotting function for Bayesian optimization results.

### main.py
The main script to run the optimization algorithms. (TODO)

### requirements.txt
A list of Python dependencies required to run the project.
