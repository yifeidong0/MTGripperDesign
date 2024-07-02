# Multi-task Gripper Co-Design

## Folder Descriptions

### optimizer
Contains the three optimization algorithms:
- `grid_search.py`: Implements the grid search optimization algorithm.
- `mtbo.py`: Implements the multi-task Bayesian optimization algorithm.
- `random_search.py`: Implements the random search optimization algorithm.

### sim
Contains the three simulation environments:

#### `push_sim.py` (toy example)
- **Action Space**: 
  - Dimensions: 3
  - Meaning: linear and angular velocity change applied to the gripper
- **Design Space**: 
  - Dimensions: 1
  - Meaning: length of the bar-gripper
- **Number of Tasks**: 2 (pushing a ball, pushing a box)
- **Simulator**: Custom Pybullet-based simulation

#### `vpush_sim.py`
- **Action Space**: 
  - Dimensions: 3
  - Meaning: linear and angular velocity change applied to the gripper
- **Design Space**: 
  - Dimensions: 1
  - Meaning: Angle of the V-shape gripper
- **Number of Tasks**: 2 (pushing a circle, pushing a polygon)
- **Simulator**: Custom Box2D-based simulation

#### `ucatch_sim.py`
- **Action Space**: 
  - Dimensions: 1
  - Meaning: horizontal linear velocity change applied to the gripper

- **Design Space**: 
  - Dimensions: 5
  - Meaning: Same as action space dimensions
    - `d0`, `d1`, `d2`: Lengths of different parts of the U-shaped gripper
    - `alpha0`, `alpha1`: Angles of different parts of the U-shaped gripper
- **Number of Tasks**: 2 (catching a circle, catching a polygon)
- **Simulator**: Custom Box2D-based simulation

### rl
Contains the visuomotor reinforcement learning modules in the gym environments:

#### `push_env.py`
- **Environment**: Custom Gym environment for the push task.
- **Action Space**: 
  - Dimensions: 3
  - Meaning: linear and angular velocity change applied to the gripper
- **Observation Space**: 
  - Dimensions: Varies (e.g., image data from a camera, state vectors)
  - Meaning: Observations include the current state of the object and possibly visual data
- **Reward Structure**: Typically designed to encourage the agent to push the object to a target region.

#### `vpush_env.py`
- **Environment**: Custom Gym environment for the variable push task.
- **Action Space**: 
  - Dimensions: 3
  - Meaning: linear and angular velocity change applied to the gripper
- **Observation Space**: 
  - Dimensions: Varies (e.g., image data from a camera, state vectors)
  - Meaning: Observations include the current state of the object and possibly visual data
- **Reward Structure**: Typically designed to encourage the agent to approach the object (intermediate reward) and push it to a target location with variable initial conditions.

### utils
Contains utility functions for the project:
- `plot_bo.py`: Contains the plotting function for Bayesian optimization results.
- `plot_bo_multi.py`: Contains the plotting function for Bayesian optimization (multiple dimensional design space, i.e. >=2) results.

### main.py
The main script to run the optimization algorithms. (TODO)

### requirements.txt
A list of Python dependencies required to run the project.

## Tentative Environments to be Added Soon (TODO)

1. **Scooping deformable with a shovel**:
   - **Simulator**: Pybullet
   - **Design Space**: Spline parameters

2. **Wearing masks (linear deformable objects)**:
   - **Simulator**: Pybullet
   - **Design Space**: (To be determined)
   - **Observation Space**: Key point positions

3. **Scooping meat (volumetric deformable objects)**:
   - **Simulator**: Pybullet / Isaac Gym
   - **Design Space**: 3D design parameters
