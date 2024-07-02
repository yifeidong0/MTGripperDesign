# Multi-task Gripper Co-Design

Folder structure and description:

.
├── optimizer
│   ├── grid_search.py
│   ├── mtbo.py
│   └── random_search.py
├── sim
│   ├── push_sim.py
│   ├── ucatch_sim.py
│   └── vpush_sim.py
├── utils
│   └── plot_bo.py
├── main.py
├── README.md
└── requirements.txt

- optimizer: Contains the three optimization algorithms: grid search, random search, and multi-task Bayesian optimization.
- sim: Contains the three simulation environments: push, vpush, and ucatch.
- utils: Contains the plotting function for Bayesian optimization.
- main.py: The main script to run the optimization algorithms. (TODO)