import collections
from typing import Dict, Union
import time
import numpy as np
from pynput.keyboard import Listener, KeyCode
import os
import sys
import threading
import wandb
import torch as th
# Add this line to import custom environments
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # This registers the custom environments

import pickle
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
import imitation.policies.base as base_policies
from imitation.util import util
from imitation.util.util import make_vec_env
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data import rollout
from imitation.data.types import Transitions
from imitation.algorithms.bc import BC
from imitation.data import types
import tempfile

from imitation.policies.base import FeedForward32Policy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
from imitation.data.types import maybe_unwrap_dictobs, Trajectory
from stable_baselines3.common.torch_layers import CombinedExtractor, FlattenExtractor

class DeepPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        if isinstance(observation_space, gym.spaces.Dict):
            kwargs["features_extractor_class"] = CombinedExtractor
        else:
            kwargs["features_extractor_class"] = FlattenExtractor

        kwargs["net_arch"] = [dict(pi=[256, 256, 256, 256], vf=[256, 256, 256, 256])]
        kwargs["activation_fn"] = nn.ReLU
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

def flatten_demos(trajs):
    flat = []
    for traj in trajs:
        obs_dict = maybe_unwrap_dictobs(traj.obs)
        keys = sorted(obs_dict.keys())
        obs_flat = np.concatenate([obs_dict[k] for k in keys], axis=1)

        flat.append(
            Trajectory(
                obs=obs_flat,
                acts=traj.acts,
                infos=None,
                terminal=True,
            )
        )
    return flat

if __name__ == "__main__":
    mode = "train"  # "train" or "test"
    n_train_epochs = 500 # for BC
    eval_every_n_epochs = 1 # for BC
    n_eval_episodes = 20
    movement_scale = 0.007
    n_demo_episodes = 0
    dagger_steps = 2000
    render_mode = "rgb_array"  # "human" (w. Bullet GUI), "rgb_array" (w.o. GUI)
    wandb_mode = "disabled" # "online", "disabled"
    algo = "dagger"  # Algorithm to use, e.g., "bc", "dagger", etc.
    
    # Initialize wandb
    wandb.init(
        mode=wandb_mode,
        project="panda-imitation-learning",
        config={
            "environment": "PandaUPushEnv-v0",
            "movement_scale": movement_scale,
            "n_train_epochs": n_train_epochs,
            "n_eval_episodes": n_eval_episodes,
            "n_demo_episodes": n_demo_episodes,
        }
    )
    
    rng = np.random.default_rng(0)
    if algo == "bc":
        env = make_vec_env(
            'PandaUPushEnv-v0',
            n_envs=1,
            max_episode_steps=1000,
            rng=rng,
            env_make_kwargs={
                "render_mode": render_mode, # "human",  # Render mode for interactive policy
            },
        )
    elif algo == "dagger":
        from gymnasium.wrappers.flatten_observation import FlattenObservation
        from stable_baselines3.common.vec_env import DummyVecEnv
        def make_env():
            e = gym.make("PandaUPushEnv-v0", render_mode=render_mode, max_episode_steps=1000)
            return FlattenObservation(e)
        env = DummyVecEnv([make_env])

    if mode == "train":
        print(f'Observation space: {env.observation_space}')
        print(f'Action space: {env.action_space}')
        
        rollouts_data_file = "data/expert_rollouts_rand_morph.pkl"
        os.makedirs(os.path.dirname(rollouts_data_file), exist_ok=True)
        
        if os.path.exists(rollouts_data_file):
            with open(rollouts_data_file, "rb") as f:
                rollouts = pickle.load(f)
            print(f"Loaded expert rollouts from {rollouts_data_file} with {len(rollouts)} rollouts.")
        else:
            rollouts = []
            print("No existing rollouts found. Starting fresh collection...")

        # Remove the last rollout in rollouts
        # if rollouts:
        #     rollouts.pop()

        # with open(rollouts_data_file, "wb") as f:
        #     pickle.dump(rollouts, f)
        # print(f"Collected {len(rollouts)} expert rollouts.")
        
        # process rollouts
        action_low, action_high = env.action_space.low, env.action_space.high
        print(f"Action space low: {action_low}, high: {action_high}")
        action_low_toset = -1
        action_high_toset = 1
        import dataclasses
        for i in range(len(rollouts)):
            rollouts[i] = dataclasses.replace(rollouts[i], acts=(rollouts[i].acts - action_low) * (action_high_toset - action_low_toset) / (action_high - action_low) + action_low_toset)
        
        all_traj_obs = []
        for i, traj in enumerate(rollouts):
            obs_dict = maybe_unwrap_dictobs(traj.obs)
            keys = sorted(obs_dict.keys())
            obs_flat = np.concatenate([obs_dict[k] for k in keys], axis=1)
            # Update the trajectory with flattened observations
            rollouts[i] = dataclasses.replace(traj, obs=obs_flat)
            all_traj_obs.append(obs_flat)
            
        all_traj_obs = np.concatenate(all_traj_obs, axis=0)
        mean_obs = np.mean(all_traj_obs, axis=0)
        std_obs = np.std(all_traj_obs, axis=0)
        with open("data/obs_statistics.pkl", "wb") as f:
            pickle.dump({"mean": mean_obs, "std": std_obs}, f)
        print(f"Mean observation: {mean_obs}, Std observation: {std_obs}")
        
        for i in range(len(rollouts)):
            rollouts[i] = dataclasses.replace(rollouts[i], obs=(rollouts[i].obs - mean_obs) / (std_obs + 1e-8))
        
        rollouts_data_file_norm = "data/expert_rollouts_rand_morph_normalized.pkl"
        with open(rollouts_data_file_norm, "wb") as f:
            pickle.dump(rollouts, f)           
          
    env.close()
