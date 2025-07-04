import collections
import os
import pickle
import sys
import tempfile
import threading
import time
from typing import Dict, Union

import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.rescale_action import RescaleAction
import numpy as np
import torch as th
import torch.nn as nn
import wandb
# from pynput.keyboard import KeyCode, Listener
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.torch_layers import CombinedExtractor, FlattenExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

# Add path to import custom environments
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # This registers the custom environments

# Imitation learning imports
from imitation.algorithms import bc
from imitation.algorithms.bc import BC
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data import rollout, types
from imitation.data.types import Trajectory, Transitions, maybe_unwrap_dictobs
from imitation.policies.base import FeedForward32Policy
import imitation.policies.base as base_policies
from imitation.util import util
from imitation.util.util import make_vec_env

# TODO: 1. finetune the BC policy with PPO
# 2. Reward function might need to be reshaped. Sometimes high reward is given when goal is not reached - object is pushed past the goal.
# 3. Randomize the initial state of the environment. Tool, object and goal position, object shape, etc.
# 4. Randomize tool design after each episode. Goal is to have a generalized policy over the design space.

# Implementation: 1. Panda model compatibility: Research 3 and old version.
# 2. How to log training return in wandb from imitation library? - bc_trainer.train(n_epochs=1)

class DeepPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, use_sde=False, **kwargs):
        if isinstance(observation_space, gym.spaces.Dict):
            kwargs["features_extractor_class"] = CombinedExtractor
        else:
            kwargs["features_extractor_class"] = FlattenExtractor

        kwargs["net_arch"] = [dict(pi=[256, 256, 256, 256], vf=[256, 256, 256, 256])]
        kwargs["activation_fn"] = nn.ReLU
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

class PandaEnvWrapper(gym.Wrapper):
    """Wrapper for Panda environments to flatten observations and rescale actions."""
    
    def __init__(self, env: gym.Env, min_action: float = -1.0, max_action: float = 1.0, stat_file: str = "data/obs_statistics.pkl"):
        wrapped_env = FlattenObservation(env)
        wrapped_env = RescaleAction(wrapped_env, min_action=min_action, max_action=max_action)
        super().__init__(wrapped_env)
        with open(stat_file, "rb") as f:
            obs_info = pickle.load(f)
            self.obs_mean = obs_info["mean"]
            self.obs_std = obs_info["std"]
    
    def normalize_obs(self, obs, epsilon=1e-8):
        """Normalize observations using precomputed mean and std."""
        obs = (obs - self.obs_mean) / (self.obs_std + epsilon)
        return obs
    
    def reset(self, *, seed = None, options = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.normalize_obs(obs), info
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.normalize_obs(observation), reward, terminated, truncated, info
        
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


class KeyboardTeleopPolicy(base_policies.NonTrainablePolicy):
    """Keyboard teleoperation policy for xarm continuous control with synchronous input."""
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        movement_scale: float = 0.02,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )
        
        # Ensure we're working with continuous action space
        assert isinstance(action_space, gym.spaces.Box)
        
        self.movement_scale = movement_scale
        self.action = np.zeros(action_space.shape[0])
        self.action_ready = threading.Event()  # Event to signal when action is ready
        self.quit_requested = False
        
        # Define key mappings for xarm control (similar to panda)
        self.key_mapping = {
            'i': np.array([self.movement_scale, 0.0, 0.0]),   # Forward X
            'k': np.array([-self.movement_scale, 0.0, 0.0]),  # Backward X
            'j': np.array([0.0, self.movement_scale, 0.0]),  # Left Y
            'l': np.array([0.0, -self.movement_scale, 0.0]),   # Right Y
            'q': np.array([0.0, 0.0, self.movement_scale]),   # Rotate Z
            'e': np.array([0.0, 0.0, -self.movement_scale]),  # Rotate -Z
            ' ': np.array([0.0, 0.0, 0.0])  # Space for no movement/stay
        }
        
        # Start keyboard listener
        self.listener = Listener(on_press=self._on_press, on_release=None)
        self.listener.start()
        
        print("Synchronous teleoperation mode for xarm:")
        print("Use i/k to move along X, j/l along Y, Q/E along Z, SPACE for no movement.")
        print("Press 'Esc' to quit.")
    
    def _on_press(self, key):
        """Handle key press events - set action and signal ready."""
        if hasattr(key, 'char') and key.char in self.key_mapping:
            movement = self.key_mapping[key.char]
            # Pad with zeros if action space is larger than 3D (e.g., includes gripper)
            self.action = np.zeros(self.action_space.shape[0])
            self.action[:len(movement)] = movement
            self.action_ready.set()  # Signal that action is ready
        elif key == KeyCode.from_char('\x1b'):  # ESC key
            self.quit_requested = True
            self.action_ready.set()  # Signal to wake up waiting thread
            return False  # Stop listener
    
    def _choose_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Wait for keyboard input and return the action."""
        # Clear the event and wait for next key press
        self.action_ready.clear()
        self.action_ready.wait()  # Block until a key is pressed
        
        if self.quit_requested:
            # Return zero action if quit was requested
            return np.zeros(self.action_space.shape[0])
        
        return self.action.copy()
    
    def stop(self):
        """Stop the keyboard listener."""
        if hasattr(self, 'listener'):
            self.listener.stop()


if __name__ == "__main__":
    mode = "train"  # "train" or "test"
    n_train_epochs = 300 # for BC
    eval_every_n_epochs = 10 # for BC
    n_eval_episodes = 100
    movement_scale = 0.007
    dagger_steps = 2000
    render_mode = "human"  # "human" (w. Bullet GUI), "rgb_array" (w.o. GUI)
    wandb_mode = "disabled" # "online", "disabled"
    algo = "bc"  # Algorithm to use, e.g., "bc", "dagger", etc.
    
    # Initialize wandb
    wandb.init(
        mode=wandb_mode,
        project="panda-imitation-learning",
        config={
            "environment": "PandaUPushEnv-v0",
            "movement_scale": movement_scale,
            "n_train_epochs": n_train_epochs,
            "n_eval_episodes": n_eval_episodes,
        }
    )
    
    rng = np.random.default_rng(0)
    def make_env():
        e = gym.make("PandaUPushEnv-v0", render_mode=render_mode, max_episode_steps=1000)
        return PandaEnvWrapper(e)
    env = DummyVecEnv([make_env])

    if mode == "train":
        print(f'Observation space: {env.observation_space}')
        print(f'Action space: {env.action_space}')

        expert = KeyboardTeleopPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            movement_scale=movement_scale
        )
        
        try:
            # rollouts_data_file = "data/expert_rollouts.pkl"
            rollouts_data_file = "data/expert_rollouts_rand_morph_normalized.pkl"
            os.makedirs(os.path.dirname(rollouts_data_file), exist_ok=True)
            
            if os.path.exists(rollouts_data_file):
                with open(rollouts_data_file, "rb") as f:
                    rollouts = pickle.load(f)
                print(f"Loaded expert rollouts from {rollouts_data_file} with {len(rollouts)} rollouts.")
            else:
                rollouts = []
                print("No existing rollouts found. Starting fresh collection...")
            
            transitions = rollout.flatten_trajectories(rollouts)
            
            # Log number of transitions collected
            wandb.log({"n_transitions": len(transitions)})
            
            print("!!! Starting training...")
            bc_trainer = bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=transitions if algo == "bc" else None,
                rng=rng,
                batch_size=256,
                policy=DeepPolicy(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    lr_schedule=lambda _: 1e-4,
                ),
            )

            # Training loop with wandb logging
            returns = []
            device = bc_trainer.policy.device
            
            # Convert observations and actions to tensors
            obs_dict = types.maybe_unwrap_dictobs(transitions.obs)
            obs_tensor = types.map_maybe_dict(lambda x: th.tensor(x, device=device), obs_dict)
            acts_tensor = th.tensor(transitions.acts, device=device)
            
            if algo == "bc":
                for epoch in range(n_train_epochs):
                    # Train for one epoch
                    bc_trainer.train(n_epochs=1)
                    print(f"Epoch {epoch + 1}/{n_train_epochs} completed.")

                    if epoch % eval_every_n_epochs == 0:
                        # Evaluate training loss on the full training set
                        with th.no_grad():
                            metrics = bc_trainer.loss_calculator(bc_trainer.policy, obs_tensor, acts_tensor)
                            train_loss = float(metrics.loss)
                        
                        # Log to wandb
                        wandb.log({
                            "epoch": epoch,
                            "train_loss": train_loss,
                        })
                        print(f"Epoch {epoch}: Loss = {train_loss:.4f}")
                            
                # Evaluate policy
                mean_return, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes) # evaluation return (not train or validation loss)

                print(f"Evaluation mean return: {mean_return:.2f}")
                bc_trainer.policy.save("data/panda_bc_policy_rand_morph.pth")

            elif algo == "dagger":
                # TODO: Import policy from path - data/panda_bc_policy.pth

                flat_rollouts = flatten_demos(rollouts)
                with tempfile.TemporaryDirectory(prefix="dagger_") as scratch_dir:
                    dagger_trainer = SimpleDAggerTrainer(
                        venv=env,
                        scratch_dir=scratch_dir,
                        expert_policy=expert,
                        bc_trainer=bc_trainer,
                        expert_trajs=flat_rollouts,
                        rng=rng,
                    )
                    # run dataset aggregation + BC rounds
                    dagger_trainer.train(dagger_steps)
                final_policy = dagger_trainer.policy
                mean_return, _ = evaluate_policy(final_policy, env, n_eval_episodes)
                print(f"Final mean return after DAgger: {mean_return:.2f}")
                final_policy.save("data/panda_dagger_policy.pth")

            print("!!! Training completed!")
            
        except KeyboardInterrupt:
            print("\nStopping teleoperation...")
        finally:
            expert.stop()
            env.close()
            wandb.finish()  # Close wandb run

    else:  # test mode
        # select correct saved policy
        if algo == 'bc':
            policy_path = "data/panda_bc_policy_rand_morph.pth"
        else:
            policy_path = "data/panda_dagger_policy.pth"
        assert os.path.exists(policy_path), f"Policy file not found: {policy_path}"

        # load and evaluate
        policy: ActorCriticPolicy = ActorCriticPolicy.load(policy_path)
        mean_return, std_return = evaluate_policy(policy, env, n_eval_episodes=n_eval_episodes)
        print(f"Test mode ({algo}): mean_return={mean_return:.2f} ± {std_return:.2f} over {n_eval_episodes} episodes")

    env.close()
    wandb.finish()
