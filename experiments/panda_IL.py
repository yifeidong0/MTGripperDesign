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

# TODO: 1. finetune the BC policy with PPO
# 2. Reward function might need to be reshaped. Sometimes high reward is given when goal is not reached - object is pushed past the goal.
# 3. Randomize the initial state of the environment. Tool, object and goal position, object shape, etc.
# 4. Randomize tool design after each episode. Goal is to have a generalized policy over the design space.

# Implementation: 1. Panda model compatibility: Research 3 and old version.
# 2. How to log training return in wandb from imitation library? - bc_trainer.train(n_epochs=1)


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
    n_train_epochs = 200
    eval_every_n_epochs = 1
    n_eval_episodes = 10
    movement_scale = 0.005
    n_demo_episodes = 0
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

    print(f'Observation space: {env.observation_space}')
    print(f'Action space: {env.action_space}')

    expert = KeyboardTeleopPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        movement_scale=movement_scale
    )
    
    try:
        rollouts_data_file = "data/expert_rollouts.pkl"
        os.makedirs(os.path.dirname(rollouts_data_file), exist_ok=True)
        
        if os.path.exists(rollouts_data_file):
            with open(rollouts_data_file, "rb") as f:
                rollouts = pickle.load(f)
            print(f"Loaded expert rollouts from {rollouts_data_file} with {len(rollouts)} rollouts.")
        else:
            rollouts = []
            print("No existing rollouts found. Starting fresh collection...")

        if render_mode == "human" and n_demo_episodes > 0:
            rollouts.extend(
                rollout.rollout(
                expert,
                env,
                rollout.make_sample_until(min_episodes=n_demo_episodes),
                unwrap=False,
                rng=rng,
                )
            )

        # Remove the last rollout in rollouts
        # if rollouts:
        #     rollouts.pop()

        with open(rollouts_data_file, "wb") as f:
            pickle.dump(rollouts, f)
        print(f"Collected {len(rollouts)} expert rollouts.")
        
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
            bc_trainer.policy.save("data/panda_bc_policy.pth")

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
            mean_return, _ = evaluate_policy(final_policy, env, n_eval_episodes*5)
            print(f"Final mean return after DAgger: {mean_return:.2f}")
            final_policy.save("data/panda_dagger_policy.pth")

        print("!!! Training completed!")
        
    except KeyboardInterrupt:
        print("\nStopping teleoperation...")
    finally:
        expert.stop()
        env.close()
        wandb.finish()  # Close wandb run