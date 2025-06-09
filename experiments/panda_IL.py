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
from imitation.data import rollout
from imitation.data.types import Transitions
from imitation.algorithms.bc import BC
from imitation.data import types

# TODO: 1. finetune the BC policy with PPO
# 2. Reward function might need to be reshaped. Sometimes high reward is given when goal is not reached - object is pushed past the goal.
# 3. Randomize the initial state of the environment. Tool, object and goal position, object shape, etc.
# 4. Randomize tool design after each episode. Goal is to have a generalized policy over the design space.

# Implementation: 1. Panda model compatibility: Research 3 and old version.
# 2. How to log training return in wandb from imitation library? - bc_trainer.train(n_epochs=1)


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
    n_train_epochs = 10
    n_eval_episodes = 1
    movement_scale = 0.002
    n_demo_episodes = 1
    use_saved_demo = 1
    transitions_file = "results/il/expert_transitions.pkl"
    rollouts_file = "results/il/rollouts.pkl"
    wandb_mode = "online" # "online", "disabled"
    
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
            "use_saved_demo": use_saved_demo
        }
    )
    
    rng = np.random.default_rng(0)
    env = make_vec_env(
        'PandaUPushEnv-v0',
        n_envs=1,
        max_episode_steps=1000,
        rng=rng,
        env_make_kwargs={
            "render_mode": "human",  # Render mode for interactive policy
        },
    )
    
    expert = KeyboardTeleopPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        movement_scale=movement_scale
    )
    
    try:
        if use_saved_demo:
            with open(transitions_file, "rb") as f:
                transitions = pickle.load(f)
            with open(rollouts_file, "rb") as f:
                rollouts = pickle.load(f)
            print(f"Loaded {len(rollouts)} rollouts from {rollouts_file}")
        else:
            print("Collecting demonstrations...")
            print(f"Please complete {n_demo_episodes} episodes. Press ESC to quit at any time.")
            rollouts = rollout.rollout(
                expert,
                env,
                rollout.make_sample_until(min_episodes=n_demo_episodes),
                unwrap=False,
                rng=rng,
            )
            transitions = rollout.flatten_trajectories(rollouts)

            try:
                # Try to load existing rollouts
                with open(rollouts_file, "rb") as f:
                    saved_rollouts = pickle.load(f)
                print(f"Loaded {len(saved_rollouts)} saved rollouts")
                all_rollouts = saved_rollouts + rollouts
                transitions = rollout.flatten_trajectories(all_rollouts)
            except (FileNotFoundError, EOFError):
                # If no saved transitions exist or file is empty, use only new transitions
                transitions = transitions
                print(f"Using {len(rollouts)} new rollouts")

            # Save the combined transitions
            with open(transitions_file, "wb") as f:
                pickle.dump(transitions, f)
            with open(rollouts_file, "wb") as f:
                pickle.dump(all_rollouts, f)
            print(f"Saved {len(all_rollouts)} rollouts to {rollouts_file}")
            print(f"Expert transitions saved to {transitions_file}")
        
        # Log number of transitions collected
        wandb.log({"n_transitions": len(transitions)})
        
        print("!!! Starting training...")
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            rng=rng,
        )

        # Training loop with wandb logging
        returns = []
        device = bc_trainer.policy.device
        
        # Convert observations and actions to tensors
        obs_dict = types.maybe_unwrap_dictobs(transitions.obs)
        obs_tensor = types.map_maybe_dict(lambda x: th.tensor(x, device=device), obs_dict)
        acts_tensor = th.tensor(transitions.acts, device=device)
        
        for epoch in range(n_train_epochs):
            # Train for one epoch
            bc_trainer.train(n_epochs=1)
            
            # Evaluate training loss on the full training set
            with th.no_grad():
                metrics = bc_trainer.loss_calculator(bc_trainer.policy, obs_tensor, acts_tensor)
                train_loss = float(metrics.loss)
                
            # Evaluate policy
            mean_return, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes) # evaluation return (not train or validation loss)
            returns.append(mean_return)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "mean_return": mean_return,
                "train_loss": train_loss,
            })
            print(f"Epoch {epoch}: Mean Return = {mean_return:.2f}, Loss = {train_loss:.4f}")

        print("!!! Training completed!")
        print(f"Final mean return: {np.mean(returns[-5:]):.2f}")  # Average of last 5 epochs
        
    except KeyboardInterrupt:
        print("\nStopping teleoperation...")
    finally:
        expert.stop()
        env.close()
        wandb.finish()  # Close wandb run



