import collections
from typing import Dict, Union
import time
import numpy as np
from pynput.keyboard import Listener, KeyCode
import os
import sys
import threading

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
            'q': np.array([0.0, 0.0, self.movement_scale]),   # Up Z
            'e': np.array([0.0, 0.0, -self.movement_scale]),  # Down Z
            ' ': np.array([0.0, 0.0, 0.0])  # Space for no movement/stay
        }
        
        # Start keyboard listener
        self.listener = Listener(on_press=self._on_press, on_release=None)
        self.listener.start()
        
        print("Synchronous teleoperation mode for xarm:")
        print("Use i/k to move along X, j/l along Y, Q/E along Z, SPACE for no movement.")
        print("Press any mapped key to execute one step. Press 'Esc' to quit.")
    
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
        print("Waiting for keyboard input... (press a movement key)")
        
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
    
    print(f'Observation space: {env.observation_space}')
    print(f'Action space: {env.action_space}')

    expert = KeyboardTeleopPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        movement_scale=0.002
    )
    
    try:
        
        rollouts_data_file = "data/expert_rollouts.pkl"
        if os.path.exists(rollouts_data_file):
            with open(rollouts_data_file, "rb") as f:
                rollouts = pickle.load(f)
            print(f"Loaded expert rollouts from {rollouts_data_file}")
        else:
            rollouts = []

        rollouts.extend(
            rollout.rollout(
            expert,
            env,
            rollout.make_sample_until(min_episodes=3),
            unwrap=False,
            rng=rng,
            )
        )
        with open(rollouts_data_file, "wb") as f:
            pickle.dump(rollouts, f)
        print(f"Collected {len(rollouts)} expert rollouts.")
        
        transitions = rollout.flatten_trajectories(rollouts)
        
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            rng=rng,
        )
        
        bc_trainer.train(n_epochs=5)
        reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 3)
        print(f"Reward after training: {reward_after_training}")

        bc_trainer.policy.save("data/panda_bc_policy.pth")
        
    except KeyboardInterrupt:
        print("\nStopping teleoperation...")
    finally:
        expert.stop()
        env.close()




