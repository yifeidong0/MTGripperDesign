import time
import os
import sys
import gymnasium as gym
import numpy as np
from pynput.keyboard import Listener, KeyCode

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

import envs
from experiments.args_utils import get_args

# Define initial action (reset each step)
action = np.array([0.0, 0.0, 0.0])
action_lock = False  # Prevent multiple key events stacking

# Define key mappings
key_mapping = {
    'z': np.array([0.002, 0.0, 0.0]),  # Forward X
    'c': np.array([-0.002, 0.0, 0.0]), # Backward X
    'a': np.array([0.0, -0.002, 0.0]), # Left Y
    'd': np.array([0.0, 0.002, 0.0]),  # Right Y
    'q': np.array([0.0, 0.0, 0.002]),  # Up Z
    'e': np.array([0.0, 0.0, -0.002])  # Down Z
}

def on_press(key):
    """Update action for one step only."""
    global action, action_lock
    if not action_lock and hasattr(key, 'char') and key.char in key_mapping:
        action = key_mapping[key.char]  # Apply movement
        action_lock = True  # Lock input until release

def on_release(key):
    """Reset action to zero when key is released."""
    global action, action_lock
    action = np.array([0.0, 0.0, 0.0])  # Reset action
    action_lock = False  # Unlock input

# Start listening for keyboard input
listener = Listener(on_press=on_press, on_release=on_release)
listener.start()

def get_teleop_action():
    """Return current action (reset each step)."""
    global action
    return action.copy()

def main():
    args = get_args()
    env_id = 'PandaUPushEnv-v0'
    env_kwargs = {'obs_type': args.obs_type, 
                  'using_robustness_reward': args.using_robustness_reward, 
                  'render_mode': 'human',
                  'perturb': args.perturb,
                  'reward_type': args.reward_type,
                  }
    
    env = gym.make(env_id, **env_kwargs)
    obs, _ = env.reset(seed=1)

    print("Teleoperation mode: Use W/S to move along X, A/D along Y, Q/E along Z. Press 'Esc' to quit.")

    while True:
        action = get_teleop_action()
        obs, reward, done, truncated, _ = env.step(action)
        env.render()

        time.sleep(0.02)  # Adjust sleep to match simulation rate

    env.close()

if __name__ == "__main__":
    main()
