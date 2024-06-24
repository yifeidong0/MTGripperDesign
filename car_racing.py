import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# Custom callback to render the environment during training
class RenderCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        self.env.render("human")  # Render only the first environment
        return True

# Create the CarRacing-v2 environment
env_id = "CarRacing-v2"
env = gym.make(env_id, render_mode="rgb_array")  # Use "rgb_array" for training and "human" for rendering

# Parallel environments
vec_env = make_vec_env(env_id, n_envs=4, env_kwargs={"render_mode": "rgb_array"})

# Create the PPO model
model = PPO("CnnPolicy", vec_env, verbose=1)

# Create an instance of the callback
render_callback = RenderCallback(vec_env)

# Train the model with the callback
model.learn(total_timesteps=25000, callback=render_callback)
model.save("ppo_carracing")

# Load the trained model
model = PPO.load("ppo_carracing")

# Reset the environment
obs = vec_env.reset()

# Run the trained model in the environment
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")  # Render only the first environment

    if dones.any():
        obs = vec_env.reset()

# Close the environment
env.close()
