import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import pygame
import random
import math
from sim.vpush_sim import Box2DSimulation  # Assuming Box2DSimulation is in a separate file.
import time
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "training_log.txt")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def _on_step(self) -> bool:
        if self.n_calls % self.locals['self'].n_steps == 0:  # Save after each epoch
            # Access the logger's storage
            log_data = self.logger.name_to_value
            metrics = {
                "iterations": self.num_timesteps,
                "ep_len_mean": log_data.get("rollout/ep_len_mean", 0),
                "ep_rew_mean": log_data.get("rollout/ep_rew_mean", 0),
                "fps": log_data.get("time/fps", 0),
                "time_elapsed": log_data.get("time/time_elapsed", 0),
                "approx_kl": log_data.get("train/approx_kl", 0),
                "clip_fraction": log_data.get("train/clip_fraction", 0),
                "clip_range": log_data.get("train/clip_range", 0),
                "entropy_loss": log_data.get("train/entropy_loss", 0),
                "explained_variance": log_data.get("train/explained_variance", 0),
                "learning_rate": log_data.get("train/learning_rate", 0),
                "loss": log_data.get("train/loss", 0),
                "n_updates": log_data.get("train/n_updates", 0),
                "policy_gradient_loss": log_data.get("train/policy_gradient_loss", 0),
                "std": log_data.get("train/std", 0),
                "value_loss": log_data.get("train/value_loss", 0)
            }
            with open(self.log_file, "a") as f:
                f.write(f"{metrics}\n")
        return True

def handle_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

class Box2DSimulationEnv(gym.Env):
    def __init__(self, object_type='circle', v_angle=np.pi/6, gui=True, img_size=(84, 84)):
        super(Box2DSimulationEnv, self).__init__()
        self.simulation = Box2DSimulation(object_type, v_angle)
        self.gui = gui
        self.img_size = img_size  # New parameter for image size
        self.action_space = spaces.Box(low=np.array([-0.03, -0.03, -0.01]), high=np.array([0.03, 0.03, 0.01]), dtype=np.float32)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Observation space: smaller RGB image of the simulation
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        
        # Goal parameters
        self.goal_radius = 5.0
        self.goal_position = np.array([random.uniform(-20, 20), random.uniform(10, 50)])
        self.simulation.goal_position = self.goal_position
        self.simulation.goal_radius = self.goal_radius

        # Pygame setup
        if self.gui:
            pygame.init()
            self.screen = pygame.display.set_mode((self.simulation.width, self.simulation.height))
            pygame.display.set_caption('Box2D with Pygame - Robot Pushing Object')

    def reset(self, seed=None, options=None):
        print("Resetting the environment")
        super().reset(seed=seed)
        self.simulation.step_count = 0

        # Set goal position randomly within specified range
        self.goal_position = np.array([random.uniform(-10, 10), random.uniform(15, 45)])
        self.simulation.goal_position = self.goal_position

        # Randomize robot position and orientation
        self.simulation.robot_body.position = (random.uniform(-20, 20), random.uniform(10, 50))
        self.simulation.robot_body.angle = random.uniform(-math.pi, math.pi)

        # Ensure object does not penetrate gripper: place it at least some distance away from the gripper
        min_distance = 10  # Minimum distance to ensure no penetration
        while True:
            obj_x = random.uniform(-25, 25)
            obj_y = random.uniform(10, 50)
            distance = np.linalg.norm(np.array([obj_x, obj_y]) - np.array(self.simulation.robot_body.position))
            if distance >= min_distance:
                break
        
        self.simulation.object_body.position = (obj_x, obj_y)
        self.simulation.object_body.angle = random.uniform(-math.pi, math.pi)

        # Reset velocities of robot and object
        self.simulation.robot_body.linearVelocity = (0, 0)
        self.simulation.robot_body.angularVelocity = 0
        self.simulation.object_body.linearVelocity = (0, 0)
        self.simulation.object_body.angularVelocity = 0
        self.simulation.world.ClearForces()

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Add noise to action for exploration
        noise = np.random.normal(0, 0.01, size=action.shape)
        action = action + noise

        # Ensure action is within bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Get the current velocities
        current_linear_velocity = self.simulation.robot_body.linearVelocity
        current_angular_velocity = self.simulation.robot_body.angularVelocity

        # Apply the velocity offsets
        new_linear_velocity = (
            float(current_linear_velocity[0] + action[0]),
            float(current_linear_velocity[1] + action[1])
        )
        new_angular_velocity = float(current_angular_velocity + action[2])

        # Set the new velocities
        self.simulation.robot_body.linearVelocity = new_linear_velocity
        self.simulation.robot_body.angularVelocity = new_angular_velocity

        # Step the simulation
        self.simulation.world.Step(self.simulation.timeStep, self.simulation.vel_iters, self.simulation.pos_iters)
        self.simulation.world.ClearForces()
        self.simulation.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        truncated = self._is_truncated()

        return obs, reward, done, truncated, {}

    def _get_obs(self):
        # Render the screen and capture the image
        self.screen.fill(self.simulation.colors['background'])
        self.simulation.draw()
        self._draw_goal()

        pygame.display.flip()
        img = pygame.surfarray.array3d(pygame.display.get_surface())
        img = np.transpose(img, (1, 0, 2))  # Convert to the shape (height, width, 3)
        
        # Resize the image to the desired observation size
        img = pygame.transform.scale(pygame.surfarray.make_surface(img), self.img_size)
        img = pygame.surfarray.array3d(img)
        img = np.transpose(img, (1, 0, 2))  # Convert back to the shape (height, width, 3)
        
        return img

    def _draw_goal(self):
        pygame.draw.circle(self.screen, (255, 255, 0), self.simulation.to_pygame(self.goal_position), int(self.goal_radius * 10))

    def _compute_reward(self):
        object_pos = np.array(self.simulation.object_body.position)
        distance_to_goal = np.linalg.norm(object_pos - self.goal_position)
        if distance_to_goal > self.goal_radius:
            reward = -distance_to_goal*0.1  # Negative reward proportional to distance
        else:
            reward = 1000  # High positive reward for reaching the goal
        return reward

    def _is_done(self):
        object_pos = np.array(self.simulation.object_body.position)
        distance_to_goal = np.linalg.norm(object_pos - self.goal_position)
        return bool(distance_to_goal < self.goal_radius)

    def _is_truncated(self):
        # Get the position of the gripper and object
        gripper_pos = np.array(self.simulation.robot_body.position)
        object_pos = np.array(self.simulation.object_body.position)
        
        # Define canvas boundaries
        canvas_min_x, canvas_max_x = -self.simulation.width / 20, self.simulation.width / 20 # -40,40
        canvas_min_y, canvas_max_y = 0, self.simulation.height / 10 # 0,60
        
        # Check if the gripper/object is out of the canvas
        gripper_out_of_canvas = not (canvas_min_x <= gripper_pos[0] <= canvas_max_x and canvas_min_y <= gripper_pos[1] <= canvas_max_y)
        object_out_of_canvas = not (canvas_min_x <= object_pos[0] <= canvas_max_x and canvas_min_y <= object_pos[1] <= canvas_max_y)
    
        time_ended = self.simulation.step_count >= 1000

        return bool(gripper_out_of_canvas or object_out_of_canvas or time_ended)

    def render(self, mode='human'):
        if self.gui:
            handle_pygame_events()
            pygame.display.flip()

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    v_angle = math.pi / 6  # Example angle in radians (30 degrees)
    object_type = 'polygon'  # Can be 'circle' or 'polygon'
    log_dir = "./logs"
    env = Box2DSimulationEnv(object_type, v_angle, gui=True, img_size=(42, 42))  # Use smaller image size
    check_env(env)

    callback = CustomCallback(log_dir=log_dir, verbose=1)
    model = PPO("CnnPolicy", env, verbose=1, learning_rate=1e-5)

    model.learn(total_timesteps=300000, progress_bar=True, callback=callback)

    obs, _ = env.reset(seed=0)
    for episode in range(1):
        time.sleep(2)
        print(f"Episode {episode + 1} begins")
        done, truncated = False, False
        while not (done or truncated):
            action, _states = model.predict(np.array(obs))
            obs, reward, done, truncated, _ = env.step(action)
            env.render()
            handle_pygame_events()  # Ensure events are handled during each step
            time.sleep(.002)
        print("Done!" if done else "Truncated")
        print(f"Episode {episode + 1} finished")
    
    env.close()

# Plot the Logged Metrics Using Matplotlib
import ast
import matplotlib.pyplot as plt

def plot_metrics(log_file):
    metrics = {
        "iterations": [],
        "ep_len_mean": [],
        "ep_rew_mean": [],
        "fps": [],
        "time_elapsed": [],
        "approx_kl": [],
        "clip_fraction": [],
        "clip_range": [],
        "entropy_loss": [],
        "explained_variance": [],
        "learning_rate": [],
        "loss": [],
        "n_updates": [],
        "policy_gradient_loss": [],
        "std": [],
        "value_loss": []
    }
    
    with open(log_file, "r") as f:
        for line in f:
            data = ast.literal_eval(line.strip())
            for key in metrics.keys():
                metrics[key].append(data[key])
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(metrics["iterations"], metrics["ep_len_mean"], label="Episode Length Mean")
    plt.plot(metrics["iterations"], metrics["ep_rew_mean"], label="Episode Reward Mean")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(metrics["iterations"], metrics["loss"], label="Loss")
    plt.plot(metrics["iterations"], metrics["value_loss"], label="Value Loss")
    plt.plot(metrics["iterations"], metrics["policy_gradient_loss"], label="Policy Gradient Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

log_file = "./logs/training_log.txt"
plot_metrics(log_file)
