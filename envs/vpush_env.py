import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import pygame
import random
import math
from sim.vpush_sim import VPushSimulation  # Assuming VPushSimulation is in a separate file.
import time
from stable_baselines3.common.utils import get_linear_fn

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def handle_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

class VPushSimulationEnv(gym.Env):
    def __init__(self, render_mode='human', object_type='circle', v_angle=np.pi/3, img_size=(42, 42), obs_type='pose'):
        super(VPushSimulationEnv, self).__init__()
        self.simulation = VPushSimulation(object_type, v_angle)
        self.gui = True if render_mode == 'human' else False
        self.img_size = img_size  # New parameter for image size
        self.obs_type = obs_type  # New parameter for observation type
        self.action_space = spaces.Box(low=np.array([-1, -1, -0.2]), high=np.array([1, 1, 0.2]), dtype=np.float32)
        # self.action_res = 10
        # self.action_space = spaces.MultiDiscrete([self.action_res, self.action_res, ])
        # discrete action space
        # self.action_space = spaces.Discrete(6)
        
        if self.obs_type == 'image':
            # Observation space: smaller RGB image of the simulation
            self.observation_space = spaces.Box(low=0, high=1.0, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.float64)
        else:
            # Observation space: low-dimensional pose (object pose, gripper pose)
            self.observation_space = spaces.Box(low=np.array([-1.0]*6), high=np.array([1.0]*6), dtype=np.float64)
        
        # Goal parameters
        self.goal_radius = 10.0
        self.goal_position = np.array([30, 30])
        self.simulation.goal_position = self.goal_position
        self.simulation.goal_radius = self.goal_radius

        # Pygame setup
        if self.gui:
            pygame.init()
            self.screen = pygame.display.set_mode((self.simulation.width, self.simulation.height))
            pygame.display.set_caption('Box2D with Pygame - Robot Pushing Object')
            
        self.last_gripper_pose = None
        self.last_object_pose = None
        self.last_action = None
        self.last_angle_difference = None

    def reset(self, seed=None, options=None):
        print("Resetting the environment")
        super().reset(seed=seed)
        self.simulation.step_count = 0

        # Set goal position randomly within specified range
        # self.goal_position = np.array([random.uniform(-10, 10), random.uniform(20, 40)])
        # self.simulation.goal_position = self.goal_position

        # Randomize robot position and orientation
        # self.simulation.robot_body.position = (random.uniform(-20, 20), random.uniform(10, 50))
        self.simulation.robot_body.position = (-30+random.normalvariate(0, 3), 30+random.normalvariate(0, 3))
        # self.simulation.robot_body.angle = random.normalvariate(0, math.pi/12)
        self.simulation.robot_body.angle = random.uniform(-math.pi/4, math.pi/4)

        # Ensure object does not penetrate gripper: place it at least some distance away from the gripper
        min_distance = 10  # Minimum distance to ensure no penetration
        while True:
            obj_x = random.normalvariate(0, 3)
            obj_y = random.normalvariate(30, 3)
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

    def step(self, action, sim_steps=1):        
        
        # noise = np.random.normal(0, 0.1, size=action.shape)
        # action = action + noise
        
        # Rescale action to desired range
        # action = 2*action / self.action_res - 1
        # action = action * np.array([.1, .1,])
        # action = action * np.array([.01, .01,])
        # action = action * np.array([.01, .01, .001])
        # action = np.clip(rescaled_action, self.action_space.low, self.action_space.high)
        # if action == 0:
        #     action = (0.0, 0.02, 0)
        # elif action == 1:
        #     action = (0.0, -0.02, 0)
        # elif action == 2:
        #     action = (-0.02, 0.0, 0)
        # elif action == 3:
        #     action = (0.02, 0.0, 0)
        # elif action == 4:
        #     action = (0.0, 0.0, 0.001)
        # elif action == 5:
        #     action = (0.0, 0.0, -0.001)

        self.last_object_pose = np.array(list(self.simulation.object_body.position) + [self.simulation.object_body.angle,])
        self.last_gripper_pose = np.array(list(self.simulation.robot_body.position) + [self.simulation.robot_body.angle,])

        # Get the current velocities
        current_linear_velocity = self.simulation.robot_body.linearVelocity
        current_angular_velocity = self.simulation.robot_body.angularVelocity

        # Apply the velocity offsets
        new_linear_velocity = (
            float(action[0]),
            float(action[1])
        )
        new_angular_velocity = float(action[2])

        # Set the new velocities
        self.simulation.robot_body.linearVelocity = new_linear_velocity
        self.simulation.robot_body.angularVelocity = new_angular_velocity

        # Step the simulation
        for _ in range(sim_steps):
            self.simulation.world.Step(self.simulation.timeStep, self.simulation.vel_iters, self.simulation.pos_iters)
        self.simulation.world.ClearForces()
        self.simulation.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        done = self._is_done()
        truncated = self._is_truncated()
        self.last_action = action
        return obs, reward, done, truncated, {}

    def _get_obs(self):
        # Render the screen and capture the image
        if self.gui:
            self.screen.fill(self.simulation.colors['background'])
            self.simulation.draw()
        pygame.display.flip()
        if self.obs_type == 'image':
            img = pygame.surfarray.array3d(pygame.display.get_surface())
            img = np.transpose(img, (1, 0, 2))  # Convert to the shape (height, width, 3)
            
            # Resize the image to the desired observation size
            img = pygame.transform.scale(pygame.surfarray.make_surface(img), self.img_size)
            img = pygame.surfarray.array3d(img)
            img = np.transpose(img, (1, 0, 2))  # Convert back to the shape (height, width, 3)
            
            # Normalize image observation
            img = img / 255.0
            return img
        else:
            # Low-dimensional observation: object pose, gripper pose, goal position
            object_pose = np.array(list(self.simulation.object_body.position) + [self.simulation.object_body.angle,])
            gripper_pose = np.array(list(self.simulation.robot_body.position) + [self.simulation.robot_body.angle,])
            # goal_position = self.goal_position

            obs = np.concatenate([object_pose, gripper_pose])
            
            # Normalize low-dimensional observation
            max_vals = np.array([self.simulation.width / 20, self.simulation.height / 10, np.pi] * 2 ) 
                                # + [self.simulation.width / 20, self.simulation.height / 10,])
            obs_normalized = obs / max_vals
            
            return obs_normalized

    def _compute_reward(self, action):
        object_pos = np.array(self.simulation.object_body.position)
        gripper_pos = np.array(self.simulation.robot_body.position)
        # object_vel = np.array(self.simulation.object_body.linearVelocity)
        # gripper_vel = np.array(self.simulation.robot_body.linearVelocity)
        distance_to_goal = np.linalg.norm(object_pos - self.goal_position)

        reward = 0
        
        # if self.last_action is not None:
        #     reward -= abs(self.last_action[-1] - action[-1])
        
        # last_direction_angle = math.atan2(self.last_object_pose[1] - self.last_gripper_pose[1], self.last_object_pose[0] - self.last_gripper_pose[0])
        gripper_angle = self.simulation.robot_body.angle
        current_direction_angle = math.atan2(self.goal_position[1] - object_pos[1], self.goal_position[0] - object_pos[0])
        current_angle_difference = abs(current_direction_angle - gripper_angle)
        # print('gripper_angle', gripper_angle, 'current_direction_angle', current_direction_angle)
        if self.last_angle_difference is not None:
            reward = (self.last_angle_difference - current_angle_difference) * 5
        self.last_angle_difference = current_angle_difference
        
        if distance_to_goal > self.goal_radius:
            # reward = -distance_to_object*0.01 + min(1000, -np.log(distance_to_goal*0.3))  # Negative reward proportional to distance
            # reward = 10/(distance_to_goal+3*distance_to_object)
            # reward = np.exp(-0.1*distance_to_object) + 3*np.exp(-0.1*distance_to_goal)
            # reward = -distance_to_object*0.01
            last_distance_to_object = np.linalg.norm(self.last_gripper_pose[:2] - self.last_object_pose[:2])
            current_distance_to_object = np.linalg.norm(gripper_pos - object_pos)
            reward += last_distance_to_object - current_distance_to_object
        else:
            reward += 100  # High positive reward for reaching the goal

        # if abs(object_pos[0]) > 38 or (object_pos[1] < 2 or object_pos[1] > 58) or abs(gripper_pos[0]) > 38 or (gripper_pos[1] < 2 or gripper_pos[1] > 58):
        #     reward -= 100  # Negative reward for going out of bounds

        # Penalize high velocities
        # reward += np.exp((-np.linalg.norm(object_vel) * 10 - np.linalg.norm(gripper_vel) * 10))
        # print('reward', reward)

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
        canvas_min_x, canvas_max_x = -self.simulation.width / 20, self.simulation.width / 20  # -40,40
        canvas_min_y, canvas_max_y = 0, self.simulation.height / 10  # 0,60
        
        # Check if the gripper/object is out of the canvas
        gripper_out_of_canvas = not (canvas_min_x <= gripper_pos[0] <= canvas_max_x and canvas_min_y <= gripper_pos[1] <= canvas_max_y)
        object_out_of_canvas = not (canvas_min_x <= object_pos[0] <= canvas_max_x and canvas_min_y <= object_pos[1] <= canvas_max_y)
    
        time_ended = self.simulation.step_count >= 10000  # Maximum number of steps

        return bool(gripper_out_of_canvas or object_out_of_canvas or time_ended)

    def render(self, mode='human'):
        if self.gui:
            handle_pygame_events()
            pygame.display.flip()

    def close(self):
        pygame.quit()

