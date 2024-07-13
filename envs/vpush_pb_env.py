import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pybullet as p
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import random
import math
from sim.vpush_pb_sim import VPushPbSimulation  # Assuming VPushPbSimulation is in a separate file.
import time
from stable_baselines3.common.utils import get_linear_fn

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class VPushPbSimulationEnv(gym.Env):
    def __init__(self, gui=1, img_size=(42, 42), obs_type='pose'):
        super(VPushPbSimulationEnv, self).__init__()
        self.task = 'circle' 
        self.task_int = 0 if self.task == 'circle' else 1
        self.v_angle = np.pi/3
        self.simulation = VPushPbSimulation(self.task, self.v_angle, gui)
        self.gui = gui
        self.img_size = img_size  # New parameter for image size
        self.obs_type = obs_type  # New parameter for observation type
        self.action_space = spaces.Box(low=np.array([-1, -1, -0.2]), high=np.array([1, 1, 0.2]), dtype=np.float32)
        self.canvas_min_x, self.canvas_max_x = 0, 5
        self.canvas_min_y, self.canvas_max_y = 0, 5

        if self.obs_type == 'image':
            # Observation space: smaller RGB image of the simulation
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.float64)
        else:
            # Observation space: low-dimensional pose (object pose, gripper pose)
            self.observation_space = spaces.Box(low=np.array([0., 0., -1.]*2 +[0.,]*4), high=np.array([1.0]*10), dtype=np.float64)
        
        # Goal parameters
        self.goal_radius = 0.5
        self.goal_position = np.array([4.5, 2.5]) # workspace [[0,5], [0,5]]
        self.simulation.goal_position = self.goal_position
        self.simulation.goal_radius = self.goal_radius

        self.last_gripper_pose = None
        self.last_object_pose = None
        self.last_action = None
        self.last_angle_difference = None
        
        self.is_success = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulation.step_count = 0
        self.task = random.choice(['circle', 'polygon'])
        self.task_int = 0 if self.task == 'circle' else 1
        self.v_angle = random.uniform(0, np.pi)
        self.simulation.reset_task_and_design(self.task, self.v_angle)
        obs = self._get_obs()
        self.is_success = False
        return obs, {}

    def step(self, action, sim_steps=1):
        self.last_object_pose = np.array(list(p.getBasePositionAndOrientation(self.simulation.object_id)[0][:2])
                                          + [pi_2_pi(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.object_id)[1])[2]),])
        self.last_gripper_pose = np.array(list(p.getBasePositionAndOrientation(self.simulation.robot_id)[0][:2])
                                          + [pi_2_pi(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.robot_id)[1])[2]),])

        # Set the new velocities
        new_linear_velocity = [float(action[0]), float(action[1]), 0]
        new_angular_velocity = float(action[2])
        
        p.resetBaseVelocity(self.simulation.robot_id, linearVelocity=new_linear_velocity, angularVelocity=[0, 0, new_angular_velocity])

        # Step the simulation
        for _ in range(sim_steps):
            p.stepSimulation()
            width, height, rgbPixels, _, _ = p.getCameraImage(320, 320, 
                                                              viewMatrix=self.simulation.viewMatrix, 
                                                              projectionMatrix=self.simulation.projectionMatrix)
            # if self.gui:
            #     time.sleep(self.simulation.time_step)

            frame = np.reshape(rgbPixels, (height, width, 4))[:, :, :3]
            self.frame = np.uint8(frame)

        self.simulation.step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward(action)
        done = self._is_done()
        truncated = self._is_truncated()
        self.last_action = action
        
        info = {}
        if done or truncated:
            info['is_success'] = self.is_success
        return obs, reward, done, truncated, info

    def get_rgb_image(self):
        return self.frame
    
    def _get_obs(self):
        if self.obs_type == 'image': # TODO: debug image observation
            width, height, rgb, _, _ = p.getCameraImage(width=self.img_size[0], height=self.img_size[1])
            rgb = np.array(rgb).reshape(self.img_size[1], self.img_size[0], 4)
            rgb = rgb[:, :, :3]  # Discard the alpha channel
            rgb = rgb / 255.0  # Normalize image observation
            return rgb
        else:
            object_pose = np.array(list(p.getBasePositionAndOrientation(self.simulation.object_id)[0][:2]) + 
                                   [p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.object_id)[1])[2]])
            gripper_pose = np.array(list(p.getBasePositionAndOrientation(self.simulation.robot_id)[0][:2]) + 
                                    [p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.robot_id)[1])[2]])
            obs = np.concatenate([object_pose, gripper_pose, self.goal_position])
            max_vals = np.array([5.0, 5.0, np.pi, 5.0, 5.0, np.pi, 5, 5])  # Normalization constants
            obs_normalized = obs / max_vals

            # Concatenate with task and design parameters
            task_design_params_normalized = np.array([self.task_int, self.v_angle/np.pi])

            return np.concatenate([obs_normalized, task_design_params_normalized])

    def _compute_reward(self, action):
        object_pos = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0][:2])
        gripper_pos = np.array(p.getBasePositionAndOrientation(self.simulation.robot_id)[0][:2])
        distance_to_goal = np.linalg.norm(object_pos - self.goal_position)

        # Reward for aligning the gripper with the goal
        reward = 0
        gripper_angle = pi_2_pi(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.robot_id)[1])[2])
        current_direction_angle = math.atan2(self.goal_position[1] - object_pos[1], self.goal_position[0] - object_pos[0])
        current_angle_difference = abs(current_direction_angle - gripper_angle)
        
        if self.last_angle_difference is not None:
            reward = (self.last_angle_difference - current_angle_difference) * 5
        self.last_angle_difference = current_angle_difference
        
        # Reward for moving towards the goal
        if distance_to_goal > self.goal_radius:
            last_distance_to_object = np.linalg.norm(self.last_gripper_pose[:2] - self.last_object_pose[:2])
            current_distance_to_object = np.linalg.norm(gripper_pos[:2] - object_pos[:2])
            reward += last_distance_to_object - current_distance_to_object
        else:
            reward += 100  # High positive reward for reaching the goal

        return reward

    def _is_done(self):
        object_pos = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0][:2])
        distance_to_goal = np.linalg.norm(object_pos - self.goal_position)
        if bool(distance_to_goal < self.goal_radius):
            self.is_success = True
            return True
        return False

    def _is_truncated(self):
        gripper_pos = np.array(p.getBasePositionAndOrientation(self.simulation.robot_id)[0][:2])
        object_pos = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0][:2])
        
        gripper_out_of_canvas = not (self.canvas_min_x <= gripper_pos[0] <= self.canvas_max_x 
                                     and self.canvas_min_y <= gripper_pos[1] <= self.canvas_max_y)
        object_out_of_canvas = not (self.canvas_min_x <= object_pos[0] <= self.canvas_max_x 
                                    and self.canvas_min_y <= object_pos[1] <= self.canvas_max_y)
    
        time_ended = self.simulation.step_count >= 10000  # Maximum number of steps

        return bool(gripper_out_of_canvas or object_out_of_canvas or time_ended)

    def render(self):
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def close(self):
        p.disconnect()

# Example usage
if __name__ == "__main__":
    env = VPushPbSimulationEnv(gui=True) # TODO: include task and design parameters in the observation space
    check_env(env)

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for i in range(1000):
        # action, _states = model.predict(obs)
        action = env.action_space.sample()
        obs, rewards, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            obs = env.reset()
    env.close()
