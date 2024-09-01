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
from sim.vpush_pb_sim import VPushPbSimulation
import time
from stable_baselines3.common.utils import get_linear_fn

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class VPushPbSimulationEnv(gym.Env):
    def __init__(self, 
                 render_mode: str = "human",
                 obs_type: str = "pose",
                 using_robustness_reward: bool = False, 
                 img_size=(42, 42), 
                 run_id: str = "default",
                 reward_weights: list = [5.0, 1.0, 1.0, 1.0, 100.0, 0.0, 0.0, 0.0],
                 reward_type: str = "dense", # dense, sparse
                 perturb: bool = False,
                 perturb_sigma: float = 1.8,
        ):
        super(VPushPbSimulationEnv, self).__init__()
        self.task = 'circle' 
        self.task_int = 0 if self.task == 'circle' else 1
        self.v_angle = np.pi/3
        self.gui = True if render_mode == 'human' else False
        self.simulation = VPushPbSimulation(self.task, self.v_angle, self.gui)
        self.img_size = img_size  # New parameter for image size
        self.obs_type = obs_type  # New parameter for observation type
        self.using_robustness_reward = using_robustness_reward
        self.reward_weights = reward_weights
        self.perturb = perturb
        self.perturb_sigma = perturb_sigma
        self.simulation.run_id = run_id
        self.action_space = spaces.Box(low=np.array([-1, -1, -0.4]), high=np.array([1, 1, 0.4]), dtype=np.float32)
        self.canvas_min_x, self.canvas_max_x = 0, 5
        self.canvas_min_y, self.canvas_max_y = 0, 5

        if self.obs_type == 'image':
            # Observation space: smaller RGB image of the simulation
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.float64)
        else:
            # Observation space: low-dimensional pose (object pose, gripper pose)
            self.observation_space = spaces.Dict(
                dict(
                    observation=spaces.Box(low=np.array([0., 0., -1.]*2 +[0.,]*4), high=np.array([1.0]*10), dtype=np.float64),
                    desired_goal=spaces.Box(0.0, 5.0, shape=(2,), dtype=np.float32),
                    achieved_goal=spaces.Box(0.0, 5.0, shape=(2,), dtype=np.float32),
                )
            )
        # Goal parameters
        self.goal_radius = 0.4
        self.goal_position = np.array([4.5, 2.5]) # workspace [[0,5], [0,5]]
        self.goal_position += np.array([random.uniform(-0.2, 0.1), random.uniform(-1.0, 1.0)])
        self.simulation.goal_position = self.goal_position
        self.simulation.goal_radius = self.goal_radius

        self.last_gripper_pose = None
        self.last_object_pose = None
        self.last_action = None
        self.last_angle_difference = None
        self.last_angle_difference_approach = None
        
        self.is_success = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulation.step_count = 0
        self.task = random.choice(['circle', 'polygon'])
        self.task_int = 0 if self.task == 'circle' else 1
        self.v_angle = random.uniform(np.pi/12, np.pi*11/12)
        self.goal_position = np.array([4.5, 2.5]) # workspace [[0,5], [0,5]]
        self.goal_position += np.array([random.uniform(-0.2, 0.1), random.uniform(-0.3, 0.3)])
        self.simulation.goal_position = self.goal_position
        self.simulation.reset_task_and_design(self.task, self.v_angle)
        obs = self._get_obs()

        self.is_success = False
        return obs, {}

    def reset_task_and_design(self, task_int, v_angle, seed=None, options=None):
        super().reset(seed=seed)
        self.simulation.step_count = 0
        self.task_int = task_int
        self.task = 'circle' if self.task_int == 0 else 'polygon'
        self.v_angle = v_angle[0]
        self.simulation.reset_task_and_design(self.task, self.v_angle)
        obs = self._get_obs()

        self.is_success = False
        return obs, {}
    
    def step(self, action):
        self.last_object_pose = np.array(list(p.getBasePositionAndOrientation(self.simulation.object_id)[0][:2])
                                          + [pi_2_pi(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.object_id)[1])[2]),])
        self.last_gripper_pose = np.array(list(p.getBasePositionAndOrientation(self.simulation.robot_id)[0][:2])
                                          + [pi_2_pi(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.robot_id)[1])[2]),])

        # Set the new velocities
        new_linear_velocity = [float(action[0]), float(action[1]), 0]
        new_angular_velocity = float(action[2])
        
        p.resetBaseVelocity(self.simulation.robot_id, linearVelocity=new_linear_velocity, angularVelocity=[0, 0, new_angular_velocity])

        # Step the simulation
        sim_steps = 5 # 48Hz
        for _ in range(sim_steps):
            # add random perturbation force to the target object
            if self.perturb:
                p.applyExternalForce(self.simulation.object_id, -1, [random.normalvariate(0, 0.5), random.normalvariate(0, 0.5), 0], [0, 0, 0], p.LINK_FRAME)
            p.stepSimulation()

        # width, height, rgbPixels, _, _ = p.getCameraImage(64, 64, 
        #                                                     viewMatrix=self.simulation.viewMatrix, 
        #                                                     projectionMatrix=self.simulation.projectionMatrix)
        # if self.gui:
        #     time.sleep(1./240.)

        # frame = np.reshape(rgbPixels, (height, width, 4))[:, :, :3]
        # self.frame = np.uint8(frame)

        self.simulation.step_count += 1
        obs = self._get_obs()
        done = self._is_done()
        reward = self._compute_reward(action)
        truncated = self._is_truncated()
        self.last_action = action
        
        info = {}
        info['robustness'] = self.robustness
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

            # return np.concatenate([obs_normalized, task_design_params_normalized])
            return {"observation": np.concatenate([obs_normalized, task_design_params_normalized]),
                    "achieved_goal": np.array(object_pose[:2]).astype(np.float32),
                    "desired_goal": np.array(self.goal_position).astype(np.float32),} 
        
    def _compute_reward(self, action):
        object_position = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0][:2])
        gripper_position = np.array(p.getBasePositionAndOrientation(self.simulation.robot_id)[0][:2])
        current_dist_gripper_to_object = np.linalg.norm(gripper_position - object_position)
        current_dist_object_to_goal = np.linalg.norm(object_position - self.goal_position)
        self.robustness = self.simulation.eval_robustness(slack=self.simulation.object_rad)
        condition_approached = bool(self.robustness>0)

        # Reward for aligning the gripper with the goal
        reward = 0
        gripper_angle = pi_2_pi(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.robot_id)[1])[2])
        current_direction_angle = math.atan2(self.goal_position[1] - object_position[1], self.goal_position[0] - object_position[0])
        current_angle_difference = abs(current_direction_angle - gripper_angle)
        
        if self.last_angle_difference is not None:
            reward = (self.last_angle_difference - current_angle_difference) * self.reward_weights[0]
        self.last_angle_difference = current_angle_difference
        
        # Reward for approaching the object and the goal
        if not condition_approached:
            last_dist_gripper_to_object = np.linalg.norm(self.last_gripper_pose[:2] - self.last_object_pose[:2])
            reward += (last_dist_gripper_to_object - current_dist_gripper_to_object) * self.reward_weights[1]
        else:
            last_dist_object_to_goal = np.linalg.norm(self.last_object_pose[:2] - self.goal_position)
            reward += (last_dist_object_to_goal - current_dist_object_to_goal) * self.reward_weights[2]
        
        # Reward of caging robustness
        if self.using_robustness_reward:
            reward += self.robustness * self.reward_weights[3]

        if self.is_success:
            reward += self.reward_weights[4]
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