import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pybullet as p
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from sim.scoop_sim import ScoopingSimulation

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class ScoopSimulationEnv(gym.Env):
    def __init__(self, render_mode='human', img_size=(42, 42), obs_type='pose'):
        super(ScoopSimulationEnv, self).__init__()
        self.task = 'pillow' # insole or pillow
        self.task_int = 0 if self.task == 'insole' else 1
        self.coef = [1,1]
        self.gui = True if render_mode == 'human' else False
        self.simulation = ScoopingSimulation(self.task, self.coef, self.gui)
        self.img_size = img_size  # New parameter for image size
        self.obs_type = obs_type  # New parameter for observation type
        self.action_space = spaces.Box(low=np.array([-1,]*2+[-0.2,]), high=np.array([1,]*2+[0.2,]), dtype=np.float32)
        self.canvas_min_x, self.canvas_max_x = 0, 6
        self.canvas_min_y, self.canvas_max_y = 0, 6

        if self.obs_type == 'image':
            # Observation space: smaller RGB image of the simulation
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.float64)
        else:
            # Observation space: low-dimensional pose (object pose, gripper pose)
            self.observation_space = spaces.Box(low=np.array(([0.,]*3 +[-1.,]*4)*2+[0.,]*3), high=np.array(([1.,]*3 +[1.,]*4)*2+[2.,]*3), dtype=np.float64)
        
        # Goal parameters
        # self.goal_radius = 0.5
        # self.goal_position = np.array([4.5, 2.5]) # workspace [[0,5], [0,5]]
        # self.simulation.goal_position = self.goal_position
        # self.simulation.goal_radius = self.goal_radius

        self.last_gripper_pose = None
        self.last_object_pose = None
        # self.last_action = None
        self.last_angle_difference = None
        self.current_dist_gripper_to_object = np.inf
        
        self.num_end_steps = 0
        self.is_success = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulation.step_count = 0
        self.task = random.choice(['insole', 'pillow'])
        self.task_int = 0 if self.task == 'insole' else 1
        self.coef = [random.uniform(.5, 2), 
                     random.uniform(0.2,1.3),]
        self.simulation.reset_task_and_design(self.task, self.coef)
        obs = self._get_obs()
        self.num_end_steps = 0
        self.is_success = False
        return obs, {}

    def step(self, action):
        self.last_object_pose = np.array(list(p.getBasePositionAndOrientation(self.simulation.object_id)[0][:2])
                                          + [pi_2_pi(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.object_id)[1])[2]),])
        self.last_gripper_pose = np.array(list(p.getBasePositionAndOrientation(self.simulation.robot_id)[0][:2])
                                          + [pi_2_pi(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.robot_id)[1])[2]),])

        # Set the new velocities
        new_linear_velocity = [float(a) for a in action[:2]]+[0,]
        new_angular_velocity = [0, 0, action[2]]
        
        p.resetBaseVelocity(self.simulation.robot_id, linearVelocity=new_linear_velocity, angularVelocity=new_angular_velocity)

        # Step the simulation (slow with deformable objects)
        sim_steps = 5 # 48Hz
        for _ in range(sim_steps):
            p.stepSimulation()

        # # Visualize the simulation
        # width, height, rgbPixels, _, _ = p.getCameraImage(128, 128, 
        #                                                     viewMatrix=self.simulation.viewMatrix, 
        #                                                     projectionMatrix=self.simulation.projectionMatrix)
        # # if self.gui:
        # #     time.sleep(self.simulation.time_step)
        # frame = np.reshape(rgbPixels, (height, width, 4))[:, :, :3]
        # self.frame = np.uint8(frame)

        self.simulation.step_count += 1
        obs = self._get_obs()
        done = self._is_done()
        reward = self._compute_reward(action)
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
            object_pose = np.array(list(p.getBasePositionAndOrientation(self.simulation.object_id)[0]) + 
                                   list(p.getBasePositionAndOrientation(self.simulation.object_id)[1]))
            gripper_pose = np.array(list(p.getBasePositionAndOrientation(self.simulation.robot_id)[0]) + 
                                   list(p.getBasePositionAndOrientation(self.simulation.robot_id)[1]))
            obs = np.concatenate([object_pose, gripper_pose])
            max_vals = np.array(([self.canvas_max_x,]*3+[1.0,]*4)*2) # Normalization constants
            obs_normalized = obs / max_vals

            # Concatenate with task and design parameters
            task_design_params_normalized = np.array([self.task_int, *self.coef])

            return np.concatenate([obs_normalized, task_design_params_normalized])

    def _compute_reward(self, action):
        object_position = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0])
        gripper_position = np.array(p.getBasePositionAndOrientation(self.simulation.robot_id)[0])
        self.current_dist_gripper_to_object = np.linalg.norm(gripper_position - object_position)

        # # Reward for aligning the scoop with the object
        reward = 0
        gripper_angle = pi_2_pi(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.robot_id)[1])[2])
        current_direction_angle = math.atan2(object_position[1] - gripper_position[1], 
                                             object_position[0] - gripper_position[0])
        current_angle_difference = abs(current_direction_angle - gripper_angle)
        
        if self.last_angle_difference is not None:
            reward = (self.last_angle_difference - current_angle_difference) * 5
        self.last_angle_difference = current_angle_difference
        
        # Reward for approaching the object
        last_dist_gripper_to_object = np.linalg.norm(self.last_gripper_pose[:2] - self.last_object_pose[:2])
        reward += last_dist_gripper_to_object - self.current_dist_gripper_to_object
        
        # TODO: Reward of caging robustness

        if self.is_success:
            reward += 100
        return reward

    def _is_done(self):
        # if self.simulation.reached_height < p.getBasePositionAndOrientation(self.simulation.object_id)[0][2]:
        if self.current_dist_gripper_to_object < 0.5:
            self.num_end_steps += 1
        if self.num_end_steps >= 100: # end if object stays in the basket for 100 steps
            self.is_success = True
            return True
        return False

    def _is_truncated(self):
        gripper_pos = np.array(p.getBasePositionAndOrientation(self.simulation.robot_id)[0])
        object_pos = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0])
        
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