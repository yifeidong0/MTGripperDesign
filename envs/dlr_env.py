import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pybullet as p
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from sim.dlr_sim import DLRSimulation

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class DLRSimulationEnv(gym.Env):
    def __init__(self, gui=1, img_size=(42, 42), obs_type='pose'):
        super(DLRSimulationEnv, self).__init__()
        self.task = 'fish' # fish
        self.task_int = 0 if self.task == 'fish' else 1
        self.design_params = [1,1]
        self.simulation = DLRSimulation(self.task, self.design_params, gui)
        self.gui = gui
        self.img_size = img_size
        self.obs_type = obs_type
        self.action_space = spaces.Box(low=np.array([-0.001,]*6), 
                                       high=np.array([0.001,]*6), 
                                       dtype=np.float32) # z, rot_z, a0, ..., a3
        self.canvas_min_x, self.canvas_max_x = 0, 6
        self.canvas_min_y, self.canvas_max_y = 0, 6

        if self.obs_type == 'image':
            # Observation space: smaller RGB image of the simulation
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.float64)
        else:
            # Observation space: low-dimensional pose. TODO: adjust boundaries
            # (object base pose[3+4], joint angles[9]; gripper height[1], rot_z[1], joint angles[4])
            self.observation_space = spaces.Box(low=np.array([-1,-1,0]+[-1,]*4+[-0.5,]*9+[0,-1]+[0,]*4+[0,0,0]), # TODO: fish joint angle limits
                                                high=np.array([1,1,1]+[1,]*4+[0.5,]*9+[1,1]+[1,]*4+[1,1,1]), 
                                                dtype=np.float64)

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
        self.task = 'fish' # random.choice(['fish',])
        self.task_int = 0 if self.task == 'fish' else 1
        base_lengths = np.arange(60, 150, 5)
        distal_lengths = np.arange(20, 60, 5)
        self.design_params = [random.choice(base_lengths), 
                              random.choice(distal_lengths),]
        self.simulation.reset_task_and_design(self.task, self.design_params)
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
        dz, drot_z, da0, da1, da2, da3 = action
        da = action[2:]

        # Step the simulation (slow with deformable objects)
        sim_steps = 5 # 48Hz
        for _ in range(sim_steps):
            p.stepSimulation()

            # Reset joint angles
            for i in range(4):
                jointPosition = p.getJointState(self.simulation.robot_id, i)[0]
                p.resetJointState(self.simulation.robot_id, i, jointPosition+da[i])
                
            # Reset gripper base
            gripper_position, gripper_orientation = p.getBasePositionAndOrientation(self.simulation.robot_id)
            gripper_orientation = p.getEulerFromQuaternion(gripper_orientation)
            p.resetBasePositionAndOrientation(self.simulation.robot_id,
                                              [gripper_position[0], gripper_position[1], gripper_position[2]+dz],
                                              p.getQuaternionFromEuler([math.pi, 0, pi_2_pi(gripper_orientation[2]+drot_z)]))

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
        if self.obs_type == 'image':
            width, height, rgb, _, _ = p.getCameraImage(width=self.img_size[0], height=self.img_size[1])
            rgb = np.array(rgb).reshape(self.img_size[1], self.img_size[0], 4)
            rgb = rgb[:, :, :3]  # Discard the alpha channel
            rgb = rgb / 255.0  # Normalize image observation
            return rgb
        else:
            # (object base pose[3+4], joint angles[9]; gripper height[1], rot_z[1], joint angles[4])
            object_pose = np.array(list(p.getBasePositionAndOrientation(self.simulation.object_id)[0]) + 
                                   list(p.getBasePositionAndOrientation(self.simulation.object_id)[1]))
            object_joint_angles = np.array([p.getJointState(self.simulation.object_id, i)[0] for i in range(9)])
            gripper_pose = np.array([list(p.getBasePositionAndOrientation(self.simulation.robot_id)[0])[2],] + # z
                                    [list(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.robot_id)[1]))[2],]) # rot_z
            gripper_joint_angles = np.array([p.getJointState(self.simulation.robot_id, i)[0] for i in range(4)])
            obs = np.concatenate([object_pose, object_joint_angles, gripper_pose, gripper_joint_angles])
            max_vals = np.array([3,3,1]+[1,]*4+[1,]*9+[5,np.pi]+[1,]*4) # Normalization constants
            obs_normalized = obs / max_vals

            # Concatenate with task and design parameters
            task_design_params = np.array([self.task_int, *self.design_params])
            task_design_params_normalized = task_design_params / np.array([1,150,60])

            return np.concatenate([obs_normalized, task_design_params_normalized])

    def _compute_reward(self, action):
        # object_position = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0])
        # gripper_position = np.array(p.getBasePositionAndOrientation(self.simulation.robot_id)[0])
        # self.current_dist_gripper_to_object = np.linalg.norm(gripper_position - object_position)

        # Reward for aligning the scoop with the object
        reward = 0
        # gripper_angle = pi_2_pi(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.robot_id)[1])[2])
        # current_direction_angle = math.atan2(object_position[1] - gripper_position[1], 
        #                                      object_position[0] - gripper_position[0])
        # current_angle_difference = abs(current_direction_angle - gripper_angle)
        
        # if self.last_angle_difference is not None:
        #     reward = (self.last_angle_difference - current_angle_difference) * 5
        # self.last_angle_difference = current_angle_difference
        
        # # Reward for approaching the object
        # last_dist_gripper_to_object = np.linalg.norm(self.last_gripper_pose[:2] - self.last_object_pose[:2])
        # reward += last_dist_gripper_to_object - self.current_dist_gripper_to_object
        
        # # TODO: Reward of caging robustness

        if self.is_success:
            reward += 100
        return reward

    def _is_done(self):
        object_position = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0])
        if object_position[2] > 0.5:
            self.num_end_steps += 1
        if self.num_end_steps >= 100:
            self.is_success = True
            return True

    def _is_truncated(self):
        gripper_position = np.array(p.getBasePositionAndOrientation(self.simulation.robot_id)[0])
        object_position = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0])
        
        gripper_out_of_canvas = bool(5 < gripper_position[2])
        object_out_of_canvas = not (-3 <= object_position[0] <= 3
                                    and -3 <= object_position[1] <= 3)
    
        time_ended = self.simulation.step_count >= 10000  # Maximum number of steps

        return bool(gripper_out_of_canvas or object_out_of_canvas or time_ended)
        # return False

    def render(self):
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def close(self):
        p.disconnect()