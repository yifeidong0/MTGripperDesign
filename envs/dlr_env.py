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
    def __init__(self, 
                 render_mode: str = "human",
                 obs_type: str = "pose",
                 using_robustness_reward: bool = False, 
                 img_size=(42, 42), 
        ):
        super(DLRSimulationEnv, self).__init__()
        self.task = 'fish' # fish
        self.task_int = 0 if self.task == 'fish' else 1
        self.design_params = [1,1]        
        self.gui = True if render_mode == 'human' else False
        self.img_size = img_size
        self.obs_type = obs_type
        self.using_robustness_reward = using_robustness_reward
        self.action_space = spaces.Box(low=np.array([-0.0001,-0.0001,-0.001,-0.0003,-0.001,-0.003,]), 
                                       high=np.array([0.0001,0.0001,0.001,0.0003,0.001,0.003,]), 
                                       dtype=np.float32) # x, y, z, rot_z, a02, a13
        self.simulation = DLRSimulation(self.task, self.design_params, self.gui)

        if self.obs_type == 'image':
            # Observation space: smaller RGB image of the simulation
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.float64)
        else:
            # Observation space: low-dimensional pose. TODO: adjust boundaries
            # (object base pose[3+4], joint angles[9]; gripper height[1], rot_z[1], joint angles[4])
            self.observation_space = spaces.Box(low=np.array([-1,-1,0]+[-1,]*4+[-0.5,]*9+[0,-1]+[0,]*4+[0,0,0]), # TODO: fish joint angle limits
                                                high=np.array([1,1,1]+[1,]*4+[0.5,]*9+[1,1]+[1,]*4+[1,1,1]), 
                                                dtype=np.float64)
        self.robot_joint_limits = [[-0.1,0.2], [0,1.047],]
        self.robot_base_limits = [[-0.5,0.5], [-0.5,0.5], [2.0,4.5]]

        self.last_object_position = None
        self.last_robot_position = None
        self.num_end_steps = 0
        self.is_success = False
        self.count_episodes = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the simulation environment to avoid memory leak (Pybullet bug)
        if (self.count_episodes+1) % 10 == 0:
            self.close()
            self.simulation = DLRSimulation(self.task, self.design_params, self.gui)

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
        self.count_episodes += 1
        self.is_success = False
        return obs, {}

    def step(self, action):
        dxyz = action[:3]
        drot_z = action[4]
        da = action[4:]

        # Step the simulation
        sim_steps = 5 # 48Hz
        for _ in range(sim_steps):
            p.stepSimulation()

            # Reset joint angles
            for i in range(2):
                joint_position = p.getJointState(self.simulation.robot_id, i)[0]
                new_joint_position = np.clip(joint_position+da[i], self.robot_joint_limits[i][0], self.robot_joint_limits[i][1])
                p.resetJointState(self.simulation.robot_id, i, new_joint_position)
                p.resetJointState(self.simulation.robot_id, i+2, new_joint_position)
                
            # Reset gripper base
            gripper_position, gripper_orientation = p.getBasePositionAndOrientation(self.simulation.robot_id)
            gripper_orientation = p.getEulerFromQuaternion(gripper_orientation)
            new_gripper_position = np.zeros(3)
            for i in range(3):
                new_gripper_position[i] = np.clip(gripper_position[i]+dxyz[i], self.robot_base_limits[i][0], self.robot_base_limits[i][1])
            # new_gripper_position_z = np.clip(gripper_position[2]+dz, self.robot_base_z_limits[0], self.robot_base_z_limits[1])
            p.resetBasePositionAndOrientation(self.simulation.robot_id,
                                              new_gripper_position,
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
        # Rotate the gripper to align with the object
        reward = 0
        object_position = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0])
        gripper_position = np.array(p.getBasePositionAndOrientation(self.simulation.robot_id)[0])
        object_orientation = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.object_id)[1])
        gripper_orientation = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.robot_id)[1])
        reward += -0.05 * abs(object_orientation[2]-gripper_orientation[2])/np.pi

        # Approach the object: fish and gripper tip are close
        result = p.getClosestPoints(self.simulation.robot_id, self.simulation.object_id, 10.0, 3, 5)
        if len(result) > 0:
            position_on_robot, position_on_object = result[0][5:7]
            closest_distance = np.linalg.norm(np.array(position_on_robot) - np.array(position_on_object))
            if np.random.rand() < 0.002:
                print("Closest distance: ", closest_distance)
                print("height diff: ", position_on_robot[2] - position_on_object[2])
            reward += 0.05 * (1-closest_distance/3)

        # Penalize the gripper penetrating the floor
        result1 = p.getClosestPoints(self.simulation.robot_id, self.simulation.object_id, -0.02, 3, 5)
        result2 = p.getClosestPoints(self.simulation.robot_id, self.simulation.plane_id, -0.02, 3, -1)
        if len(result1) > 0:
            # print("Penetrating the fish!")
            reward -= 0.5
        if len(result2) > 0:
            # print("Penetrating the floor!")
            reward -= 0.5

        if len(result) > 0:
            # Place tips underneath the object
            if position_on_robot[2] > position_on_object[2]:
                reward -= 0.01 * (position_on_robot[2]-position_on_object[2])
            else:
                reward += 0.05 * (position_on_object[2]-position_on_robot[2])

            # # Close fingers and tips
            # gripper_joint_angles = np.array([p.getJointState(self.simulation.robot_id, i)[0] for i in range(4)])
            # if position_on_robot[2] > position_on_object[2]: # not yet scooped
            #     reward -= 0.01 * gripper_joint_angles-self.robot_joint_limits[0][0]
            # else: # scooped

            # Lift up the object
            if self.last_object_position is not None:
                reward += 10 * (position_on_object[2]-self.last_object_position[2])
            if self.last_robot_position is not None and (position_on_robot[2] < position_on_object[2]):
                reward += (position_on_robot[2]-self.last_robot_position[2])
            self.last_object_position = position_on_object
            self.last_robot_position = position_on_robot
        
        # TODO: Reward of caging robustness
        if self.using_robustness_reward:
            pass

        if self.is_success:
            reward += 100
        return reward

    def _is_done(self):
        object_position = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0])
        if object_position[2] > 0.6:
            self.num_end_steps += 1
        if self.num_end_steps >= 100:
            self.is_success = True
            return True

    def _is_truncated(self):
        gripper_position = np.array(p.getBasePositionAndOrientation(self.simulation.robot_id)[0])
        object_position = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0])
        
        gripper_out_of_canvas = bool(4.5 < gripper_position[2])
        object_out_of_canvas = not (-2 <= object_position[0] <= 2
                                    and -2 <= object_position[1] <= 2)
    
        time_ended = self.simulation.step_count >= 30000  # Maximum number of steps

        return bool(gripper_out_of_canvas or object_out_of_canvas or time_ended)
        # return False

    def render(self):
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def close(self):
        p.disconnect()