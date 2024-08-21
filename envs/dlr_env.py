import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any, Dict

import pybullet as p
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from sim.dlr_sim import DLRSimulation
from panda_gym.utils import distance

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class DLRSimulationEnv(gym.Env):
    def __init__(self, 
                 render_mode: str = "human",
                 obs_type: str = "pose",
                 using_robustness_reward: bool = False, 
                 reward_type: str = "none", # dense, sparse, none
                 img_size=(42, 42), 
        ):
        super(DLRSimulationEnv, self).__init__()
        self.task = 'cube' # fish, cube
        self.task_param = np.random.uniform(0.1, 0.2)
        self.design_params = [1,1]        
        self.gui = True if render_mode == 'human' else False
        self.img_size = img_size
        self.obs_type = obs_type
        self.reward_type = reward_type
        self.using_robustness_reward = using_robustness_reward
        self.action_space = spaces.Box(low=np.array([-1e-5,-1e-5,-0.0005,-1e-5,-0.001,-0.001,]), 
                                       high=np.array([1e-5,1e-5,0.0005,1e-5,0.001,0.001,]), 
                                       dtype=np.float32) # x, y, z, rot_z, a02, a13
        self.simulation = DLRSimulation(self.task, self.task_param, self.design_params, self.gui)

        if self.obs_type == 'image':
            # Observation space: smaller RGB image of the simulation
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.float64)
        else:
            # Observation space: low-dimensional pose. TODO: adjust boundaries
            # (object base pose[3+4], joint angles[9]; gripper height[1], rot_z[1], joint angles[4], task and design parameters[3])
            self.observation_space = spaces.Dict(
                dict(
                    # observation=spaces.Box(low=np.array([-1,-1,0]+[-1,]*4+[-0.5,]*9+[0,-1]+[-1,]*4+[0,0,0]), # TODO: fish joint angle limits
                    #                             high=np.array([1,1,1]+[1,]*4+[0.5,]*9+[1,1]+[1,]*4+[1,1,1]), 
                    #                             dtype=np.float64),
                    observation=spaces.Box(low=np.array([-1,-1,0]+[-1,]*4+[0,-1]+[-1,]*4+[0,0,0]),
                                                high=np.array([1,1,1]+[1,]*4+[1,1]+[1.2]*4+[1,1,1]), 
                                                dtype=np.float64),
                    desired_goal=spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                    achieved_goal=spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                )
            )
        self.robot_joint_limits = [[-0.3,0.3], [0.0,1.2],]
        self.robot_base_limits = [[-0.5,0.5], [-0.5,0.5], [1.5,4]]
        self.desired_goal_height = 0.6

        self.last_object_position = None
        self.last_tip_object_distance = None
        self.last_base_object_distance = None
        self.last_object_tip_height = None
        self.last_tips_distance = None
        self.count_penetration = 0
        self.num_end_steps = 0
        self.is_success = False
        self.count_episodes = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the simulation environment to avoid memory leak (Pybullet bug)
        if (self.count_episodes+1) % 9 == 0:
            self.close()
            self.simulation = DLRSimulation(self.task, self.task_param, self.design_params, self.gui)
            print(f"INFO: episode {self.count_episodes}")

        self.simulation.step_count = 0
        self.task = 'cube' # cube, fish
        self.task_param = np.random.uniform(0.08, 0.14)
        # base_lengths = np.arange(60, 100, 5)
        distal_lengths = np.arange(20, 60, 5)
        # distal_curvature = np.arange(1, 5, 1) # TODO: add curvature as design parameter
        self.design_params = [60, # random.choice(base_lengths), 
                              random.choice(distal_lengths),]
        # self.design_params = [60,30]
        self.simulation.reset_task_and_design(self.task, self.task_param, self.design_params)
        obs = self._get_obs()
        self.num_end_steps = 0
        self.count_penetration = 0
        self.count_episodes += 1
        self.is_success = False
        return obs, {}

    def step(self, action):
        dxyz = action[:3]
        drot_z = action[3]
        da = action[4:]

        # Step the simulation
        sim_steps = 10 # 48Hz
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
        truncated = self._is_truncated()
        info = {}
        if done or truncated:
            info['is_success'] = self.is_success
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))
        self.last_action = action
        
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
            # object_joint_angles = np.array([p.getJointState(self.simulation.object_id, i)[0] for i in range(9)])
            gripper_pose = np.array([list(p.getBasePositionAndOrientation(self.simulation.robot_id)[0])[2],] + # z
                                    [list(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.simulation.robot_id)[1]))[2],]) # rot_z
            gripper_joint_angles = np.array([p.getJointState(self.simulation.robot_id, i)[0] for i in range(4)])
            # obs = np.concatenate([object_pose, object_joint_angles, gripper_pose, gripper_joint_angles])
            obs = np.concatenate([object_pose, gripper_pose, gripper_joint_angles])
            # max_vals = np.array([3,3,1]+[1,]*4+[1,]*9+[5,np.pi]+[1,]*4) # Normalization constants
            max_vals = np.array([3,3,1]+[1,]*4+[5,np.pi]+[1,]*4) # Normalization constants
            obs_normalized = obs / max_vals

            # Concatenate with task and design parameters
            task_design_params = np.array([self.task_param, *self.design_params])
            task_design_params_normalized = task_design_params / np.array([1.0,150,60])

            # return np.concatenate([obs_normalized, task_design_params_normalized])
            return {"observation": np.concatenate([obs_normalized, task_design_params_normalized]),
                    "achieved_goal": np.array([object_pose[2]]).astype(np.float32),
                    "desired_goal": np.array([self.desired_goal_height]).astype(np.float32),} 

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        # Rotate the gripper to align with the object
        if self.reward_type == 'sparse':
            d = distance(achieved_goal, desired_goal)
            return -np.array(d > 0.01, dtype=np.float32)
        elif self.reward_type == 'dense':
            d = distance(achieved_goal, desired_goal)
            return -d.astype(np.float32)
        else:
            reward = 0
            reward1, reward2, reward3, reward4, reward5, reward6, reward_robustness = None, None, None, None, None, None, None

            # Approach the object: object and gripper tips/base are close
            result_approach_left = p.getClosestPoints(self.simulation.robot_id, self.simulation.object_id, 100.0, 3, -1)
            result_approach_right = p.getClosestPoints(self.simulation.robot_id, self.simulation.object_id, 100.0, 1, -1)
            result_approach_base = p.getClosestPoints(self.simulation.robot_id, self.simulation.object_id, 100.0, -1, -1)
            if len(result_approach_left)>0 and len(result_approach_right)>0 and len(result_approach_base)>0:
                position_on_robot, position_on_object = result_approach_left[0][5:7]
                tip_object_distance_left = np.linalg.norm(np.array(position_on_robot) - np.array(position_on_object))
                position_on_robot, position_on_object = result_approach_right[0][5:7]
                tip_object_distance_right = np.linalg.norm(np.array(position_on_robot) - np.array(position_on_object))
                position_on_robot, position_on_object = result_approach_base[0][5:7]
                base_object_distance = np.linalg.norm(np.array(position_on_robot) - np.array(position_on_object))

                # object-tip relative distance
                tip_object_distance = (tip_object_distance_left+tip_object_distance_right) / 2
                if self.last_tip_object_distance is not None:
                    reward1 = 0.1 * (self.last_tip_object_distance - tip_object_distance)
                    reward += reward1
                self.last_tip_object_distance = tip_object_distance

                # object and gripper base relative distance
                if self.last_base_object_distance is not None:
                    reward2 = 0.001 * (self.last_base_object_distance - base_object_distance)
                    reward += reward2
                self.last_base_object_distance = base_object_distance

            # Penalize the gripper penetrating the floor or object
            result_penalize_object_tip_left = p.getClosestPoints(self.simulation.robot_id, self.simulation.object_id, -0.01, 3, -1)
            result_penalize_object_tip_right = p.getClosestPoints(self.simulation.robot_id, self.simulation.object_id, -0.01, 1, -1)
            result_penalize_floor_tip_right = p.getClosestPoints(self.simulation.robot_id, self.simulation.plane_id, -0.02, 3, -1)
            result_penalize_floor_tip_left = p.getClosestPoints(self.simulation.robot_id, self.simulation.plane_id, -0.02, 1, -1)
            result_penalize_floor_finger_left = p.getClosestPoints(self.simulation.robot_id, self.simulation.plane_id, -0.01, 2, -1)
            result_penalize_floor_finger_right = p.getClosestPoints(self.simulation.robot_id, self.simulation.plane_id, -0.01, 0, -1)
            result_penalize_tips = p.getClosestPoints(self.simulation.robot_id, self.simulation.robot_id, -0.02, 3, 1)
            reward3 = -0.03 * (2*(len(result_penalize_object_tip_left)>0) + 2*(len(result_penalize_object_tip_right)>0) + 
                             (len(result_penalize_floor_tip_left)>0) + (len(result_penalize_floor_tip_right)>0) + 
                             10*(len(result_penalize_floor_finger_left)>0) + 10*(len(result_penalize_floor_finger_right)>0) +
                             3*(len(result_penalize_tips)>0))
            if reward3 < 0:
                self.count_penetration += 1
            reward += reward3
            # if self.count_penetration > 100:
            #     reward += -5.0

            # Reward for making gripper tips lower than the object
            result_floor_tip_left = p.getClosestPoints(self.simulation.robot_id, self.simulation.plane_id, 100.0, 3, -1)
            result_floor_tip_right = p.getClosestPoints(self.simulation.robot_id, self.simulation.plane_id, 100.0, 1, -1)
            if len(result_floor_tip_left) > 0 and len(result_floor_tip_right) > 0:
                position_on_robot, position_on_object = result_floor_tip_left[0][5:7]
                tip_floor_distance_left = np.linalg.norm(np.array(position_on_robot) - np.array(position_on_object))
                position_on_robot, position_on_object = result_floor_tip_right[0][5:7]
                tip_floor_distance_right = np.linalg.norm(np.array(position_on_robot) - np.array(position_on_object))
                tip_floor_distance = (tip_floor_distance_left+tip_floor_distance_right) / 2

                # object-tip relative height
                object_position = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0])
                object_tip_height = object_position[2] - tip_floor_distance
                if self.last_object_tip_height is not None:
                    reward4 = 0.1 * (object_tip_height - self.last_object_tip_height)
                    reward += reward4
                self.last_object_tip_height = object_tip_height

            # Reward for minimizing the distance between the gripper tips
            if self.last_object_tip_height is not None:
                result_tips = p.getClosestPoints(self.simulation.robot_id, self.simulation.robot_id, 100.0, 3, 1)
                if len(result_tips) > 0:
                    position_on_robot1, position_on_robot2 = result_tips[0][5:7]
                    tips_distance = np.linalg.norm(np.array(position_on_robot1) - np.array(position_on_robot2))
                    if self.last_tips_distance is not None:
                        if self.last_object_tip_height>0:
                            reward5 = 10.0 * (self.last_tips_distance - tips_distance)
                        else:
                            reward5 = 10.0 * (tips_distance - self.last_tips_distance)
                        reward += reward5
                    self.last_tips_distance = tips_distance

            # Reward for lifting the object
            if self.last_object_position is not None:
                reward6 = 50.0 * (object_position[2] - self.last_object_position[2])
                reward += reward6
            self.last_object_position = object_position

            # Reward of caging robustness
            if self.using_robustness_reward and object_position[2]>0.5*self.desired_goal_height and np.random.rand() < 0.004:
                reward_robustness = self.simulation.eval_robustness(0.5*object_position[2]) / 200
                if reward_robustness is not None:
                    print("!!!!!!!!!!!!reward robustness", round(reward_robustness, 3))
                reward += reward_robustness

            # Reward of success and caging robustness
            if self.is_success:
                reward += 100.0

            # Debugging, show rewards with 2 digits after the decimal point
            if np.random.rand() < 3e-4:
                if reward1 is not None:
                    print("reward object-tip distance", round(reward1, 3))
                if reward2 is not None:
                    print("reward object-gripper-base distance", round(reward2, 3))
                if reward3 is not None:
                    print("reward penetration", round(reward3, 3))
                if reward4 is not None:
                    print("reward object-tip relative height", round(reward4, 3))
                if reward5 is not None:
                    print("reward gripper-tip distance", round(reward5, 3))
                if reward6 is not None:
                    print("reward lift", round(reward6, 3))
                print("")


            return reward * np.ones(desired_goal.shape).squeeze()
        
    def _is_done(self):
        object_position = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0])
        if object_position[2] > self.desired_goal_height:
            self.num_end_steps += 1
        if self.num_end_steps >= 100:
            self.is_success = True
            return True
        return False

    def _is_truncated(self):
        gripper_position = np.array(p.getBasePositionAndOrientation(self.simulation.robot_id)[0])
        object_position = np.array(p.getBasePositionAndOrientation(self.simulation.object_id)[0])
        
        gripper_out_of_canvas = bool(4 < gripper_position[2])
        object_out_of_canvas = not (-1 <= object_position[0] <= 1
                                    and -1 <= object_position[1] <= 1)
    
        time_ended = self.simulation.step_count >= 8000  # Maximum number of steps
        too_many_penetrations = self.count_penetration > 100
        if too_many_penetrations:
            print("INFO: Penetration count exceeded 100!")

        return bool(gripper_out_of_canvas or object_out_of_canvas or time_ended or too_many_penetrations)
        # return False

    def render(self):
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def close(self):
        p.disconnect()