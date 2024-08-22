from typing import Optional, Any, Dict, Optional, Tuple

import numpy as np
import time
import random
import pybullet as p
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
from panda_gym.envs.tasks.push import Push
from typing import Any, Dict, Optional, Tuple
from gymnasium.utils import seeding

from .panda_push_robot import PandaCustom
from .panda_push_task import VPush
import os
import cv2
import gc

class PandaUPushEnv(RobotTaskEnv):
    """Push task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    def __init__(
        self,
        render_mode: str = "human",
        obs_type: str = "pose",
        time_stamp: str = "",
        using_robustness_reward: bool = False,
        reward_weights: list = [],
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = PandaCustom(sim, block_gripper=True, base_position=np.array([0.0, 0.0, 0.0]), control_type=control_type, time_stamp=time_stamp)
        task = VPush(sim, reward_type=reward_type, using_robustness_reward=using_robustness_reward)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )
        self.step_count = 0
        self.canvas_min_x = 0.1
        self.canvas_max_x = 0.9
        self.canvas_min_y = -0.40
        self.canvas_max_y = 0.40
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:    
        self.step_count += 1
        self.robot.set_action(action)
        self.sim.step()
        observation = self._get_obs()
        terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        if terminated:
            print(f"is_success: {self.step_count}")
        info = {"is_success": terminated, "observation": observation["observation"]}
        truncated = self._is_truncated()
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), info))
        
        # reward = float(self.task.compute_reward(observation, info))
        # time.sleep(40./240.)
        # print( self.robot.get_ee_position())
        # print( self.robot.get_arm_joint_angles())

        # # pybullet take image
        # width, height, rgbPixels, _, _ = p.getCameraImage(256, 256, 
        #                                                     viewMatrix=[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
        #                                                     projectionMatrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
        # # save image to path
        # frame = np.reshape(rgbPixels, (height, width, 4))[:, :, :3]
        # self.frame = np.uint8(frame)
        # os.makedirs('images', exist_ok=True)
        # cv2.imwrite(f'images/{self.step_count}.png', self.frame)

        return observation, reward, terminated, truncated, info
        
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self.step_count = 0

        # Reset task and design parameters
        self.task.task_object_name = random.choice(self.task.task_object_names)
        self.task.task_int = 0 if self.task.task_object_name == 'circle' else 1
        self.robot.v_angle = random.uniform(np.pi/6, np.pi*5/6)
        self.robot.finger_length = random.uniform(0.07, 0.15)
        while True:
            self.robot.finger_angle = random.uniform(-np.pi/3, np.pi/3)
            self.robot.distal_phalanx_length = random.uniform(0.00, 0.10)
            if self._pusher_forward_kinematics()[1] > 0.02:
                break

        return super().reset(seed=seed, options=options)
    
    def _pusher_forward_kinematics(self):
        l1 = self.robot.finger_length
        l2 = self.robot.distal_phalanx_length
        a1 = self.robot.v_angle / 2
        a2 = self.robot.finger_angle
        x = l1 * np.cos(a1) + l2 * np.cos(a2)
        y = l1 * np.sin(a1) + l2 * np.sin(a2)
        return (x, y)
    
    def _is_truncated(self):
        truncated = False

        ee_position = self.robot.get_ee_position()
        object_position = self.task.get_achieved_goal()
        
        gripper_out_of_canvas = not (self.canvas_min_x <= ee_position[0] <= self.canvas_max_x 
                                     and self.canvas_min_y <= ee_position[1] <= self.canvas_max_y)
        object_out_of_canvas = not (self.canvas_min_x <= object_position[0] <= self.canvas_max_x 
                                    and self.canvas_min_y <= object_position[1] <= self.canvas_max_y)
        time_ended = self.step_count > 500
    
        truncated = (gripper_out_of_canvas or object_out_of_canvas or time_ended)
        if truncated:
            print(f"step count: {self.step_count}")
            print(f"gripper: {ee_position}")
            print(f"object: {object_position}")
        return truncated
