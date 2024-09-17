# import os
# import sys
# # add parent parent directory to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Any, Dict, Optional, Tuple
import time

import numpy as np
import random
import pybullet as p
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet

from .panda_push_robot import PandaCustom
from .panda_push_task import UPush
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
        run_id: str = "default",
        using_robustness_reward: bool = True,
        reward_weights: list = [1.0, 0.01, 1.0, 1.0, 100.0, 0.0, 0.0, 0.0],
        perturb: bool = False,
        perturb_sigma: float = 1.8,
        reward_type: str = "dense",
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
        sim = PyBullet(render_mode=render_mode, renderer=renderer, n_substeps=50)
        robot = PandaCustom(sim, block_gripper=True, base_position=np.array([0.0, 0.0, 0.0]), control_type=control_type, run_id=run_id)
        print("reward_weights", reward_weights)
        task = UPush(sim, reward_type=reward_type, using_robustness_reward=using_robustness_reward, reward_weights=reward_weights)
        task.ee_init_pos_2d = robot.ee_init_pos_2d
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
        self.perturb = perturb
        self.perturb_sigma = perturb_sigma
        self.obs_type = obs_type
        self.reward_weights = reward_weights
        self.step_count = 0
        self.canvas_min_x = 0.20
        self.canvas_max_x = 0.75
        self.canvas_min_y = -0.4
        self.canvas_max_y = 0.4
        self.is_safe = True
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:    
        self.step_count += 1
        self.robot.set_action(action)
        if self.perturb:
            p.applyExternalForce(self.sim._bodies_idx["object"], -1, [random.normalvariate(0, self.perturb_sigma), random.normalvariate(0, self.perturb_sigma), 0], [0, 0, 0], p.LINK_FRAME)
        self.sim.step()
        observation = self._get_obs()
        terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        # terminated = self.task.is_success_flag
        info = {"is_success": terminated}
        truncated = self._is_truncated()
        self.task.is_safe = self.is_safe
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), observation["observation"]))
        info['robustness'] = self.task.robustness_score

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
        self.is_safe = True

        # Reset task and design parameters
        self.task.task_object_name = random.choice(self.task.task_object_names)
        self.task.task_int = self.task.task_ints[self.task.task_object_name]
        self.robot.v_angle = random.uniform(np.pi/6, np.pi*5/6)
        self.robot.finger_length = random.uniform(0.05, 0.12)
        while True:
            self.robot.finger_angle = random.uniform(-np.pi/3, np.pi/3)
            self.robot.distal_phalanx_length = random.uniform(0.00, 0.08)
            if self._pusher_forward_kinematics()[1] > 0.02: # make sure the pusher does not self-intersect
                break

        # self.robot.v_angle = 0.5235987755982988 # iter3,0.16, red, second-printed upusher
        # self.robot.finger_length = 0.05
        # self.robot.finger_angle = -0.11635528346628865
        # self.robot.distal_phalanx_length = 0.053333333333

        # self.robot.v_angle = 0.936 # test, white, first-printed upusher
        # self.robot.finger_length = 0.08
        # self.robot.finger_angle = 0.468
        # self.robot.distal_phalanx_length = 0.025

        # self.robot.v_angle = 2.6179938779914944 # iter9,0.44
        # self.robot.finger_length = 0.12
        # self.robot.finger_angle = -0.11635528346628865
        # self.robot.distal_phalanx_length = 0.08

        # self.robot.v_angle = 1.68715 # iter24,0.56 - notused
        # self.robot.finger_length = 0.12
        # self.robot.finger_angle = -0.3490
        # self.robot.distal_phalanx_length = 0.071111

        # self.robot.v_angle = 1.6871516102611852 # iter43,0.88
        # self.robot.finger_length = 0.12
        # self.robot.finger_angle = -0.11635528346628865
        # self.robot.distal_phalanx_length = 0.08

        return super().reset(seed=seed, options=options)

    def reset_task_and_design(
        self, 
        task_int: int,
        design_params: list,
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self.step_count = 0

        # Reset task and design parameters
        self.is_invalid_design = True
        self.task.task_int = task_int
        self.task.task_object_name = self.task.task_object_names_dict[task_int]
        self.robot.v_angle = design_params[0]
        self.robot.finger_length = design_params[1]
        self.robot.finger_angle = design_params[2]
        self.robot.distal_phalanx_length = design_params[3]
        if self._pusher_forward_kinematics()[1] > 0.0: # make sure the pusher does not self-intersect
            self.is_invalid_design = False

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
        if (gripper_out_of_canvas):
            self.is_safe = False
        # if truncated:
        #     print(f"gripper_out_of_canvas") if gripper_out_of_canvas else None
        #     print(f"object_out_of_canvas") if object_out_of_canvas else None
        #     print(f"time_ended") if time_ended else None

        return truncated
