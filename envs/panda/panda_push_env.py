from typing import Optional, Any, Dict, Optional, Tuple

import numpy as np
import random
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
from typing import Any, Dict, Optional, Tuple
from gymnasium.utils import seeding

from .panda_robot_custom import PandaCustom
from .panda_push_task import VPush

class PandaPushEnv(RobotTaskEnv):
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
        gui: bool = False,
        obs_type: str = "pose",
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
        if gui == False:
            render_mode: str = "rgb_array"
        else:
            render_mode: str = "human"
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = PandaCustom(sim, block_gripper=True, base_position=np.array([0.0, 0.0, 0.0]), control_type=control_type)
        task = VPush(sim, reward_type=reward_type)
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
        # self.task_object_name = 'circle' 
        # self.task_int = 0 if self.task_object_name == 'circle' else 1
        # self.v_angle = np.pi/3
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:    
        self.step_count += 1
        self.robot.set_action(action)
        self.sim.step()
        observation = self._get_obs()
        terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        info = {"is_success": terminated}
        truncated = self._is_truncated()
        reward = float(self.task.compute_reward(observation, info))
        return observation, reward, terminated, truncated, info
        
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        # TODO: panda/object cannot be loaded in gui after about 7 episodes
        self.step_count = 0
        self.task.task_object_name = random.choice(['circle', 'polygon'])
        self.task.task_int = 0 if self.task.task_object_name == 'circle' else 1
        self.robot.v_angle = random.uniform(np.pi/12, np.pi*11/12)
        return super().reset(seed=seed, options=options)
    
    def _is_truncated(self):
        truncated = False
        truncated = (self.step_count > 200)
        return truncated
