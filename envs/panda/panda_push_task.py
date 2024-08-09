from typing import Any, Dict

import numpy as np
import math

from panda_gym.envs.core import Task
from panda_gym.utils import distance
import pybullet as p

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class VPush(Task):
    def __init__(
        self,
        sim,
        reward_type=None,
        distance_threshold=0.03,
        goal_xy_range=0.3,
        obj_xy_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([0.7, 0, 0])
        self.goal_range_high = np.array([0.75, 0.1, 0])
        self.obj_range_low = np.array([0.4, 0, 0])
        self.obj_range_high = np.array([0.5, 0.1, 0])
        self.task_object_name = 'circle' # 'circle', 'polygon'
        self.task_int = 0 if self.task_object_name == 'circle' else 1
        self.last_ee_object_distance = 0
        self.last_object_target_distance = 0
        with self.sim.no_rendering():
            self._create_scene()
            
    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=4, width=2, height=0.4, x_offset=0)
        self._create_task_object()

    def _create_task_object(self) -> None:
        if self.task_object_name == 'circle':
            self.sim.create_cylinder(
                body_name="object",
                radius=self.object_size/2,
                height=self.object_size,
                mass=1.0,
                position=np.array([0.0, 0.0, self.object_size/4]),
                rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
            )
            self.sim.create_cylinder(
                body_name="target",
                radius=self.object_size/2,
                height=self.object_size,
                mass=0.0,
                ghost=True,
                position=np.array([0.0, 0.0, self.object_size/4]),
                rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
            )
        elif self.task_object_name == 'polygon':
            self.sim.create_box(
                body_name="object",
                half_extents=np.array([1,1,1]) * self.object_size / 2,
                mass=1.0,
                position=np.array([0.0, 0.0, self.object_size / 4]),
                rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
            )
            self.sim.create_box(
                body_name="target",
                half_extents=np.array([1,1,1]) * self.object_size / 2,
                mass=0.0,
                ghost=True,
                position=np.array([0.0, 0.0, self.object_size / 4]),
                rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
            )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = np.array(self.sim.get_base_position("object"))
        object_rotation = np.array(self.sim.get_base_rotation("object"))
        object_velocity = np.array(self.sim.get_base_velocity("object"))
        object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object"))
        target_position = np.array(self.sim.get_base_position("target"))
        task_int = np.array([self.task_int,])
        observation = np.concatenate(
            [
                object_position,
                object_rotation,
                object_velocity,
                object_angular_velocity,
                target_position,
                task_int,
            ]
        )
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        # Reset task object
        p.removeBody(self.sim._bodies_idx["target"])
        p.removeBody(self.sim._bodies_idx["object"])
        self._create_task_object()
        # print(p.getNumBodies())
        # print(self.sim._bodies_idx)

        # Reset the object and goal position
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise = np.random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = np.random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, observation_dict, info) -> np.ndarray:
        observation = observation_dict["observation"]
        achieved_goal = observation_dict["achieved_goal"]
        desired_goal = observation_dict["desired_goal"]

        # Unpack the observation
        assert observation.shape == (24,)
        ee_position = observation[:3]
        ee_velocity = observation[3:6]
        ee_yaw = observation[6:7]
        v_angle = observation[7:8]
        object_position = observation[8:11]
        object_rotation = observation[11:14]
        object_velocity = observation[14:17]
        object_angular_velocity = observation[17:20]
        target_position = observation[20:23]
        task_int = observation[23:24]
        
        reward = 0
        ee_object_distance = distance(ee_position, object_position)
        diff_ee_object_distance = ee_object_distance - self.last_ee_object_distance
        self.last_ee_object_distance = ee_object_distance
        object_target_distance = distance(object_position, target_position) 
        diff_object_target_distance = object_target_distance - self.last_object_target_distance
        self.last_object_target_distance = object_target_distance
        object_target_yaw = math.atan2(target_position[1] - object_position[1], target_position[0] - object_position[0])
        yaw_difference = abs(pi_2_pi(object_rotation[2] - object_target_yaw))
        
        weight_ee_object_distance = 1.0
        weight_object_target_distance = 1.0
        weight_yaw = 0.1
        
        reward += -weight_ee_object_distance * diff_ee_object_distance - weight_object_target_distance * diff_object_target_distance - weight_yaw * yaw_difference
        if self.is_success(achieved_goal, desired_goal):
            reward += 100
        
        # reward -= distance(object_position, target_position)
        
        return reward
    
