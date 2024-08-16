from typing import Any, Dict

import numpy as np
import math

from panda_gym.envs.core import Task
from panda_gym.utils import distance
import pybullet as p
from shapely.geometry import Polygon, Point

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class VPush(Task):
    def __init__(
        self,
        sim,
        reward_type="sparse",
        using_robustness_reward=False,
        distance_threshold=0.1,
        goal_xy_range=0.3,
        obj_xy_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.08 # if too smaller than the geometry of tool, the object geometry variation will not be demonstrated
        self.goal_range_low = np.array([0.7, 0, 0])
        self.goal_range_high = np.array([0.75, 0.1, 0])
        self.obj_range_low = np.array([0.4, 0, 0])
        self.obj_range_high = np.array([0.5, 0.1, 0])
        self.task_object_name = 'circle' # initial choice
        self.task_object_names = ['circle', 'square', 'polygon0', 'narrow', 'oval']
        self.task_int = 0 if self.task_object_name == 'circle' else 1
        self.last_ee_object_distance = 0
        self.last_object_target_distance = 0
        self.design_params = None
        self.robustness_opening = None
        self.using_robustness_reward = using_robustness_reward
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
                height=self.object_size/2,
                mass=1.0,
                position=np.array([0.0, 0.0, self.object_size/4]),
                rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
            )
            self.sim.create_cylinder(
                body_name="target",
                radius=self.object_size/2,
                height=self.object_size/2,
                mass=0.0,
                ghost=True,
                position=np.array([0.0, 0.0, self.object_size/4]),
                rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
            )
        elif self.task_object_name == 'square':
            self.sim.create_box(
                body_name="object",
                half_extents=np.array([1,1,0.5]) * self.object_size / 2,
                mass=1.0,
                position=np.array([0.0, 0.0, self.object_size / 4]),
                rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
            )
            self.sim.create_box(
                body_name="target",
                half_extents=np.array([1,1,0.5]) * self.object_size / 2,
                mass=0.0,
                ghost=True,
                position=np.array([0.0, 0.0, self.object_size / 4]),
                rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
            )
        elif self.task_object_name == 'polygon0':
            file_name = "asset/polygons/poly0.obj"
            mesh_scale = [1,] * 3
            height = self.object_size / 2 # make sure the height is right in obj file
            self._create_task_object_mesh(file_name, mesh_scale, height)
        elif self.task_object_name == 'narrow':
            file_name = "asset/polygons/narrow.obj"
            mesh_scale = [1,] * 3
            height = self.object_size / 2 # make sure the height is right in obj file
            self._create_task_object_mesh(file_name, mesh_scale, height)
        elif self.task_object_name == 'oval':
            file_name = "asset/polygons/oval.obj"
            mesh_scale = [1,] * 3
            height = self.object_size / 2 # make sure the height is right in obj file
            self._create_task_object_mesh(file_name, mesh_scale, height)

        # for caging escape computation
        self.object_rad = {"circle": self.object_size/2, "square": self.object_size/2, 
                           "polygon0": 2.7/4.0*self.object_size/2, "narrow": 2.9/4.0*self.object_size/2, 
                           "oval": self.object_size/2}

    def _create_task_object_mesh(self, file_name, mesh_scale, height) -> None:
            self.sim._create_geometry(
                body_name="object",
                geom_type=self.sim.physics_client.GEOM_MESH,
                mass=1.0,
                position=np.array([0.0, 0.0, height/2]),
                visual_kwargs={
                        "fileName": file_name,
                        "meshScale": mesh_scale,
                        "rgbaColor": np.array([0.1, 0.9, 0.1, 1.0]),
                    },
                collision_kwargs={
                        "fileName": file_name,
                        "meshScale": mesh_scale,
                    },
            )
            self.sim._create_geometry(
                body_name="target",
                geom_type=self.sim.physics_client.GEOM_MESH,
                mass=0.0,
                position=np.array([0.0, 0.0, height/2]),
                ghost=True,
                visual_kwargs={
                        "fileName": file_name,
                        "meshScale": mesh_scale,
                        "rgbaColor": np.array([0.1, 0.9, 0.1, 0.3]),
                    },
                collision_kwargs={
                        "fileName": file_name,
                        "meshScale": mesh_scale,
                    },
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
        self.design_params = None
        self.robustness_opening = None
        # print(p.getNumBodies())
        # print(self.sim._bodies_idx)

        # Reset the object and goal position
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = np.array([0.0, 0.0, self.object_size/2])  # z offset for the cube center
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
        
    def _is_point_inside_polygon(self, point, vertices, slack=2):
        polygon = Polygon(vertices)
        point = Point(point)
        if polygon.contains(point):
            return True
        if polygon.distance(point) <= slack:
            return True

        return False
    
    def _eval_robustness(self, tool_position, tool_angle, object_position, design_params, slack=0.0):
        a1, l1, a2, l2 = design_params

        # Calculate the object position in the robot frame
        # object_pos = np.array(p.getBasePositionAndOrientation(self.object_id)[0][:2])
        # robot_pos = np.array(p.getBasePositionAndOrientation(self.robot_id)[0][:2])
        # robot_angle = p.getBasePositionAndOrientation(self.robot_id)[1]
        # robot_angle = p.getEulerFromQuaternion(robot_angle)[2]
        object_pos_R = object_position - tool_position
        object_pos_R = np.array([ # robot frame
            object_pos_R[0] * math.cos(tool_angle) + object_pos_R[1] * math.sin(tool_angle),
            -object_pos_R[0] * math.sin(tool_angle) + object_pos_R[1] * math.cos(tool_angle)
        ])

        # Compute robot vertices position in the robot frame
        robot_vertices = [(0, 0),  # robot frame
                               (l1*math.cos(a1/2), l1*math.sin(a1/2)),
                               (l1*math.cos(-a1/2), l1*math.sin(-a1/2)),
                               (l1*math.cos(a1/2) + l2*math.cos(a2), l1*math.sin(a1/2) + l2*math.sin(a2)),
                               (l1*math.cos(-a1/2) + l2*math.cos(-a2), l1*math.sin(-a1/2) + l2*math.sin(-a2)),
        ]
                            #    (self.finger_length*math.cos(self.v_angle/2), self.finger_length*math.sin(self.v_angle/2)),
                            #    (self.finger_length*math.cos(-self.v_angle/2), self.finger_length*math.sin(-self.v_angle/2))]
        
        if self._is_point_inside_polygon(object_pos_R, robot_vertices, slack=slack):
            soft_fixture_metric = l1*math.cos(a1/2) + l2*math.cos(a2) - object_pos_R[0] + self.object_rad[self.task_object_name]
            # soft_fixture_metric = l1*math.cos(a1/2) +  - object_pos_R[0] + self.object_rad
            robustness_depth = max(0.0, soft_fixture_metric) # depth
            robustness = 10*robustness_depth + self.robustness_opening
            # print("robustness_depth: ", robustness_depth)
            # print("robustness: ", robustness)
        else:
            robustness = 0

        return robustness

    def _compute_robustness_opening(self,):
        a1, l1, a2, l2 = self.design_params
        opening_gap = 2 * (l1 * np.sin(a1/2) + l2 * np.sin(a2))
        self.robustness_opening = 0.03 / opening_gap
  
  
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
      
    # def compute_reward(self, observation_dict, info) -> np.ndarray:
    #     observation = observation_dict["observation"]
    #     achieved_goal = observation_dict["achieved_goal"]
    #     desired_goal = observation_dict["desired_goal"]

    #     # Unpack the observation
    #     assert observation.shape == (27,)
    #     ee_position = observation[:3]
    #     ee_velocity = observation[3:6]
    #     ee_yaw = observation[6:7]
    #     self.design_params = observation[7:11]
    #     object_position = observation[11:14]
    #     object_rotation = observation[14:17]
    #     object_velocity = observation[17:20]
    #     object_angular_velocity = observation[20:23]
    #     target_position = observation[23:26]
    #     task_int = observation[26:27]
        
    #     # Caging robostness
    #     if self.design_params is not None and self.robustness_opening is None:
    #         self._compute_robustness_opening()
    #     robustness_score = self._eval_robustness(ee_position[:2], 
    #                                             ee_yaw[0],
    #                                             object_position[:2],
    #                                             self.design_params,
    #                                             slack=self.object_size/2,)

    #     reward = 0
    #     ee_object_distance = distance(ee_position, object_position)            
    #     diff_ee_object_distance = ee_object_distance - self.last_ee_object_distance
    #     self.last_ee_object_distance = ee_object_distance
    #     object_target_distance = distance(object_position, target_position) 
    #     diff_object_target_distance = object_target_distance - self.last_object_target_distance
    #     self.last_object_target_distance = object_target_distance
    #     object_target_yaw = math.atan2(target_position[1] - object_position[1], target_position[0] - object_position[0])
    #     yaw_difference = abs(pi_2_pi(ee_yaw - object_target_yaw))
    

    #     if ee_object_distance > 0.2:
    #         weight_ee_object_distance = 1.0
    #         weight_object_target_distance = 0
    #         weight_yaw = 0.5
    #     else:
    #         weight_ee_object_distance = 0
    #         weight_object_target_distance = 1.0
    #         weight_yaw = 0
    #     reward += -weight_ee_object_distance * ee_object_distance - weight_object_target_distance * object_target_distance - weight_yaw * yaw_difference
        
    #     if self.is_success(achieved_goal, desired_goal):
    #         reward += 500
        
    #     if self.using_robustness_reward:
    #         reward += robustness_score      
    #     return reward
    
