from typing import Any, Dict

import numpy as np
import math

from panda_gym.envs.core import Task
from panda_gym.utils import distance
import pybullet as p
from shapely.geometry import Polygon, Point
from polygon import is_point_inside_polygon

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class UPush(Task):
    def __init__(
        self,
        sim,
        reward_type,
        using_robustness_reward=True,
        reward_weights=[1.0, 0.01, 1.0, 1.0, 100.0, 0.0, 0.0, 0.0],
        distance_threshold=0.05,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.08 # if too smaller than the geometry of tool, the object geometry variation will not be demonstrated
        self.task_object_name = 'circle' # initial choice
        self.task_object_names = ['circle', 'square', 'polygon0', 'narrow', 'oval']
        self.task_object_names_dict = {0: 'circle', 1: 'square', 2: 'polygon0', 3: 'narrow', 4: 'oval'}
        self.task_ints = {'circle': 0, 'square': 1, 'polygon0': 2, 'narrow': 3, 'oval': 4}
        self.task_int = self.task_ints[self.task_object_name]
        self.last_ee_object_distance = 0
        self.last_object_target_distance = 0
        self.using_robustness_reward = using_robustness_reward
        self.reward_weights = reward_weights
        self.ee_init_pos_2d = None
        self.robustness_score = 0
        self.is_safe = False
        self.is_success_flag = False
        with self.sim.no_rendering():
            self._create_scene()
            
    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=2, width=2, height=0.4, x_offset=0, lateral_friction=40/77)
        self._create_task_object()

    def _create_task_object(self) -> None:
        if self.task_object_name == 'circle':
            self.sim.create_cylinder(
                body_name="object",
                radius=self.object_size/2,
                height=self.object_size/2*1.6, # 6.4cm
                mass=0.079, # TODO: adapt it after printing the objects and weighing them
                position=np.array([0.0, 0.0, self.object_size/2*1.6/2]),
                rgba_color=np.array([0.9, 0.2,  0.2, 1.0]),
                lateral_friction=40/77,
            )
            self.sim.create_cylinder(
                body_name="target",
                radius=self.object_size/2,
                height=self.object_size/2*1.6,
                mass=0.0,
                ghost=True,
                position=np.array([0.0, 0.0, self.object_size/2*1.6/2]),
                rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
            )
        elif self.task_object_name == 'square':
            self.sim.create_box(
                body_name="object",
                half_extents=np.array([1,1,0.8]) * self.object_size / 2,
                mass=0.104, # 3d printed object
                position=np.array([0.0, 0.0, self.object_size/2*1.6/2]),
                rgba_color=np.array([0.9, 0.2,  0.2, 1.0]),
                lateral_friction=40/77,
            )
            self.sim.create_box(
                body_name="target",
                half_extents=np.array([1,1,0.8]) * self.object_size / 2,
                mass=0.0,
                ghost=True,
                position=np.array([0.0, 0.0, self.object_size/2*1.6/2]),
                rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
            )
        elif self.task_object_name == 'polygon0':
            file_name = "asset/polygons/poly0.obj"
            mesh_scale = [1,1,1.6]
            height = self.object_size / 2 * mesh_scale[2] # make sure the height is right in obj file
            self._create_task_object_mesh(file_name, mesh_scale, height, mass=0.084)
        elif self.task_object_name == 'narrow':
            file_name = "asset/polygons/narrow.obj"
            mesh_scale = [1,1,1.6]
            height = self.object_size / 2 * mesh_scale[2] # make sure the height is right in obj file
            self._create_task_object_mesh(file_name, mesh_scale, height, mass=0.116)
        elif self.task_object_name == 'oval':
            file_name = "asset/polygons/oval.obj"
            mesh_scale = [1,1,1.6]
            height = self.object_size / 2 * mesh_scale[2] # make sure the height is right in obj file
            self._create_task_object_mesh(file_name, mesh_scale, height, mass=0.111)

        # for caging escape computation
        self.all_object_rad = {"circle": self.object_size/2, "square": self.object_size/2, 
                           "polygon0": 2.7/4.0*self.object_size/2, "narrow": 2.9/4.0*self.object_size/2, 
                           "oval": self.object_size/2}

    def _create_task_object_mesh(self, file_name, mesh_scale, height, mass=0.1) -> None:
            self.sim._create_geometry(
                body_name="object",
                geom_type=self.sim.physics_client.GEOM_MESH,
                mass=mass,
                position=np.array([0.0, 0.0, height/2]),
                visual_kwargs={
                        "fileName": file_name,
                        "meshScale": mesh_scale,
                        "rgbaColor": np.array([0.9, 0.2,  0.2, 1.0]),
                    },
                collision_kwargs={
                        "fileName": file_name,
                        "meshScale": mesh_scale,
                    },
                lateral_friction=40/77,
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
        object_rotation = np.array(self.sim.get_base_rotation("object"))[2:]
        task_int = np.array([self.task_int,])
        object_rad = np.array([self.all_object_rad[self.task_object_name]])
        
        # observation = np.concatenate([object_rotation, task_int, object_rad, self.random_target])
        observation = np.concatenate([object_rotation, task_int, object_rad])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        self.is_truncated = False
        # Reset task object
        p.removeBody(self.sim._bodies_idx["target"])
        p.removeBody(self.sim._bodies_idx["object"])
        self._create_task_object()

        # Reset the object and goal position
        self.init_object_position = self._sample_object()
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", self.init_object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        
        # # only for debugging
        # self.random_target = np.zeros(2)
        # self.random_target[0] = np.random.uniform(0.3, 0.6)
        # self.random_target[1] = np.random.uniform(0.2, 0.4)
        # self.is_success_flag = False

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        self.goal_range_low = np.array([0.35, -0.3, 0])
        self.goal_range_high = np.array([0.55, self.init_object_position[1] - 0.08, 0])
        while True:
            goal = np.array([0.0, 0.0, self.object_size/2])  # z offset for the cube center
            noise = np.random.uniform(self.goal_range_low, self.goal_range_high)
            goal += noise
            if distance(goal, self.init_object_position) > 0.1:
                break
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        self.obj_range_low = np.array([0.35, -0.1, 0])
        self.obj_range_high = np.array([0.55, self.ee_init_pos_2d[1] - 0.05, 0])
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = np.random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        # return self.is_success_flag
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def _is_point_inside_polygon(self, points, vertices, slack=2):
        """
        Batch processing version of _is_point_inside_polygon.
        points: numpy array of shape [N, 2]
        polygons: numpy array of shape [N, 5, 2] where 5 is the number of vertices in each U shape polygon
        slack: tolerance distance for considering a point inside the polygon
        returns: boolean array of shape [N] indicating whether each point is inside the polygon or within slack distance
        """
        results = np.zeros(points.shape[0], dtype=bool)
        for i in range(points.shape[0]):
            polygon = Polygon(vertices[i])
            point = Point(points[i])
            if polygon.contains(point):
                results[i] = True
            elif polygon.distance(point) <= slack:
                results[i] = True
        return results

    def _eval_robustness(self, tool_positions, tool_angles, object_positions, design_params, object_rad, is_inside_gripper, slack=0.0):
        """
        Batch processing version of _eval_robustness.
        tool_positions: numpy array of shape [N, 2]
        tool_angles: numpy array of shape [N]
        object_positions: numpy array of shape [N, 2]
        design_params: numpy array of shape [N, 4] where each row contains [a1, l1, a2, l2]
        slack: tolerance value used in is_point_inside_polygon
        returns: numpy array of shape [N] containing the robustness scores for each evaluation
        """
        N = tool_positions.shape[0]
        assert design_params.shape == (N, 4) and object_positions.shape == (N, 2) and tool_positions.shape == (N, 2) and tool_angles.shape == (N, 1)
        a1 = design_params[:, 0]
        l1 = design_params[:, 1]
        a2 = design_params[:, 2]
        l2 = design_params[:, 3]
        
        # Calculate opening_gap and robustness_opening for each evaluation
        opening_gap = 2 * (l1 * np.sin(a1/2) + l2 * np.sin(a2))
        robustness_opening = 0.03 / opening_gap
        
        # Calculate object positions in the robot frame for each evaluation
        delta_positions = object_positions - tool_positions
        cos_angles = np.cos(tool_angles)
        sin_angles = np.sin(tool_angles)
        
        object_pos_R = np.stack([
            delta_positions[:, 0] * cos_angles[:, 0] + delta_positions[:, 1] * sin_angles[:, 0],
            -delta_positions[:, 0] * sin_angles[:, 0] + delta_positions[:, 1] * cos_angles[:, 0]
        ], axis=1)

        vertex_1 = np.zeros((N, 2))
        vertex_2 = np.stack([l1 * np.cos(a1/2), l1 * np.sin(a1/2)], axis=1)
        vertex_3 = np.stack([l1 * np.cos(-a1/2), l1 * np.sin(-a1/2)], axis=1)
        vertex_4 = np.stack([l1 * np.cos(a1/2) + l2 * np.cos(a2), l1 * np.sin(a1/2) + l2 * np.sin(a2)], axis=1)
        vertex_5 = np.stack([l1 * np.cos(-a1/2) + l2 * np.cos(-a2), l1 * np.sin(-a1/2) + l2 * np.sin(-a2)], axis=1)

        robot_vertices = np.stack([vertex_1, vertex_2, vertex_3, vertex_4, vertex_5], axis=1)  # (N, 5, 2)
        assert robot_vertices.shape == (N, 5, 2) and object_pos_R.shape == (N, 2)
        # Check if each object position is inside the corresponding robot polygon
        # is_inside = self._is_point_inside_polygon(object_pos_R, robot_vertices, slack=slack)
        # is_inside = is_point_inside_polygon(object_pos_R, robot_vertices, slack=slack) # TODO
        # Calculate robustness for each evaluation
        soft_fixture_metric = l1 * np.cos(a1/2) + l2 * np.cos(a2) - object_pos_R[:, 0] + object_rad[:, 0]
        robustness_depth = np.maximum(0.0, soft_fixture_metric)
        if is_inside_gripper:
            return 10 * robustness_depth[0] + robustness_opening[0]
        else:
            return 0        
        # robustness = np.where(is_inside_gripper, 10 * robustness_depth + robustness_opening, 0)
        # return robustness
  
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, observation: np.ndarray,) -> np.ndarray:
        if observation.ndim == 1:
            achieved_goal = np.expand_dims(achieved_goal, axis=0)  
            desired_goal = np.expand_dims(desired_goal, axis=0)
            observation = np.expand_dims(observation, axis=0) 
        # assert observation.shape[-1] == 10
        ee_position_2d = observation[..., :2]
        ee_yaw = observation[..., 2].reshape(-1, 1)
        design_params = observation[..., 3:7]
        object_rotation = observation[..., 7].reshape(-1, 1)
        task_int = observation[..., 8].reshape(-1, 1)
        object_rad = observation[..., 9].reshape(-1, 1)
        
        # random_target = self.random_target.reshape(1, 2)
        # reward = -distance(ee_position_2d, random_target)
        # if distance(ee_position_2d, random_target) < 0.05:
        #     self.is_success_flag = True
        #     reward += 10
        # return reward # only for debugging
        
        ee_object_distance = distance(ee_position_2d, achieved_goal[..., :2])
        object_target_distance = distance(achieved_goal[..., :2], desired_goal[..., :2])
        object_target_yaw = np.arctan2(desired_goal[..., 1] - achieved_goal[..., 1], desired_goal[..., 0] - achieved_goal[..., 0])
        ee_object_yaw = np.arctan2(achieved_goal[..., 1] - ee_position_2d[..., 1], achieved_goal[..., 0] - ee_position_2d[..., 0])
        ee_target_yaw = np.arctan2(desired_goal[..., 1] - ee_position_2d[..., 1], desired_goal[..., 0] - ee_position_2d[..., 0])
        yaw_difference_ee_object = abs(pi_2_pi(ee_yaw - np.pi / 2 - ee_object_yaw))
        yaw_difference_ee_target = abs(pi_2_pi(ee_yaw - np.pi / 2 - ee_target_yaw))

        reward = 0
        
        if self.reward_type == "sparse":
            d = distance(achieved_goal, desired_goal)
            return reward - np.array(d > self.distance_threshold, dtype=np.float32)

        is_inside_gripper = False
        if ee_object_distance < 0.15 and yaw_difference_ee_object < np.pi / 6:
            is_inside_gripper = True

        self.robustness_score = self._eval_robustness(ee_position_2d[:, :2],
                                                     ee_yaw,
                                                     achieved_goal[..., :2],
                                                     design_params,
                                                     object_rad,
                                                     is_inside_gripper,
                                                     slack=0)
        
        
        # Caging robustness
        if self.using_robustness_reward:
            reward += self.robustness_score * self.reward_weights[4]

        # imaged goal locating at the extension of the target-object line
        # line_vector = achieved_goal[..., :2] - desired_goal[..., :2]
        # direction_vector = line_vector / np.linalg.norm(line_vector)
        # delta = 0.1
        # middle_goal = achieved_goal[..., :2] + direction_vector * delta
        # ee_middle_goal_distance = distance(ee_position_2d, middle_goal)
        
        if is_inside_gripper is False:
            weight_ee_object_distance = self.reward_weights[5]
            weight_yaw_ee_object = self.reward_weights[0]
            reward += - weight_yaw_ee_object * yaw_difference_ee_object - weight_ee_object_distance * ee_object_distance
        else:
            weight_ee_object_distance = self.reward_weights[5]
            weight_yaw_ee_object = self.reward_weights[0]
            weight_object_target_distance = self.reward_weights[3]
            weight_yaw_ee_target = self.reward_weights[1]
            reward += - weight_yaw_ee_object * yaw_difference_ee_object - weight_ee_object_distance * ee_object_distance\
                      - weight_object_target_distance * object_target_distance - weight_yaw_ee_target * yaw_difference_ee_target
        
        if self.is_success(achieved_goal, desired_goal):
            reward += self.reward_weights[2]
                                
        return reward
    
