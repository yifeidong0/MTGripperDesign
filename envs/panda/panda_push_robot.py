from typing import Optional

import numpy as np
from gymnasium import spaces
import pybullet as p
from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet

import os
import time
from utils.generate_vpush import generate_v_shape_pusher
from utils.vhacd import decompose_mesh

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class PandaCustom(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optional): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = True, # Not using gripper
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
        run_id: str = "",
    ) -> None:
        self.base_position = base_position if base_position is not None else np.zeros(3)
        self.control_type = control_type
        # action space (dx, dy, dyaw)
        action_space = spaces.Box(low=np.array([-0.02, -0.02, -0.05,]), high=np.array([0.02, 0.02, 0.05,]), dtype=np.float32)
        # action_space = spaces.Box(low=np.array([-0.01, -0.01,]), high=np.array([0.01, 0.01,]), dtype=np.float32)

        self.v_angle = 0.5
        self.panda_file_name = "asset/franka_panda_custom/panda_template.urdf"
        self.finger_length = 0.2
        self.finger_thickness = 0.01
        self.body_height = 0.02 # finger's height
        self.finger_angle = 0.0
        self.distal_phalanx_length = 0.1
        self.design_params = np.array([self.v_angle, self.finger_length, self.finger_angle, self.distal_phalanx_length])

        self.num_episodes = 0
        self.constraint_id = None
        self.run_id = run_id
        super().__init__(
            sim,
            body_name="panda",
            file_name=self.panda_file_name,
            base_position=self.base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]), # TODO
        )

        self.fingers_indices = np.array([9, 10])
        # self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        # self.neutral_joint_values = np.array([0.000, 0.146, 0.000, -3.041, 0.000, 3.182, 0.790, 0.000, 0.000])
        # self.neutral_joint_values = np.array([0.54664663, 0.5078128, -0.21580158, -2.10046213, 0.19967674, 2.58959611, 0.97221048])
        self.neutral_joint_values = np.array([0.8, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79])
        self.neutral_joint_zeros = np.array([0.000,]*9)
        
        # self.ee_link = 11 # vpush
        self.ee_link = 10
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)
        self.joint_limits = [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973),
        ]
        self.z_offset = 0.12
        self.ee_target_position = np.zeros(3)
        self.ee_target_position[2] = self.z_offset
        self.ee_target_yaw = 0
        self.ee_init_pos_2d = np.zeros(2)
        self.attached_tool = False
        self.last_observation = None
        self.last_action = None
        self.current_action = None

    def set_action(self, action: np.ndarray) -> None:
        if self.control_type == "ee":
            ee_displacement = np.array([action[0], action[1], 0])
            ee_orientation_change = action[2]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement, ee_orientation_change)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)            
        for i, (lower, upper) in enumerate(self.joint_limits):
            target_arm_angles[i] = np.clip(target_arm_angles[i], lower, upper)
        target_fingers_width = 0.02 # fixed fingers width
        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_joints(target_angles=target_angles)

    def ee_displacement_to_target_arm_angles(self, 
                                             ee_displacement: np.ndarray, 
                                             ee_orientation_change: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dz).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        # Set target end-effector position
        ee_displacement = ee_displacement[:3]  # limit maximum change in position
        # ee_position = self.get_ee_position() # get the current position and the target position
        # target_ee_position = ee_position + ee_displacement
        self.ee_target_position[:2] += ee_displacement[:2]

        # Set target end-effector orientation
        ee_quaternion = self.get_ee_orientation()
        ee_euler = list(p.getEulerFromQuaternion(ee_quaternion)) # tuple
        # target_ee_euler = [-np.pi, 0, 0]
        self.ee_target_yaw = pi_2_pi(ee_euler[2] + ee_orientation_change)
        target_ee_euler = [-np.pi, 0, self.ee_target_yaw]
        target_ee_quaternion = np.array(list(p.getQuaternionFromEuler(target_ee_euler)))

        # Clip the height target. For some reason, it has a great impact on learning
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=self.ee_target_position, orientation=target_ee_quaternion
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        # clip the target joint angles according to the joint limits
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position_2d = np.array(self.get_ee_position())[:2]
        ee_velocity = np.array(self.get_ee_velocity())
        ee_yaw = list(p.getEulerFromQuaternion(self.get_ee_orientation()))[-1]
        arm_joint_angles = self.get_arm_joint_angles()
        # fingers opening
        self.design_params = np.array([self.v_angle, self.finger_length, self.finger_angle, self.distal_phalanx_length])
        current_observation = np.concatenate((ee_position_2d, [ee_yaw,], self.design_params, ))
        if self.last_observation is None:
            observation = np.concatenate((current_observation, current_observation))
        else:
            observation = np.concatenate((current_observation, self.last_observation))
        self.last_observation = current_observation.copy()
        return observation

    def reset(self) -> None:
        # self._reload_robot()
        self.num_episodes += 1
        # if self.num_episodes % 5 == 0:
        #     print(f"INFO: episode {self.num_episodes}")
        # if self.constraint_id is not None:
        #     p.removeConstraint(self.constraint_id)
        self._set_random_ee_pose()
        if "tool" in self.sim._bodies_idx:
                p.removeBody(self.sim._bodies_idx["tool"])
        self._attach_tool_to_ee()
        # if self.attached_tool is False:
        #     if "tool" in self.sim._bodies_idx:
        #         p.removeBody(self.sim._bodies_idx["tool"])
        #     self._attach_tool_to_ee()
        #     self.attached_tool = True

    def _set_random_ee_pose(self) -> None:
        """Set the robot to a random end-effector initial pose after reset."""
        self.set_joint_neutral() # set neutral first to avoid singularities
        init_ee_position = np.array([0.0, 0.0, self.z_offset])
        init_ee_position[0] = np.random.uniform(0.3, 0.6) 
        init_ee_position[1] = np.random.uniform(0.2, 0.3)
        self.ee_init_pos_2d = init_ee_position[:2]
        self.ee_target_position[:2] = self.ee_init_pos_2d
        self.ee_target_yaw = 0
        init_ee_euler = [-np.pi, 0, 0]
        init_ee_quaternion = np.array(list(p.getQuaternionFromEuler(init_ee_euler)))
        init_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=init_ee_position, orientation=init_ee_quaternion
        )
        init_arm_angles = init_arm_angles[:7]  # remove fingers angles
        for i, (lower, upper) in enumerate(self.joint_limits):
            init_arm_angles[i] = np.clip(init_arm_angles[i], lower, upper)
        init_arm_angles = np.concatenate((init_arm_angles, [0.02, 0.02])) 
        self.set_joint_angles(init_arm_angles)

    def _attach_tool_to_ee(self) -> None:
        """Attach the tool to the end-effector."""
        os.makedirs(f"asset/{self.run_id}", exist_ok=True)
        os.system(f"rm -rf asset/{self.run_id}/*")
        unique_obj_filename = f"v_pusher_{self.v_angle:.2f}_{self.finger_length:.2f}_{self.finger_angle:.2f}_{self.distal_phalanx_length:.2f}.obj"
        tool_obj_path = f"asset/{self.run_id}/{unique_obj_filename}"
        generate_v_shape_pusher(self.finger_length, self.v_angle, self.finger_thickness, self.body_height, 
                                tool_obj_path, self.finger_angle, self.distal_phalanx_length)
        decompose_mesh(pb_connected=True, input_file=tool_obj_path)
        with open("asset/franka_panda_custom/vpusher_template.urdf", 'r') as file:
            self.urdf_content = file.read()
        self.urdf_content = self.urdf_content.replace('{v_pusher_name}', tool_obj_path)

        self.modified_vpusher_path = f"asset/{self.run_id}/vpusher_modified_{self.v_angle:.3f}.urdf"
        with open(self.modified_vpusher_path, 'w') as file:
            file.write(self.urdf_content)

        # Load the tool
        ee_position_center = (self.get_ee_position() + self.get_link_position(self.ee_link-1)) / 2 # center of the two fingers
        ee_position_center[2] -= 0.05
        self.sim.loadURDF(
            body_name="tool",
            fileName=self.modified_vpusher_path,
            basePosition=ee_position_center,
            useFixedBase=0,
        )

        # Create a constraint to attach the tool to the end-effector
        end_effector_index = 10
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.sim._bodies_idx[self.body_name],
            parentLinkIndex=end_effector_index,
            childBodyUniqueId=self.sim._bodies_idx["tool"],
            childLinkIndex=-1,  # -1 means we are attaching to the base of the tool
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0.0, 0.02, 0.075], # align with center of two parallel jaw fingers (z: leave space for handle in real world)
            childFramePosition=[0, 0, 0],
            childFrameOrientation=p.getQuaternionFromEuler([3.14, 0, np.pi/2]), # important to avoid arm jerk when adding constraint
        )
    
    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def set_joint_zero(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_zeros)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_orientation(self) -> np.ndarray:
        """Returns the orientation of the end-effector as an euler angle (r, p, y)"""
        return self.sim.get_link_orientation(self.body_name, self.ee_link)
    
    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

    def get_arm_joint_angles(self) -> np.ndarray:
        """Returns the angles of the 7 arm joints."""
        return np.array([self.get_joint_angle(joint=i) for i in range(7)])