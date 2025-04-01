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


class Xarm7(PyBulletRobot):
    def __init__(
            self,
            sim: PyBullet,
            base_position: Optional[np.ndarray] = None,
            control_type: str = "ee",
            run_id: str = "",
    ) -> None:
        self.base_position = base_position if base_position is not None else np.zeros(3)
        self.control_type = control_type
        # action space (dx, dy, dyaw)
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)

        self.v_angle = 0.5
        self.xarm7_file_name = "asset/xarm7/xarm7_template.urdf"
        self.finger_length = 0.2
        self.finger_thickness = 0.01
        self.body_height = 0.02  # finger's height
        self.finger_angle = 0.0
        self.distal_phalanx_length = 0.1
        self.design_params = np.array([self.v_angle, self.finger_length, self.finger_angle, self.distal_phalanx_length])

        self.num_episodes = 0
        self.constraint_id = None
        self.run_id = run_id
        super().__init__(
            sim,
            body_name="xarm7",
            file_name=self.xarm7_file_name,
            base_position=self.base_position,
            action_space=self.action_space,
            joint_indices=np.array([1, 2, 3, 4, 5, 6, 7,]),
            joint_forces=np.array([60.0, 60.0, 40.0, 40.0, 40.0, 20.0, 20.0,])
        )

        self.neutral_joint_values = np.array([0.0, -0.93368, 0.0, 0.68935, 0.0, 1.6213, 0.0,])
        self.ee_link = 9
        self.z_offset = 0.12
        self.ee_target_position = np.zeros(3)
        self.ee_target_position[2] = self.z_offset
        self.ee_target_yaw = 0
        self.ee_init_pos_2d = np.zeros(2)
        self.attached_tool = False

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_min = np.array([-0.01, -0.01, -0.05])
        action_max = np.array([0.01, 0.01, 0.05])
        action = (action + 1) / 2 * (action_max - action_min) + action_min
        if self.control_type == "ee":
            ee_displacement = np.array([action[0], action[1], 0])
            ee_orientation_change = action[2]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement, ee_orientation_change)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)
        self.control_joints(target_angles=target_arm_angles)

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
        ee_euler = list(p.getEulerFromQuaternion(ee_quaternion))  # tuple
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
        observation = np.concatenate((ee_position_2d, [ee_yaw, ], self.design_params))
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
        
        # Debugging
        # if self.attached_tool is False:
        #     if "tool" in self.sim._bodies_idx:
        #         p.removeBody(self.sim._bodies_idx["tool"])
        #     self._attach_tool_to_ee()
        #     self.attached_tool = True

    def _set_random_ee_pose(self) -> None:
        """Set the robot to a random end-effector initial pose after reset."""
        self.set_joint_neutral()  # set neutral first to avoid singularities
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
        with open("asset/xarm7/vpusher_template.urdf", 'r') as file:
            self.urdf_content = file.read()
        self.urdf_content = self.urdf_content.replace('{v_pusher_name}', tool_obj_path)

        self.modified_vpusher_path = f"asset/{self.run_id}/vpusher_modified_{self.v_angle:.3f}.urdf"
        with open(self.modified_vpusher_path, 'w') as file:
            file.write(self.urdf_content)

        # Load the tool
        # ee_position_center = (self.get_ee_position() + self.get_link_position(
        #     self.ee_link - 1)) / 2  # center of the two fingers
        ee_position_center = self.get_ee_position()
        ee_position_center[2] -= 0.05
        self.sim.loadURDF(
            body_name="tool",
            fileName=self.modified_vpusher_path,
            basePosition=ee_position_center,
            useFixedBase=0,
        )

        # Create a constraint to attach the tool to the end-effector
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.sim._bodies_idx[self.body_name],
            parentLinkIndex=self.ee_link,
            childBodyUniqueId=self.sim._bodies_idx["tool"],
            childLinkIndex=-1,  # -1 means we are attaching to the base of the tool
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0.0, 0.02, 0.075],
            # align with center of two parallel jaw fingers (z: leave space for handle in real world)
            childFramePosition=[0, 0, 0],
            childFrameOrientation=p.getQuaternionFromEuler([3.14, 0, np.pi / 2]),
            # important to avoid arm jerk when adding constraint
        )

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        drive_jnt = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        return 0.018 + 0.11 * np.sin(0.069-drive_jnt)

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