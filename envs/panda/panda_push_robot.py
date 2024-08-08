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
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        self.base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y, yaw) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(low=np.array([-.3,]*2+[-0.1,]), high=np.array([.3,]*2+[0.1,]), dtype=np.float32)
        # action_space = spaces.Box(low=np.array([0.,-0.3,0,]), high=np.array([0,0,0,]), dtype=np.float32)
        
        self.v_angle = 0.5
        self.template_file_name = "asset/franka_panda_custom/panda_template.urdf"
        self.modified_file_path = f"asset/franka_panda_custom/panda_modified_{self.v_angle:.3f}.urdf"
        self.finger_length = 0.2
        self.finger_thickness = 0.01
        self.body_height = 0.01
        self.num_episodes = 0

        # os.system("rm -rf asset/vpusher/*")
        # unique_obj_filename = f"v_pusher_{self.v_angle:.3f}.obj"
        # generate_v_shape_pusher(self.finger_length, self.v_angle, self.finger_thickness, self.body_height, f"asset/vpusher/{unique_obj_filename}")
        # decompose_mesh(pb_connected=True, input_file=f"asset/vpusher/{unique_obj_filename}")
        # with open(self.template_file_name, 'r') as file:
        #     self.urdf_content = file.read()
        # self.urdf_content = self.urdf_content.replace('{v_pusher_name}', f"asset/vpusher/{unique_obj_filename}")
        # with open(self.modified_file_path, 'w') as file:
        #     file.write(self.urdf_content)
            
        super().__init__(
            sim,
            body_name="panda",
            # file_name="franka_panda/panda.urdf",
            file_name=self.template_file_name,
            base_position=self.base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )

        self.fingers_indices = np.array([9, 10])
        # self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.neutral_joint_values = np.array([0.000, 0.146, 0.000, -3.041, 0.000, 3.182, 0.790, 0.000, 0.000])
        self.neutral_joint_zeros = np.array([0.000,]*9)
        
        # self.ee_link = 11 # vpush
        self.ee_link = 10
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)
        

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = np.array([action[0], action[1], 0])
            ee_orientation_change = np.array([action[2],])
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement, ee_orientation_change)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

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
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        ee_position = self.get_ee_position() # get the current position and the target position
        target_ee_position = ee_position + ee_displacement
        target_ee_position[2] = 0.06 # corresponds to the liftup in urdf

        # Set target end-effector orientation
        ee_quaternion = self.get_ee_orientation()
        ee_euler = list(p.getEulerFromQuaternion(ee_quaternion)) # tuple
        target_ee_euler = [-np.pi, 0, pi_2_pi(ee_euler[2] + ee_orientation_change[0])]
        target_ee_quaternion = np.array(list(p.getQuaternionFromEuler(target_ee_euler)))

        # Clip the height target. For some reason, it has a great impact on learning
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=target_ee_quaternion
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
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
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        ee_yaw = list(p.getEulerFromQuaternion(self.get_ee_orientation()))[-1]

        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            observation = np.concatenate((ee_position, ee_velocity, [ee_yaw, self.v_angle]))
        return observation

    def reset(self) -> None:
        # self._reload_robot()
        self.num_episodes += 1
        print(f"INFO: episode {self.num_episodes}")

        self._set_random_ee_pose()
        self._attach_tool_to_ee()
        # self.set_joint_zero()
        # print(self.get_arm_joint_angles())
        # time.sleep(3)

    def _set_random_ee_pose(self) -> None:
        self.set_joint_neutral() # set neutral first to avoid singularities
        init_ee_position = np.array([0.0, 0.0, 0.2])
        # push forward
        # init_ee_position[0] = np.random.uniform(0.25, 0.35) 
        # init_ee_position[1] = np.random.uniform(-0.2, 0.2)
        # init_ee_euler = [-np.pi, 0,  np.random.uniform(-np.pi/6, np.pi/6)]
        # # push from left side (larger workspace length sideways than upfront. but not wide enough table in robot lab?)
        # init_ee_position[0] = np.random.uniform(0.4, 0.5) 
        # init_ee_position[1] = np.random.uniform(0.5, 0.6)
        # init_ee_euler = [-np.pi, 0,  np.random.uniform(-np.pi/3, -2*np.pi/3)]
        init_ee_position[0] = np.random.uniform(0.3, 0.3) 
        init_ee_position[1] = np.random.uniform(-0.0, 0.0)
        init_ee_euler = [-np.pi, 0,  np.random.uniform(-0*np.pi/6, 0*np.pi/6)]

        init_ee_quaternion = np.array(list(p.getQuaternionFromEuler(init_ee_euler)))
        init_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=init_ee_position, orientation=init_ee_quaternion
        )
        init_arm_angles = init_arm_angles[:7]  # remove fingers angles
        init_arm_angles = np.concatenate((init_arm_angles, [0.0, 0.0])) 
        self.set_joint_angles(init_arm_angles)

    def _attach_tool_to_ee(self) -> None:
        os.system("rm -rf asset/vpusher/*")
        unique_obj_filename = f"v_pusher_{self.v_angle:.3f}.obj"
        tool_obj_path = f"asset/vpusher/{unique_obj_filename}"
        generate_v_shape_pusher(self.finger_length, self.v_angle, self.finger_thickness, self.body_height, tool_obj_path)
        decompose_mesh(pb_connected=True, input_file=tool_obj_path)
        with open("asset/franka_panda_custom/vpusher_template.urdf", 'r') as file:
            self.urdf_content = file.read()
        self.urdf_content = self.urdf_content.replace('{v_pusher_name}', tool_obj_path)

        os.system(f"rm -rf asset/franka_panda_custom/vpusher_modified_*")
        self.modified_vpusher_path = f"asset/franka_panda_custom/vpusher_modified_{self.v_angle:.3f}.urdf"
        with open(self.modified_vpusher_path, 'w') as file:
            file.write(self.urdf_content)

        # Load the tool
        if "tool" in self.sim._bodies_idx:
            p.removeBody(self.sim._bodies_idx["tool"])
        ee_position_center = (self.get_ee_position() + self.get_link_position(self.ee_link-1)) / 2
        ee_position_center[2] -= 0.1
        self.sim.loadURDF(
            body_name="tool",
            fileName=self.modified_vpusher_path,
            basePosition=ee_position_center,
            useFixedBase=0,
        )
        print('ee_position_center',ee_position_center)
        # tool_id = p.loadURDF(self.modified_vpusher_path, basePosition=[0, 0, 0])

        # Set the initial position and orientation of the tool to match the end-effector
        # p.resetBasePositionAndOrientation(tool_id, [0,0,0], [0,0,0,1])

        # Create a fixed constraint to attach the tool to the end-effector
        end_effector_index = 10
        p.createConstraint( # TODO: constraint causes singularity
            parentBodyUniqueId=self.sim._bodies_idx[self.body_name],
            parentLinkIndex=end_effector_index,
            childBodyUniqueId=self.sim._bodies_idx["tool"],
            childLinkIndex=-1,  # -1 means we are attaching to the base of the tool
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0.0, -0.01, -0.015] # align with center of two parallel jaw fingers (z: leave space for handle in real world)
        )

    # def _reload_robot(self) -> None:
    #     print("INFO: Recreating the robot")
    #     os.system("rm -rf asset/vpusher/*")
    #     unique_obj_filename = f"v_pusher_{self.v_angle:.3f}.obj"
    #     generate_v_shape_pusher(self.finger_length, self.v_angle, self.finger_thickness, self.body_height, f"asset/vpusher/{unique_obj_filename}")
    #     decompose_mesh(pb_connected=True, input_file=f"asset/vpusher/{unique_obj_filename}")
    #     with open(self.template_file_name, 'r') as file:
    #         self.urdf_content = file.read()
    #     self.urdf_content = self.urdf_content.replace('{v_pusher_name}', f"asset/vpusher/{unique_obj_filename}")

    #     os.system(f"rm -rf asset/franka_panda_custom/panda_modified_*")
    #     self.modified_file_path = f"asset/franka_panda_custom/panda_modified_{self.v_angle:.3f}.urdf"
    #     with open(self.modified_file_path, 'w') as file:
    #         file.write(self.urdf_content)
    #     with self.sim.no_rendering():
    #         p.removeBody(self.sim._bodies_idx[self.body_name])
    #         self._load_robot(self.modified_file_path, self.base_position)
    #         self.setup()
    
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