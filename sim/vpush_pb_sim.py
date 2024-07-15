import pybullet as p
import pybullet_data
import numpy as np
import math
import time
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.generate_vpush import generate_v_shape_pusher
from utils.vhacd import decompose_mesh
from shapely.geometry import Polygon, Point

class VPushPbSimulation:
    def __init__(self, object_type='circle', v_angle=np.pi/3, use_gui=True):
        self.v_angle = v_angle
        self.object_type = object_type
        self.use_gui = use_gui

        # Connect to PyBullet
        if self.use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        self.plane_id = p.loadURDF("plane.urdf")

        # Setup camera
        p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=90, cameraPitch=-89.99, cameraTargetPosition=[2.5, 2.5, 0])
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[2.5, 2.5, 0], 
                                                              distance=2, yaw=90, pitch=-89.99, roll=0, upAxisIndex=2)
        self.projectionMatrix = p.computeProjectionMatrixFOV(fov=103, aspect=1, nearVal=0.1, farVal=100)

        self.finger_length = 0.8
        self.finger_thickness = 0.1
        self.body_height = 0.1
        self.object_rad = 0.2
        self.object_mass = 0.05
        self.goal_radius = 0.5
        self.goal_position = [4.5, 2.5] # workspace [[0,5], [0,5]]
        self.object_position = None
        self.object_orientation = None
        self.robot_position = None
        self.robot_orientation = None
        self.robot_id = None
        self.object_id = None

    def setup(self, reset_task_and_design=False, reset_pose=False, min_distance=1.1):
        """ Setup the simulation environment.
            Args:
                reset_task_and_design (bool): Whether to reset the task and design parameteres (for a new iteration in MTBO).
                reset_pose (bool): Whether to reset the object and robot poses (for a new episode).
                min_distance (float): Minimum distance between object and robot.
        """
        # Set the object and robot poses
        self.object_position = [random.normalvariate(2.5, 0.3), 
                                random.normalvariate(2.5, 0.3), 
                                self.body_height / 2]
        self.object_orientation = p.getQuaternionFromEuler([0, 0, random.uniform(0, 2 * math.pi)])

        # Minimum distance to ensure no penetration
        while True:
            self.robot_position = [random.normalvariate(1, .3),
                                    random.normalvariate(2.5, .3),
                                    self.body_height / 2]
            distance = np.linalg.norm(np.array(self.object_position[:2]) - np.array(self.robot_position[:2]))
            if distance >= min_distance:
                break
        self.robot_orientation = p.getQuaternionFromEuler([0, 0, random.normalvariate(0, math.pi / 6)])

        # Create the object and robot
        if reset_task_and_design:
            if self.object_type == 'circle':
                self.object_id = self.create_circle()
            elif self.object_type == 'polygon':
                self.object_id = self.create_polygon()
            self.create_v_shape()

        if reset_pose:
            p.resetBasePositionAndOrientation(self.object_id, self.object_position, self.object_orientation)
            p.resetBasePositionAndOrientation(self.robot_id, self.robot_position, self.robot_orientation)
            p.resetBaseVelocity(self.robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
            p.resetBaseVelocity(self.object_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

    def reset_task_and_design(self, new_task, new_design):
        """Reset the task and design parameters for MTBO."""
        self.object_type = new_task
        self.v_angle = new_design
        
        # Remove the old object and robot
        if self.object_id is not None:
            p.removeBody(self.object_id)
        if self.robot_id is not None:
            p.removeBody(self.robot_id)

        self.setup(reset_task_and_design=True)

    def create_circle(self,):
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=self.object_rad, length=self.body_height, visualFramePosition=[0, 0, 0])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=self.object_rad, height=self.body_height, collisionFramePosition=[0, 0, 0])
        body_id = p.createMultiBody(baseMass=self.object_mass, baseInertialFramePosition=[0, 0, 0], baseCollisionShapeIndex=collision_shape_id,
                                    baseVisualShapeIndex=visual_shape_id, basePosition=self.object_position, baseOrientation=self.object_orientation)
        p.changeDynamics(body_id, -1, lateralFriction=1)
        p.changeVisualShape(body_id, -1, rgbaColor=[1, 0, 0, 1])
        return body_id

    def create_polygon(self,):
        half_extents = [self.object_rad, self.object_rad, self.body_height / 2]
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extents)
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extents)
        body_id = p.createMultiBody(baseMass=self.object_mass, baseInertialFramePosition=[0, 0, 0], baseCollisionShapeIndex=collision_shape_id,
                                    baseVisualShapeIndex=visual_shape_id, basePosition=self.object_position, baseOrientation=self.object_orientation)
        p.changeDynamics(body_id, -1, lateralFriction=1)
        p.changeVisualShape(body_id, -1, rgbaColor=[1, 0, 0, 1])
        return body_id

    def create_v_shape(self,):
        # Remove previous obj and urdf files
        os.system("rm -rf asset/vpusher/*")

        # Generate V-shaped pusher obj file
        unique_obj_filename = f"v_pusher_{self.v_angle:.3f}.obj"
        generate_v_shape_pusher(self.finger_length, self.v_angle, self.finger_thickness, self.body_height, f"asset/vpusher/{unique_obj_filename}")

        # Decompose the mesh
        decompose_mesh(pb_connected=True, input_file=f"asset/vpusher/{unique_obj_filename}")

        # Create a new URDF file with the updated OBJ file path
        unique_urdf_filename = f"v_pusher_{self.v_angle:.3f}.urdf"
        urdf_template = f"""
            <?xml version="1.0" ?>
            <robot name="v_pusher">
            <link name="baseLink">
                <contact>
                <lateral_friction value="1.0"/>
                <rolling_friction value="0.001"/>
                <restitution value="0.5"/>
                </contact>
                <inertial>
                <origin xyz="0 0 0" rpy="0 0 0"/>
                <mass value="10"/>
                <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
                </inertial>
                <visual>
                <origin xyz="0 0 0" rpy="0 0 0"/>
                <geometry>
                    <mesh filename="package://{unique_obj_filename}" scale="1 1 1"/>
                </geometry>
                <material name="white">
                    <color rgba="1 1 1 1"/>
                </material>
                </visual>
                <collision>
                <origin xyz="0 0 0" rpy="0 0 0"/>
                <geometry>
                    <mesh filename="package://{unique_obj_filename}" scale="1 1 1"/>
                </geometry>
                </collision>
            </link>
            </robot>
        """

        with open(f"asset/vpusher/{unique_urdf_filename}", "w") as urdf_file:
            urdf_file.write(urdf_template)

        self.robot_id = p.loadURDF(f"asset/vpusher/{unique_urdf_filename}", basePosition=self.robot_position, baseOrientation=self.robot_orientation)

        # Load urdf
        p.changeVisualShape(self.robot_id, -1, rgbaColor=[0, 0, 1, 1])
        
    def is_point_inside_polygon(self, point, vertices, slack=2):
        polygon = Polygon(vertices)
        point = Point(point)
        if polygon.contains(point):
            return True
        if polygon.distance(point) <= slack:
            return True

        return False
        
    def eval_robustness(self, slack=0.1):
        # Calculate the object position in the robot frame
        object_pos = np.array(p.getBasePositionAndOrientation(self.object_id)[0][:2])
        robot_pos = np.array(p.getBasePositionAndOrientation(self.robot_id)[0][:2])
        robot_angle = p.getBasePositionAndOrientation(self.robot_id)[1]
        object_pos_R = object_pos - robot_pos
        robot_angle = p.getEulerFromQuaternion(robot_angle)[2]
        object_pos_R = np.array([ # robot frame
            object_pos_R[0] * math.cos(robot_angle) + object_pos_R[1] * math.sin(robot_angle),
            -object_pos_R[0] * math.sin(robot_angle) + object_pos_R[1] * math.cos(robot_angle)
        ])

        # Compute robot vertices position in the robot frame
        self.robot_vertices = [(0, 0),  # robot frame
                               (self.finger_length*math.cos(self.v_angle/2), self.finger_length*math.sin(self.v_angle/2)),
                               (self.finger_length*math.cos(-self.v_angle/2), self.finger_length*math.sin(-self.v_angle/2))]
        
        if self.is_point_inside_polygon(object_pos_R, self.robot_vertices, slack=slack):
            soft_fixture_metric = self.finger_length*math.cos(self.v_angle/2) - object_pos_R[0] + self.object_rad
            robustness = max(0.0, soft_fixture_metric)
        else:
            robustness = 0

        return robustness

    def run(self, num_episodes=1):
        avg_score = 0
        for i in range(num_episodes):
            score = self.run_onetime()
            print('Episode %d: %.2f' % (i, score))
            avg_score += score
            self.setup(reset_pose=True)
        avg_score /= num_episodes
        return avg_score

    def run_onetime(self, rob_eval_freq=50):
        target_reached = False
        num_steps = 0
        avg_robustness = 0
        while True:
            object_pos = np.array(p.getBasePositionAndOrientation(self.object_id)[0][:2])
            robot_pos = np.array(p.getBasePositionAndOrientation(self.robot_id)[0][:2])
            distance_to_goal = np.linalg.norm(object_pos - np.array(self.goal_position))
            if distance_to_goal < self.goal_radius:
                target_reached = True
                break

            # Push object towards goal using velocity control with noise. TODO: replace with a RL policy
            velocity = 0.5 * np.array([1,0]) * (distance_to_goal + 2 * self.goal_radius) + np.random.normal(0, 0.05, 2)
            p.resetBaseVelocity(self.robot_id, linearVelocity=[velocity[0], velocity[1], 0])

            # Step the simulation
            p.stepSimulation()
            # time.sleep(self.time_step)

            # Evaluate robustness
            # Record average robustness every rob_eval_freq steps
            if num_steps % rob_eval_freq == 0:
                rob = self.eval_robustness(slack=self.object_rad)
                avg_robustness += rob

            num_steps += 1
            if num_steps >= 1000 or robot_pos[0]>5 or robot_pos[0]<0 or robot_pos[1]>5 or robot_pos[1]<0:
                break

            # Get camera image
            p.getCameraImage(320, 320, viewMatrix=self.viewMatrix, projectionMatrix=self.projectionMatrix)

        avg_robustness = 0 if num_steps == 0 else avg_robustness / (num_steps // rob_eval_freq)
        final_score = 1. if target_reached else 0.
        final_score += avg_robustness

        return final_score

if __name__ == "__main__":
    simulation = VPushPbSimulation('polygon', random.uniform(0, math.pi), use_gui=True)  # polygon or circle
    simulation.setup(reset_task_and_design=True, reset_pose=True)
    for i in range(3):
        final_score = simulation.run(1)
        print("Final Score:", final_score)

        # randomly select circle or polygon
        if random.random() < 0.5:
            simulation.reset_task_and_design('polygon', random.uniform(0, math.pi))
        else:
            simulation.reset_task_and_design('circle', random.uniform(0, math.pi))

