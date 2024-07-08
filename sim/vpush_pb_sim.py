import pybullet as p
import pybullet_data
import numpy as np
import math
import time
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.generate_vpush import generate_v_shape_pusher, decompose_mesh
from shapely.geometry import Polygon, Point

class VPushPbSimulation:
    def __init__(self, object_type='circle', v_angle=np.pi/3, use_gui=True):
        self.v_angle = v_angle
        self.object_type = object_type
        self.use_gui = use_gui
        self.setup()

    def setup(self):
        if self.use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Create the dynamic object
        self.body_height = 0.1
        self.object_position = [random.normalvariate(2.5, 0.3), 
                                random.normalvariate(2.5, 0.3), 
                                self.body_height / 2]
        self.object_rad = 0.2
        if self.object_type == 'circle':
            self.object_id = self.create_circle(self.object_position, radius=self.object_rad, height=self.body_height)
        elif self.object_type == 'polygon':
            self.object_id = self.create_polygon(self.object_position, size=self.object_rad, height=self.body_height)

        # Create the goal region
        self.goal_radius = 0.5
        self.goal_position = [4.5, 2.5] # workspace [[0,5], [0,5]]

        # Create a dynamic V-shaped robot
        self.robot_position = [random.normalvariate(1, .3), 
                               random.normalvariate(2.5, .3), 
                               self.body_height / 2]
        self.robot_orientation = p.getQuaternionFromEuler([0, 0, random.normalvariate(0, math.pi / 6)])
        self.finger_length = 0.8
        self.finger_thickness = 0.1
        self.robot_id = self.create_v_shape(self.robot_position, angle=self.v_angle, 
                                            length=self.finger_length, thickness=self.finger_thickness, 
                                            height=self.body_height)

        # Simulation parameters
        self.time_step = 1.0 / 240.0

        # Setup camera
        p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=90, cameraPitch=-89.99, cameraTargetPosition=[2.5, 2.5, 0])
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[2.5, 2.5, 0], 
                                                              distance=2, yaw=90, pitch=-89.99, roll=0, upAxisIndex=2)
        self.projectionMatrix = p.computeProjectionMatrixFOV(fov=103, aspect=1, nearVal=0.1, farVal=100)

    def reset_task_and_design(self, new_task, new_design):
        # Reset the design and task
        self.object_type = new_task
        self.v_angle = new_design
        self.setup()

    def reset(self, min_distance=0.5):
        self.object_position = [random.normalvariate(3, 0.5),
                                random.normalvariate(2.5, 0.5),
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
        
        p.resetBasePositionAndOrientation(self.object_id, self.object_position, self.object_orientation)
        p.resetBasePositionAndOrientation(self.robot_id, self.robot_position, [0, 0, 0, 1])
        p.resetBaseVelocity(self.robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        p.resetBaseVelocity(self.object_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

    def create_circle(self, position, radius, height):
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=radius, length=height, visualFramePosition=[0, 0, 0])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=radius, height=height, collisionFramePosition=[0, 0, 0])
        body_id = p.createMultiBody(baseMass=.1, baseInertialFramePosition=[0, 0, 0], baseCollisionShapeIndex=collision_shape_id,
                                    baseVisualShapeIndex=visual_shape_id, basePosition=position, baseOrientation=[0, 0, 0, 1])
        p.changeDynamics(body_id, -1, lateralFriction=1)
        p.changeVisualShape(body_id, -1, rgbaColor=[1, 0, 0, 1])
        return body_id

    def create_polygon(self, position, size, height):
        half_extents = [size, size, height / 2]
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extents)
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extents)
        body_id = p.createMultiBody(baseMass=.1, baseInertialFramePosition=[0, 0, 0], baseCollisionShapeIndex=collision_shape_id,
                                    baseVisualShapeIndex=visual_shape_id, basePosition=position, baseOrientation=[0, 0, 0, 1])
        p.changeDynamics(body_id, -1, lateralFriction=1)
        p.changeVisualShape(body_id, -1, rgbaColor=[1, 0, 0, 1])
        return body_id

    def create_v_shape(self, position, angle, length=0.8, thickness=0.1, height=0.1):
        # Generate V-shaped pusher obj file
        generate_v_shape_pusher(length, angle, thickness, height)

        # Decompose the mesh
        decompose_mesh(pb_connected=True)

        # Load urdf
        body_id = p.loadURDF("asset/vpusher/v_pusher.urdf", basePosition=position, baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
        p.changeVisualShape(body_id, -1, rgbaColor=[0, 0, 1, 1])
        
        return body_id

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
            self.reset()
        avg_score /= num_episodes
        return avg_score

    def run_onetime(self):
        running = True
        target_reached = False
        num_steps = 0
        avg_robustness = 0
        while running:
            object_pos = np.array(p.getBasePositionAndOrientation(self.object_id)[0])
            distance_to_goal = np.linalg.norm(object_pos[:2] - np.array(self.goal_position))
            if distance_to_goal < self.goal_radius:
                target_reached = True
                break

            # Push object towards goal using velocity control with noise
            velocity = 0.1 * np.array([1,0]) * (distance_to_goal + 2 * self.goal_radius) + np.random.normal(0, 0.05, 2)
            p.resetBaseVelocity(self.robot_id, linearVelocity=[velocity[0], velocity[1], 0])

            # Step the simulation
            p.stepSimulation()
            time.sleep(self.time_step)

            # Evaluate robustness
            rob = self.eval_robustness(slack=self.object_rad)
            # Record average robustness every 10 steps
            if num_steps % 10 == 0:
                avg_robustness += rob

            num_steps += 1
            if num_steps >= 3000:
                break

            # Get camera image
            p.getCameraImage(320, 320, viewMatrix=self.viewMatrix, projectionMatrix=self.projectionMatrix)

        avg_robustness = 0 if num_steps == 0 else avg_robustness / (num_steps // 10)
        final_score = 1 if target_reached else 0
        final_score += avg_robustness * 0.1
        print("Final Score:", final_score)

        return final_score

if __name__ == "__main__":
    simulation = VPushPbSimulation('polygon', 1, use_gui=True)  # polygon or circle
    final_score = simulation.run(3)
    print("Final Score:", final_score)
