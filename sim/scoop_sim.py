import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.generate_scoop import generate_scoop
from utils.vhacd import decompose_mesh

class ScoopingSimulation:
    def __init__(self, object_type='insole', coef=[1, 1], use_gui=True):
        self.coef = coef
        self.object_type = object_type
        self.use_gui = use_gui

        # Connect to PyBullet
        if self.use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -9.81)
        self.time_step = 1.0 / 240.0
        self.plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])

        # Setup camera
        p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=90, cameraPitch=-89.99, cameraTargetPosition=[2.5, 2.5, 0])
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[2.5, 2.5, 0],
                                                              distance=2, yaw=90, pitch=-89.99, roll=0, upAxisIndex=2)
        self.projectionMatrix = p.computeProjectionMatrixFOV(fov=103, aspect=1, nearVal=0.1, farVal=100)

        # Simulation parameters
        self.body_height = 0.2
        self.object_mass = 0.005
        self.robot_mass = 1
        self.goal_radius = 0.5
        self.goal_position = [2.5, 4.5] # workspace [[0,5], [0,5]]
        self.tex = p.loadTexture("uvmap.png")

        self.load_obstacle()
        self.setup(reset_task_and_design=True)

    def setup(self, reset_task_and_design=False, reset_pose=False,):
        """ Setup the simulation environment.
            Args:
                reset_task_and_design (bool): Whether to reset the task and design parameteres (for a new iteration in MTBO).
                reset_pose (bool): Whether to reset the object and robot poses (for a new episode).
        """
        # Set the object and robot poses
        if self.object_type == 'insole':
            self.object_orientation = p.getQuaternionFromEuler([0, 0, random.normalvariate(0, 0.1)])
            self.object_position = [
                                    random.normalvariate(2, .3,), 
                                    random.normalvariate(2.1, .3,), 
                                    self.body_height]
        elif self.object_type == 'pillow':
            self.object_orientation = p.getQuaternionFromEuler([0, 0, random.normalvariate(0, 0.1)]) 
            self.object_position = [
                                    random.normalvariate(3, .3,), 
                                    random.normalvariate(2.5, .3,), 
                                    self.body_height]
        self.robot_position = [random.normalvariate(0.5, .1),
                                random.normalvariate(2.5, .3),
                                0]
        self.robot_orientation = p.getQuaternionFromEuler([0,0,0])

        # Create the object and robot
        if reset_task_and_design:
            if self.object_type == 'insole':
                self.load_insole()
            elif self.object_type == 'pillow':
                self.load_pillow()
            self.create_scoop()

        # Reset the object and robot poses and velocities
        if reset_pose:
            p.resetBasePositionAndOrientation(self.object_id, self.object_position, self.object_orientation)
            p.resetBasePositionAndOrientation(self.robot_id, self.robot_position, self.robot_orientation)
            p.resetBaseVelocity(self.object_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
            p.resetBaseVelocity(self.robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

    def reset_task_and_design(self, new_task, new_design):
        """Reset the task and design parameters for MTBO."""
        self.object_type = new_task
        self.coef = new_design

        # Remove the old object and robot
        p.removeBody(self.object_id)
        p.removeBody(self.robot_id)

        self.setup(reset_task_and_design=True)

    def create_scoop(self,):
        # Remove previous obj and urdf files
        os.system("rm -rf asset/poly_scoop/*")

        # Generate V-shaped pusher obj file
        unique_obj_filename = f"scoop_{self.coef[0]:.2f}_{self.coef[1]:.2f}.obj"
        generate_scoop(self.coef, f"asset/poly_scoop/{unique_obj_filename}")

        # Decompose the mesh
        decompose_mesh(pb_connected=True, input_file=f"asset/poly_scoop/{unique_obj_filename}")

        # Create a new URDF file with the updated OBJ file path
        unique_urdf_filename = f"scoop_{self.coef[0]:.2f}_{self.coef[1]:.2f}.urdf"
        urdf_template = f"""
            <?xml version="1.0" ?>
            <robot name="scoop">
            <link name="baseLink">
                <contact>
                <lateral_friction value="1.0"/>
                <rolling_friction value="0.001"/>
                <restitution value="0.5"/>
                </contact>
                <inertial>
                <origin xyz="0 0 0" rpy="0 0 0"/>
                <mass value="{self.robot_mass}"/>
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

        with open(f"asset/poly_scoop/{unique_urdf_filename}", "w") as urdf_file:
            urdf_file.write(urdf_template)

        self.robot_id = p.loadURDF(f"asset/poly_scoop/{unique_urdf_filename}", basePosition=self.robot_position, baseOrientation=self.robot_orientation)
        # self.robot_id = p.loadURDF(f"asset/scoop.urdf", basePosition=self.robot_position, baseOrientation=self.robot_orientation)

        # Load urdf
        p.changeVisualShape(self.robot_id, -1, rgbaColor=[0, 0, 1, 1])
        p.changeDynamics(self.robot_id, -1, lateralFriction=0.6, spinningFriction=0.4, rollingFriction=0.4)

    def load_insole(self):
        self.object_id = p.loadSoftBody(
            "asset/insole.vtk",
            basePosition=self.object_position,
            baseOrientation=self.object_orientation,
            scale=4,
            mass=self.object_mass,
            useNeoHookean=1,
            NeoHookeanMu=1.5,
            NeoHookeanLambda=1,
            NeoHookeanDamping=0.5,
            useSelfCollision=1,
            frictionCoeff=0.5,
            collisionMargin=0.001
        )
        p.changeVisualShape(self.object_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=self.tex, flags=0)

    def load_pillow(self):
        self.object_id = p.loadSoftBody(
            "asset/pillow.vtk",
            basePosition=self.object_position,
            baseOrientation=self.object_orientation,
            scale=2.5,
            mass=self.object_mass,
            useNeoHookean=1,
            NeoHookeanMu=0.5,
            NeoHookeanLambda=1,
            NeoHookeanDamping=0.5,
            useSelfCollision=1,
            frictionCoeff=0.5,
            collisionMargin=0.001
        )
        p.changeVisualShape(self.object_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=self.tex, flags=0)

    def load_obstacle(self):
        # Create a heavy box to fix the insole at the other end
        box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 1.8, 0.1])
        self.obstacle = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=box,
            basePosition=[5, 2.5, 0.05],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
    def eval_robustness(self, slack=0.1):
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

    def run_onetime(self, rob_eval_freq=10):
        i = 0
        avg_robustness = 0
        target_reached = False
        while True:
            if i % 100 == 0:
                print(f"Step {i}")

            # Get current position of the object and robot
            self.object_position = np.array(p.getBasePositionAndOrientation(self.object_id)[0])
            self.robot_position = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])

            # # Check if the object is in the goal region
            # distance_to_goal = np.linalg.norm(self.object_position[:2] - np.array(self.goal_position))
            # if distance_to_goal < self.goal_radius: # TODO: height check
            #     target_reached = True
            #     break

            # Apply velocity on scoop
            if i > 100 and i < 1000:
                p.applyExternalForce(self.robot_id, -1, [7, 0, 0], self.robot_position, p.WORLD_FRAME)
            
            # Lift the scoop
            if i > 1000 and i < 1300:
                p.applyExternalForce(self.robot_id, -1, [0, 0, 11], self.robot_position, p.WORLD_FRAME)

            # Fix the scoop in the air
            if i > 1300:
                p.changeDynamics(self.robot_id, -1, mass=0)
                p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])

            p.stepSimulation()
            time.sleep(self.time_step)
            i += 1

            # Evaluate robustness
            if i % rob_eval_freq == 0:
                rob = self.eval_robustness(slack=1)
                avg_robustness += rob

            # Break condition for the simulation
            if (i >= 1500 
                or (self.robot_position[0]>5 or self.robot_position[0]<0 or self.robot_position[1]>6 or self.robot_position[1]<0)
                or (self.object_position[0]>5 or self.object_position[0]<0 or self.object_position[1]>6 or self.object_position[1]<0)
                ):
                break

            # Get camera image (time consuming, only for visualization purpose)
            # p.getCameraImage(320, 320, viewMatrix=self.viewMatrix, projectionMatrix=self.projectionMatrix)

        # Final score calculation
        avg_robustness = 0 if i == 0 else avg_robustness / (i // rob_eval_freq)
        final_score = 1.0 if target_reached else 0.0
        final_score += avg_robustness

        return final_score

if __name__ == "__main__":
    coef = [1, 1]
    simulation = ScoopingSimulation('pillow', coef=coef, use_gui=True)  # insole or pillow
    for i in range(3):
        print('Iteration %d' % i)
        final_score = simulation.run(1)
        print("Final Score:", final_score)

        # randomly select insole or pillow and new design parameters
        coef = [random.uniform(.5, 5), 
                random.uniform(0.2,1.3), ]
        if random.random() < 0.5:
            simulation.reset_task_and_design('insole', coef,)
        else:
            simulation.reset_task_and_design('pillow', coef,)
