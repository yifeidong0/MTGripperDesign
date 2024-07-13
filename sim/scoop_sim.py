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
    def __init__(self, object_type='bread', coef=[1, 1], use_gui=True):
        self.coef = coef
        self.object_type = object_type
        self.use_gui = use_gui
        self.g = -9.81

        # Connect to PyBullet
        if self.use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, self.g)
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
        self.hover_height = 1.
        self.reached_height = 0.8 * self.hover_height
        self.unsafe_height = 0.7 * self.hover_height

        self.load_obstacle()
        self.setup(reset_task_and_design=True)

    def setup(self, reset_task_and_design=False, reset_pose=False,):
        """ Setup the simulation environment.
            Args:
                reset_task_and_design (bool): Whether to reset the task and design parameteres (for a new iteration in MTBO).
                reset_pose (bool): Whether to reset the object and robot poses (for a new episode).
        """
        # Set the object and robot poses
        if self.object_type == 'bread':
            self.object_orientation = p.getQuaternionFromEuler([math.pi/2, 0, math.pi/2+random.normalvariate(0, 0.1)])
            self.object_position = [random.normalvariate(3, .3,), 
                                    random.normalvariate(2.5, .1,), 
                                    self.body_height]
        elif self.object_type == 'pillow':
            self.object_orientation = p.getQuaternionFromEuler([0, 0, random.normalvariate(0, 0.1)]) 
            self.object_position = [random.normalvariate(3, .3,), 
                                    random.normalvariate(2.5, .1,), 
                                    self.body_height]
        self.robot_position = [random.normalvariate(0.5, .1),
                               random.normalvariate(2.5, .1),
                               0]
        self.robot_orientation = p.getQuaternionFromEuler([0,0,random.normalvariate(0, 0.05)])

        # Create the object and robot
        if reset_task_and_design:
            if self.object_type == 'bread':
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
        decompose_mesh(pb_connected=True, input_file=f"asset/poly_scoop/{unique_obj_filename}", timeout=2)

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
        # self.robot_id = p.loadURDF(f"asset/scoop_1.00_1.00.urdf", basePosition=self.robot_position, baseOrientation=self.robot_orientation)

        # Load urdf
        p.changeVisualShape(self.robot_id, -1, rgbaColor=[0, 0, 1, 1])
        p.changeDynamics(self.robot_id, -1, lateralFriction=0.1, spinningFriction=0.1, rollingFriction=0.1)

    def load_insole(self):
        self.object_id = p.loadSoftBody(
            "asset/bread.vtk",
            basePosition=self.object_position,
            baseOrientation=self.object_orientation,
            scale=.5,
            mass=self.object_mass,
            useNeoHookean=1,
            NeoHookeanMu=5,
            NeoHookeanLambda=5,
            NeoHookeanDamping=0.1,
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
            NeoHookeanMu=5,
            NeoHookeanLambda=5,
            NeoHookeanDamping=0.1,
            useSelfCollision=1,
            frictionCoeff=0.5,
            collisionMargin=0.001
        )
        p.changeVisualShape(self.object_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=self.tex, flags=0)

    def load_obstacle(self):
        # Create a heavy box to fix the bread at the other end
        box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 1.8, 0.3])
        self.obstacle = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=box,
            basePosition=[5, 2.5, 0.05],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
    def eval_robustness(self, dt=1./240., acc_lim=1000):
        rand_acc = [random.uniform(-acc_lim, acc_lim) for _ in range(3)]
        total_acc = rand_acc
        total_acc[2] = self.g
        delta_vel = [a * dt for a in total_acc]
        curr_vel = p.getBaseVelocity(self.object_id)[0]
        new_vel = [v + dv for v, dv in zip(curr_vel, delta_vel)]
        p.resetBaseVelocity(self.object_id, new_vel, [0, 0, 0])

        if self.object_position[2] > self.unsafe_height:
            self.steps_on_the_scoop += 1
        else:
            self.done_evaluation = True
        
        return self.done_evaluation
    
    def run(self, num_episodes=1):
        avg_score = 0
        for i in range(num_episodes):
            score = self.run_onetime()
            print('Episode %d: %.2f' % (i, score))
            avg_score += score
            self.setup(reset_pose=True)
        avg_score /= num_episodes
        return avg_score

    def run_onetime(self,):
        i = 0
        target_reached = False
        pause_and_check = False
        self.steps_on_the_scoop = 0
        self.done_evaluation = False

        while True:
            if i % 500 == 0:
                print(f"Step {i}")

            # Get current position of the object and robot
            self.object_position = np.array(p.getBasePositionAndOrientation(self.object_id)[0])
            self.robot_position = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])

            # Apply velocity on scoop
            if i < 100:
                pass
            elif i >= 100 and i < 500:
                p.applyExternalForce(self.robot_id, -1, [7, 0, 0], self.robot_position, p.WORLD_FRAME)
            
            elif i >= 500:
                if self.robot_position[2] < self.hover_height: # Lift the scoop
                    p.applyExternalForce(self.robot_id, -1, [0, 0, 11], self.robot_position, p.WORLD_FRAME)
                elif not pause_and_check: # Fix the scoop in the air
                    p.changeDynamics(self.robot_id, -1, mass=0)
                    p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
                    target_reached = True if self.object_position[2] > self.reached_height else False
                    pause_and_check = True
                elif not self.done_evaluation:
                    self.eval_robustness()

            p.stepSimulation()
            if self.use_gui:
                time.sleep(self.time_step)
            i += 1

            # Break if out of bounds or done evaluation
            if ((self.robot_position[0]>6 or self.robot_position[0]<0 or self.robot_position[1]>6 or self.robot_position[1]<0)
                or (self.object_position[0]>6 or self.object_position[0]<0 or self.object_position[1]>6 or self.object_position[1]<0)
                or self.done_evaluation):
                break

            # Get camera image (time consuming, only for visualization purpose)
            # p.getCameraImage(320, 320, viewMatrix=self.viewMatrix, projectionMatrix=self.projectionMatrix)

        # Final score calculation
        final_score = 1.0 if target_reached else 0.0
        final_score += self.steps_on_the_scoop / 10000.0

        return final_score

if __name__ == "__main__":
    coef = [random.uniform(.5, 2), 
            random.uniform(0.2,1.3),]
    simulation = ScoopingSimulation('pillow', coef=coef, use_gui=1)  # bread or pillow
    for i in range(3):
        print('Iteration %d' % i)
        final_score = simulation.run(1)
        print("Final Score:", final_score)

        # randomly select bread or pillow and new design parameters
        coef = [random.uniform(.5, 2), 
                random.uniform(0.2,1.3),]
        if random.random() < 0.5:
            simulation.reset_task_and_design('bread', coef,)
        else:
            simulation.reset_task_and_design('pillow', coef,)
