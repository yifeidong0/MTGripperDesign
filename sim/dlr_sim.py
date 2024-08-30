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
from utils.change_parameter_dlr_urdf import change_parameter_lsc_urdf

class DLRSimulation:
    def __init__(self, object_type='cube', task_param=0.1, design_params=[60, 60], use_gui=True):
        self.design_params = design_params
        self.object_type = object_type
        self.task_param = task_param
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
        p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0,0,0])
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0,0,0], 
                                                              distance=2, yaw=45, pitch=-45, roll=0, upAxisIndex=2)
        self.projectionMatrix = p.computeProjectionMatrixFOV(fov=45, aspect=1, nearVal=0.1, farVal=100)

        self.setup(reset_task_and_design=True)

    def setup(self, reset_task_and_design=False, reset_pose=False,):
        """ Setup the simulation environment.
            Args:
                reset_task_and_design (bool): Whether to reset the task and design parameteres (for a new iteration in MTBO).
                reset_pose (bool): Whether to reset the object and robot poses (for a new episode).
        """
        # Set the object and robot poses
        if self.object_type == 'fish':
            self.object_orientation = p.getQuaternionFromEuler([0, 0, 0])
            self.object_position = [0,-0.8,0.35]
        elif self.object_type == 'cube':
            self.object_orientation = p.getQuaternionFromEuler([0, 0, 0])
            self.object_position = [0,0,self.task_param]
        self.robot_position = [0,0,4]
        self.robot_orientation = p.getQuaternionFromEuler([math.pi,0,0])

        # Create the object and robot
        if reset_task_and_design:
            if self.object_type == 'fish':
                self.load_fish()
            elif self.object_type == 'cube':
                self.load_cube()
            self.create_gripper()

        # Reset the object and robot poses and velocities
        if reset_pose:
            p.resetBasePositionAndOrientation(self.object_id, self.object_position, self.object_orientation)
            p.resetBasePositionAndOrientation(self.robot_id, self.robot_position, self.robot_orientation)
            p.resetBaseVelocity(self.object_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
            p.resetBaseVelocity(self.robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

    def reset_task_and_design(self, new_task, new_task_param, new_design):
        """Reset the task and design parameters for MTBO."""
        self.object_type = new_task
        self.design_params = new_design
        self.task_param = new_task_param

        # Remove the old object and robot
        p.removeBody(self.object_id)
        p.removeBody(self.robot_id)

        self.setup(reset_task_and_design=True)

    def create_gripper(self,):
        # Remove previous obj and urdf files
        os.system("rm -rf asset/lc_soft_enable_wide_grip/lc_soft_enable_wide_grip_*")

        template_urdf = 'asset/lc_soft_enable_wide_grip/lc_soft_enable_wide_grip.urdf'
        with open(template_urdf, 'r') as file:
            urdf_content = file.read()

        unique_urdf_filename = f"asset/lc_soft_enable_wide_grip/lc_soft_enable_wide_grip_{self.design_params[0]:.0f}_{self.design_params[1]:.0f}.urdf"
        with open(unique_urdf_filename, 'w') as file:
            file.write(urdf_content)
        change_parameter_lsc_urdf(self.design_params[0], self.design_params[1],
                                  urdf_path=unique_urdf_filename)

        self.robot_id = p.loadURDF(fileName=unique_urdf_filename,
                                basePosition=self.robot_position,
                                baseOrientation=self.robot_orientation, # p.getQuaternionFromEuler([1*math.pi,0,0]),
                                globalScaling=10,
                                useFixedBase=1,
                                flags=p.URDF_USE_SELF_COLLISION, # | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
                                )

    def load_fish(self):
        self.object_id = p.loadURDF(
            fileName='asset/fine-fish-10/fine-fish-10.urdf',
            basePosition=self.object_position,
            baseOrientation=self.object_orientation,
            globalScaling=.3,
            useFixedBase=0)
        # p.changeVisualShape(self.object_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=self.tex, flags=0)
    
    def load_cube(self):
        """Create from collision shape """
        self.object_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.task_param,]*3
        )
        self.object_id = p.createMultiBody(
            baseMass=.1,
            baseCollisionShapeIndex=self.object_id,
            basePosition=self.object_position,
            baseOrientation=self.object_orientation
        )
        p.changeDynamics(self.object_id, -1, lateralFriction=0.8)

    def eval_robustness(self, height_threshold, acc_lim=100):
        steps_on_the_scoop = 0
        object_initial_position = p.getBasePositionAndOrientation(self.object_id)[0]
        object_initial_orientation = p.getBasePositionAndOrientation(self.object_id)[1]
        object_initial_velocity = p.getBaseVelocity(self.object_id)[0]
        object_initial_angular_velocity = p.getBaseVelocity(self.object_id)[1]

        # Apply random acceleration/force to the object
        while True:
            rand_acc = [random.uniform(-acc_lim, acc_lim) for _ in range(3)]
            total_acc = rand_acc
            total_acc[2] = self.g
            delta_vel = [a * self.time_step for a in total_acc]
            curr_vel = p.getBaseVelocity(self.object_id)[0]
            new_vel = [v + dv for v, dv in zip(curr_vel, delta_vel)]
            p.resetBaseVelocity(self.object_id, new_vel, [0, 0, 0])
            p.stepSimulation()
            # time.sleep(self.time_step)

            object_position = p.getBasePositionAndOrientation(self.object_id)[0]
            if object_position[2] > height_threshold:
                steps_on_the_scoop += 1
            else:
                break
        
        # Reset the object to the initial state
        p.resetBasePositionAndOrientation(self.object_id, object_initial_position, object_initial_orientation)
        p.resetBaseVelocity(self.object_id, object_initial_velocity, object_initial_angular_velocity)
        return steps_on_the_scoop
    
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
            if i > 1000:
                break

            p.stepSimulation()
            if self.use_gui:
                time.sleep(self.time_step)
            i += 1

        # Final score calculation
        final_score = 1.0 if target_reached else 0.0
        final_score += self.steps_on_the_scoop / 10000.0

        return final_score

if __name__ == "__main__":
    distal_lengths = np.arange(20, 65, 5)
    distal_curvatures = np.arange(2, 10, 2)
    design_params = [random.choice(distal_lengths),
                     random.choice(distal_curvatures),]
    simulation = DLRSimulation('cube', 0.1, design_params, use_gui=1)
    for i in range(5):
        print('Iteration %d' % i)
        final_score = simulation.run(1)
        print("Final Score:", final_score)

        # randomly select insole or pillow and new design parameters
        design_params = [random.choice(distal_lengths),
                         random.choice(distal_curvatures),]
        simulation.reset_task_and_design('cube', 0.1, design_params,) #cube
