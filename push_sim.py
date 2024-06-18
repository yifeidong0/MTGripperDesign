import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import random

class ForwardSimulationPlanePush:
    def __init__(self, task_type='ball', gripper_length=1.0, gui=False):
        self.visualShapeId = -1
        self.gui = gui
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.g = -9.81
        self.task_type = task_type
        self.gripper_length = gripper_length
        self.set_params()
        self.create_shapes()
        # set gravity
        p.setGravity(0, 0, self.g)

    def set_params(self):
        self.mass_object = 1.0
        self.pos_object = [0.6, 0, 0]
        self.quat_object = p.getQuaternionFromEuler([0, 0, 0])
        self.mass_gripper = 1.0
        self.pos_gripper = [0, -1, 0]
        self.quat_gripper = p.getQuaternionFromEuler([0, 0, 0])
        self.lateral_friction_coef = 0.1
        self.z_bodies = 0.05
        self.half_extent_obstacle = [1, 0.5, self.z_bodies]
        self.pos_obstacle = [2, self.half_extent_obstacle[1], 0]
        self.quat_obstacle = p.getQuaternionFromEuler([0, 0, 0])

    def create_shapes(self):
        planeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, self.z_bodies])
        self.planeUid = p.createMultiBody(0, planeId, self.visualShapeId, [0, 0, -2 * self.z_bodies], p.getQuaternionFromEuler([0, 0, 0]))
        p.changeDynamics(self.planeUid, -1, lateralFriction=self.lateral_friction_coef)
        
        if self.task_type == 'box':
            objectId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.8, 0.8, self.z_bodies])
        elif self.task_type == 'ball':
            objectId = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.8, height=2 * self.z_bodies)
        self.objectUid = p.createMultiBody(self.mass_object, objectId, self.visualShapeId, self.pos_object, self.quat_object)
        p.changeDynamics(self.objectUid, -1, lateralFriction=self.lateral_friction_coef)
        
        gripperId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.gripper_length, 0.1, self.z_bodies])
        self.gripperUid = p.createMultiBody(self.mass_gripper, gripperId, self.visualShapeId, self.pos_gripper, self.quat_gripper)
        p.changeDynamics(self.gripperUid, -1, lateralFriction=self.lateral_friction_coef)
        
    def reset_states(self):
        # xo, yo, thetao, vxo, vyo, omegao, xg, yg, thetag, vxg, vyg, omegag = states
        self.set_params()
        p.resetBasePositionAndOrientation(self.objectUid, self.pos_object, self.quat_object)
        p.resetBasePositionAndOrientation(self.gripperUid, self.pos_gripper, self.quat_gripper)

    def run_forward_sim(self, dt = 5):
        for i in range(int(dt * 240)):
            self.pos_object, _ = p.getBasePositionAndOrientation(self.objectUid)
            force_on_object = [self.mass_object * .0 * random.uniform(0.5, 1), 
                               self.mass_object * .3 * random.uniform(0.5, 2), 
                               0]
            p.applyExternalForce(self.gripperUid, -1, force_on_object, [0,0,0], p.LINK_FRAME)
            # p.applyExternalTorque(self.objectUid, -1, torque_on_object, p.WORLD_FRAME)
            p.stepSimulation()
            if self.gui:
                time.sleep(1 / 240)
        reward = p.getBasePositionAndOrientation(self.objectUid)[0][1]
        print("Reward: ", reward)
        self.finish_sim()
        return reward

    def finish_sim(self):
        p.disconnect()

# Main script to invoke the class
if __name__ == "__main__":
    sim = ForwardSimulationPlanePush(task_type='box', gripper_length=0.1, gui=1) # box, ball
    # sim.reset_states()
    reward = sim.run_forward_sim()