import pybullet as p
import pybullet_data
import time
import numpy as np

"""Pybullet environment for pushing a custom object with a V-shap gripper. """
class CustomObjectSimulation:
    def __init__(self, angle_theta=np.pi/4, gui=False):
        self.angle_theta = angle_theta
        self.gui = gui
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.setGravity(0, 0, -9.81)
        self.visualShapeId = -1
        self.lateral_friction_coef = 0.1
        # add floor
        p.createCollisionShape(p.GEOM_PLANE)

    def create_custom_object(self):

        # Define dimensions
        length = 1.0
        thickness = 0.1

        # Create collision shapes for two thin boxes
        box_shape_id1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, thickness/2, thickness/2])
        box_shape_id2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, thickness/2, thickness/2])

        # Create multi bodies for the two boxes
        mass = 1.0
        box1_uid = p.createMultiBody(mass, box_shape_id1, self.visualShapeId, [0, 0, 0], [0, 0, 0, 1])
        box2_uid = p.createMultiBody(mass, box_shape_id2, self.visualShapeId, [0, 0, 0], [0, 0, 0, 1])

        # Calculate the positions and orientations based on angle_theta
        half_angle = self.angle_theta / 2
        pos1 = [length/2 * np.cos(half_angle)+thickness/2, length/2 * np.sin(half_angle)+thickness/2, 0]
        pos2 = [length/2 * np.cos(-half_angle), length/2 * np.sin(-half_angle), 0]

        orn1 = p.getQuaternionFromEuler([0, 0, half_angle])
        orn2 = p.getQuaternionFromEuler([0, 0, -half_angle])

        # Set the positions and orientations of the two boxes
        p.resetBasePositionAndOrientation(box1_uid, pos1, orn1)
        p.resetBasePositionAndOrientation(box2_uid, pos2, orn2)

        # Create constraint to connect the two boxes at one end
        constraint_id = p.createConstraint(
            parentBodyUniqueId=box1_uid,
            parentLinkIndex=-1,
            childBodyUniqueId=box2_uid,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[-length/2, 0, 0],
            childFramePosition=[-length/2, 0, 0],
            parentFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            childFrameOrientation=p.getQuaternionFromEuler([0, 0, self.angle_theta])
        )

        p.changeDynamics(box1_uid, -1, lateralFriction=self.lateral_friction_coef)
        p.changeDynamics(box2_uid, -1, lateralFriction=self.lateral_friction_coef)

    def run_simulation(self, steps=10000):
        self.create_custom_object()
        for _ in range(steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1./240.)
        p.disconnect()

if __name__ == "__main__":
    theta = np.pi / 2  # Example angle in radians
    sim = CustomObjectSimulation(angle_theta=theta, gui=True)
    sim.run_simulation()
