import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import random

# Physics simulation setup
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeUid = p.loadURDF("plane.urdf", basePosition=[0,0,0])
g = 9.81

# m_box = .1
# for i in range(1):
#     boxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.1,.1,.1,])
#     boxId = p.createMultiBody(m_box, boxId, -1, np.array([0, 0, 0.6]))    

fishid = p.loadURDF(fileName='asset/fine-fish-10/fine-fish-10.urdf',
                    basePosition=[0,-.5,0.7],
                    baseOrientation=p.getQuaternionFromEuler([0,0,0]),
                    globalScaling=.3,
                    useFixedBase=0)

gripperId = p.loadURDF(fileName='asset/lc_soft_enable_wide_grip/lc_soft_enable_wide_grip.urdf', 
                        basePosition=[0,0,2.6], 
                        baseOrientation=p.getQuaternionFromEuler([1*math.pi,0,0]),
                        globalScaling=10,
                        useFixedBase=1
                        )

pivotInBox = [0, 0, 0] 
pivotInWorld = [0, 0, 0]
constraint1 = p.createConstraint(gripperId, -1, -1, -1, p.JOINT_PRISMATIC, [0, 0, 1], pivotInBox, pivotInWorld, p.getQuaternionFromEuler([1*math.pi,0,0]))

jointIds = [0,1,2,3]
initial_positions = [0, 1, 0, 1] # Adjust with your desired initial positions
# stiffness = [1e1,] * len(jointIds)  # P gain for each joint
# damping = [1e9,] * len(jointIds)  # D gain for each joint
for i in range(len(jointIds)):
    p.resetJointState(gripperId, i, initial_positions[i], 1)

for i in range(80000):
    # print("i: ", i)

    lim = 0.005
    jointPosition = p.getJointState(gripperId, 1)[0]
    p.resetJointState(gripperId, 1, jointPosition+random.uniform(-0, lim))
    jointPosition = p.getJointState(gripperId, 3)[0]
    p.resetJointState(gripperId, 3, jointPosition+random.uniform(-0, lim))
    # p.resetJointState(gripperId, 0, initial_positions[0]+0.001*i)
    # p.resetJointState(gripperId, 2, initial_positions[2]+0.001*i)
    # p.resetBasePositionAndOrientation(gripperId, [0,0,2-i/1000], p.getQuaternionFromEuler([1*math.pi,0,0]))

    # apply upward force on the gripper
    pos_gripper, _ = p.getBasePositionAndOrientation(gripperId)

    # Step simulation
    p.stepSimulation()
    time.sleep(5./240.)

# Disconnect from PyBullet
p.disconnect()