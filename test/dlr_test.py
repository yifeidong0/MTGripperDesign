import pybullet as p
import pybullet_data
import time
import math
import numpy as np

# Physics simulation setup
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeUid = p.loadURDF("plane.urdf", basePosition=[0,0,0])

g = 9.81
m_box = .1

for i in range(1):
    boxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.1,.1,.1,])
    boxId = p.createMultiBody(m_box, boxId, -1, np.array([0, 0, 0.6]))    

gripperId = p.loadURDF(fileName='asset/lc_soft_enable_wide_grip/lc_soft_enable_wide_grip.urdf', 
                        basePosition=[0,0,2.6], 
                        baseOrientation=p.getQuaternionFromEuler([1*math.pi,0,0]),
                        globalScaling=10,
                        )

pivotInBox = [0, 0, 0] 
pivotInWorld = [0, 0, 0]
constraint1 = p.createConstraint(gripperId, -1, -1, -1, p.JOINT_PRISMATIC, [0, 0, 1], pivotInBox, pivotInWorld, p.getQuaternionFromEuler([1*math.pi,0,0]))

jointIds = [0,1,2,3]
initial_positions = [0, 1, 0, 1] # Adjust with your desired initial positions
stiffness = [1e1,] * len(jointIds)  # P gain for each joint
damping = [1e9,] * len(jointIds)  # D gain for each joint

for i in range(80000):
    # print("i: ", i)

    # Calculate required torques (e.g., based on current positions and desired positions)
    joint_states = p.getJointStates(gripperId, jointIds)
    current_positions = [state[0] for state in joint_states]
    position_errors = [initial_positions[j] - current_positions[j] for j in range(len(jointIds))]
    torques = [stiffness[j] * position_errors[j] - damping[j] * joint_states[j][1] for j in range(len(jointIds))]

    p.resetJointState(gripperId, 1, initial_positions[1]-0.001*i)
    p.resetJointState(gripperId, 3, initial_positions[3]-0.001*i)
    p.resetJointState(gripperId, 0, initial_positions[0]-0.001*i)
    p.resetJointState(gripperId, 2, initial_positions[2]-0.001*i)
    p.resetBasePositionAndOrientation(gripperId, [0,0,2+i/1000], p.getQuaternionFromEuler([1*math.pi,0,0]))

    res = p.getClosestPoints(gripperId, boxId, 100, 1, -1) # 1,3,
    dist = res[0][8] if (len(res)>0) else 0 # and res[0][8]>=0
    joint_states = p.getJointStates(gripperId, jointIds)
    joint_positions = [state[0] for state in joint_states]
    # print('joint_positions: ', joint_positions)

    # apply upward force on the gripper
    pos_gripper, _ = p.getBasePositionAndOrientation(gripperId)

    # Step simulation
    p.stepSimulation()
    time.sleep(5./240.)

# Disconnect from PyBullet
p.disconnect()