import pybullet as p
import pybullet_data
import time
import math
import numpy as np

# Physics simulation setup
g = -9.81
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)  # Enable Finite Element Method (FEM)
# p.setGravity(0, 0, g)
# planeUid = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])

# Load scoop as robot end-effector
mass_scoop = 1
scoop = p.loadURDF(
    "asset/vpusher/v_pusher.urdf", 
    basePosition=[0, 0, 0.0],  # Position above the ground
    baseOrientation=p.getQuaternionFromEuler([0,0, 0]),
    globalScaling=1
)
p.changeDynamics(scoop, -1, lateralFriction=0.6, spinningFriction=0.4, rollingFriction=0.4)

# Set physics engine parameters
p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)

# Run simulation
i = 0
while True:
    # Get current position and orientation of the scoop
    scoop_position, scoop_orientation = p.getBasePositionAndOrientation(scoop)
    # scoop_euler = p.getEulerFromQuaternion(scoop_orientation)
    # print(f"Step {i}: Orientation: {scoop_euler}")

    # # apply z-axis torque on the scoop
    # p.applyExternalTorque(scoop, -1, torqueObj=[0, 0, 10], flags=p.WORLD_FRAME)

    p.stepSimulation()
    time.sleep(10. / 240.)
    i += 1
