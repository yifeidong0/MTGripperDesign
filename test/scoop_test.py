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
p.setGravity(0, 0, g)
planeUid = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])

# Load scoop as robot end-effector
mass_scoop = 1
robot_orientation = p.getQuaternionFromEuler([math.pi/2, -.6-math.pi/2, 0*math.pi/2])
scoop = p.loadURDF(
    "asset/scoop.urdf", 
    basePosition=[-2, 0, 1.0],  # Position above the ground
    baseOrientation=robot_orientation,
    globalScaling=1
)
p.changeDynamics(scoop, -1, lateralFriction=0.6, spinningFriction=0.4, rollingFriction=0.4)

# Change mass
p.changeDynamics(scoop, -1, mass=mass_scoop)

# Load the insole soft body
tex = p.loadTexture("uvmap.png")
mass_pillow = 0.005
insole = p.loadSoftBody(
    "asset/insole.vtk", # pillow, insole
    basePosition=[0, -0.5, 1.2],  # Position on top of the scoop
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    scale=5,
    mass=0.005,
    useNeoHookean=1,
    NeoHookeanMu=1.5,
    NeoHookeanLambda=1,
    NeoHookeanDamping=0.1,
    useSelfCollision=1,
    frictionCoeff=0.5,
    collisionMargin=0.001
)
p.changeVisualShape(insole, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=tex, flags=0)

# Load the pillow soft body
# pillow = p.loadSoftBody(
#     "asset/pillow.vtk", # pillow, insole
#     basePosition=[0, 0, 1],  # Position on top of the scoop
#     baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
#     scale=2.5,
#     mass=mass_pillow,
#     useNeoHookean=1,
#     NeoHookeanMu=.5,
#     # NeoHookeanLambda=1,
#     NeoHookeanDamping=.01,
#     useSelfCollision=1,
#     frictionCoeff=0.5,
#     collisionMargin=0.001
# )
# p.changeVisualShape(pillow, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=tex, flags=0)

# Create a heavy box to fix the insole at the other end
box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.8, 0.1])
box_body = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=box,
    basePosition=[3, 0, 0.1],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
)

# Set physics engine parameters
p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
p.setRealTimeSimulation(0)

# Control variables
scoop_velocity = 0.01  # Set the velocity for scooping
scoop_target_position = [1, 0, 1.2]  # Target position for the scoop to move along the x-axis

# Run simulation
i = 0
while True:
    # Get current position and orientation of the scoop
    scoop_position, scoop_orientation = p.getBasePositionAndOrientation(scoop)

    # # Move the scoop towards the target position using velocity control
    # new_position = [
    #     scoop_position[0] + scoop_velocity if scoop_position[0] < scoop_target_position[0] else scoop_target_position[0],
    #     scoop_position[1],
    #     scoop_position[2]
    # ]

    # p.resetBasePositionAndOrientation(scoop, new_position, scoop_orientation)

    # Apply force on scoop
    if i > 100 and i < 800:
        p.applyExternalForce(scoop, -1, [7, 0, 0], scoop_position, p.WORLD_FRAME)
    
    # Lift the scoop
    if i > 800 and i < 1000:
        p.applyExternalForce(scoop, -1, [0, 0, 11], scoop_position, p.WORLD_FRAME)

    # Fix the scoop in the air
    if i > 1000:
    #     p.applyExternalForce(scoop, -1, [0, 0, -g*(mass_pillow+mass_scoop)], scoop_position, p.WORLD_FRAME)
        p.changeDynamics(scoop, -1, mass=0)
        p.resetBaseVelocity(scoop, [0, 0, 0], [0, 0, 0])

    p.stepSimulation()
    time.sleep(1. / 240.)
    i += 1
