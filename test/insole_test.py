import pybullet as p
import pybullet_data
import time
import math

# Physics simulation setup
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)  # Enable Finite Element Method (FEM)
p.setGravity(0, 0, -10)
planeUid = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])

# Load scoop as robot end-effector
scoop = p.loadURDF(
    "asset/scoop.urdf", 
    basePosition=[0, 0, 1.0],  # Position above the ground
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    globalScaling=1
)

# Load the insole soft body
tex = p.loadTexture("uvmap.png")
insole = p.loadSoftBody(
    "/home/yif/Documents/KTH/git/MTGripperDesign/asset/insole.vtk",
    basePosition=[0, 0, 2.2],  # Position on top of the scoop
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    scale=5,
    mass=0.005,
    useNeoHookean=1,
    useSelfCollision=1,
    frictionCoeff=0.5,
    collisionMargin=0.001
)
p.changeVisualShape(insole, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=tex, flags=0)

# # Load a box
# box = p.loadURDF(
#     "cube_small.urdf",
#     basePosition=[0, 0, 1],
#     baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
#     globalScaling=15
# )

# Set physics engine parameters
p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
p.setRealTimeSimulation(0)

# Run simulation
while True:
    p.stepSimulation()
    time.sleep(1. / 240.)
