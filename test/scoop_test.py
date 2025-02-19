import pybullet as p
import pybullet_data
import time
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Physics simulation setup
g = -9.81
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)  # Enable Finite Element Method (FEM)
p.setGravity(0, 0, g)
planeUid = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])

# p.vhacd("asset/bowl/bowl.obj", "asset/bowl/bowl-vhacd.obj", "log.txt")

# Define rectangle dimensions (width, height, depth)
width = 1.0
height = 0.5
depth = 0.02

# Create collision and visual shapes
half_extents = [width / 2, height / 2, depth / 2]
collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extents)
visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0, 1, 0, 1])

# Create the rigid body
mass = 1.0
mass_scoop = 1
base_position = [1, 2.5, depth/2]  # Initial position (x, y, z)
base_orientation = [0, 0, 0, 1]  # Quaternion (x, y, z, w)
scoop = p.createMultiBody(baseMass=mass,
                                 baseCollisionShapeIndex=collision_shape_id,
                                 baseVisualShapeIndex=visual_shape_id,
                                 basePosition=base_position,
                                 baseOrientation=base_orientation)
bowl_id = p.createCollisionShape(p.GEOM_MESH, fileName="asset/bowl/bowl-vhacd.obj", meshScale=[0.045,]*3)
p.createMultiBody(baseMass=1, baseCollisionShapeIndex=bowl_id, baseVisualShapeIndex=bowl_id, basePosition=[2.2, 2.5, 0.1])

# Load the soft body
object_type = "insole" # bread, pillow, insole
tex = p.loadTexture("uvmap.png")
mass_pillow = 0.005
if object_type == "bread":
    object_id = p.loadSoftBody(
        "asset/bread.vtk", # pillow, bread
        basePosition=[2.5,2.5,3],  # Position on top of the scoop
        baseOrientation=p.getQuaternionFromEuler([math.pi/2, 0, math.pi/2]),
        scale=.5,
        mass=0.005,
        useNeoHookean=1,
        NeoHookeanMu=5,
        NeoHookeanLambda=5,
        NeoHookeanDamping=0.1,
        useSelfCollision=1,
        frictionCoeff=0.5,
        collisionMargin=0.001
    )
    p.changeVisualShape(object_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=tex, flags=0)
elif object_type == "pillow":
    # Load the pillow soft body
    object_id = p.loadSoftBody(
        "asset/pillow.vtk", # pillow, bread
        basePosition=[2.5, 2.5, 1],  # Position on top of the scoop
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        scale=2.5,
        mass=mass_pillow,
        useNeoHookean=1,
        NeoHookeanMu=5,
        NeoHookeanLambda=5,
        NeoHookeanDamping=.1,
        useSelfCollision=1,
        frictionCoeff=0.5,
        collisionMargin=0.001
    )
    p.changeVisualShape(object_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=tex, flags=0)
elif object_type == "insole":
    # Load the pillow soft body
    object_id = p.loadSoftBody(
        "asset/insole.vtk", # pillow, bread
        basePosition=[2.5, 2.5, 1],  # Position on top of the scoop
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        scale=5,
        mass=mass_pillow,
        useNeoHookean=1,
        NeoHookeanMu=5,
        NeoHookeanLambda=5,
        NeoHookeanDamping=.1,
        useSelfCollision=1,
        frictionCoeff=0.5,
        collisionMargin=0.001
    )
    p.changeVisualShape(object_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=tex, flags=0)


# Create a heavy box to fix the bread at the other end
box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.8, 1.1])
box_body = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=box,
    basePosition=[5, 2.5, 0.05],
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
dt = 1. / 240.
while True:
    print(i) if i % 10 == 0 else None
    # Get current position and orientation of the scoop
    scoop_position, scoop_orientation = p.getBasePositionAndOrientation(scoop)
    object_position, object_orientation = p.getBasePositionAndOrientation(object_id)
    # Apply force on scoop
    if i > 100 and i < 800:
        p.applyExternalForce(scoop, -1, [7, 0, 0], scoop_position, p.WORLD_FRAME)
    
    # Lift the scoop
    if i > 800 and i < 1100:
        p.applyExternalForce(scoop, -1, [0, 0, 11], scoop_position, p.WORLD_FRAME)

    # Fix the scoop in the air
    if i == 1100:
    #     p.applyExternalForce(scoop, -1, [0, 0, -g*(mass_pillow+mass_scoop)], scoop_position, p.WORLD_FRAME)
        p.changeDynamics(scoop, -1, mass=0)
        p.resetBaseVelocity(scoop, [0, 0, 0], [0, 0, 0])

    # if i > 1100:
    #     # # FORCE on object
    #     # p.applyExternalForce(object_id, -1, [random.uniform(-10000,10000) for _ in range(3)], object_position, p.WORLD_FRAME)

    #     # VELOCITY
    #     limit = 2000
    #     rand_acc = [random.uniform(-limit, limit) for _ in range(3)]
    #     total_acc = rand_acc
    #     total_acc[2] = g
    #     delta_vel = [a * dt for a in total_acc]
    #     curr_vel = p.getBaseVelocity(object_id)[0]
    #     new_vel = [v + dv for v, dv in zip(curr_vel, delta_vel)]
    #     p.resetBaseVelocity(object_id, new_vel, [0, 0, 0])
        
        # # Random FORCE on robot
        # p.applyExternalForce(scoop, -1, [random.uniform(-10000,10000) for _ in range(3)], scoop_position, p.WORLD_FRAME)

    p.stepSimulation()
    time.sleep(4. / 240.)
    i += 1
