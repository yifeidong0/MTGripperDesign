import pybullet as p
import pybullet_data
import os

# Connect to PyBullet in GUI mode
p.connect(p.GUI)

# Set the search path for PyBullet's data
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Setup camera
p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=90, cameraPitch=-89.99, cameraTargetPosition=[2.5, 2.5, 0])
p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[2.5, 2.5, 0], 
                                                        distance=2, yaw=90, pitch=-89.99, roll=0, upAxisIndex=2)
p.computeProjectionMatrixFOV(fov=103, aspect=1, nearVal=0.1, farVal=100)

# Define the table size (wide rectangle) and mass
table_size = [15, 30, 0.1]  # Width, length, height (Z)
table_position = [0, 0, 0.01-table_size[2] / 2]  # Set the top of the table at z=0

# Create the table using a box with mass=0
table_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[table_size[0]/2, table_size[1]/2, table_size[2]/2])
table_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[table_size[0]/2, table_size[1]/2, table_size[2]/2], rgbaColor=[0.8, 0.8, 0.8, 1])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_collision_shape, baseVisualShapeIndex=table_visual_shape, basePosition=table_position)

# Load the tool URDF from your asset folder
tool_urdf_path = "asset/2ie6iynn/vpusher_modified_2.210.urdf"
tool_start_position = [2, 2.5, 0.1]  # Slightly above the table
tool_start_orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation

p.loadURDF(tool_urdf_path, tool_start_position, tool_start_orientation, globalScaling=10)

# Load an OBJ file (e.g., an object on the table)
obj_file_path = "asset/polygons/oval.obj"
scale = [6,]*3  # Scale factor (increase in size)
obj_collision_shape = p.createCollisionShape(p.GEOM_MESH, fileName=obj_file_path, meshScale=scale)
scale = [6,6,3]  # Scale factor (increase in size)
obj_visual_shape = p.createVisualShape(p.GEOM_MESH, fileName=obj_file_path, meshScale=scale, rgbaColor=[0.8, 0.2, 0.2, 1])
p.createMultiBody(baseMass=1, baseCollisionShapeIndex=obj_collision_shape, baseVisualShapeIndex=obj_visual_shape, basePosition=[2.5, 2.5, 0.2],baseOrientation=p.getQuaternionFromEuler([0, 0, 1]))

# Set gravity
p.setGravity(0, 0, -9.8)

# Run the simulation
import time
while True:
    p.stepSimulation()
    time.sleep(1./240.)
