# Load a urdf in Pybullet
import pybullet as p
import pybullet_data
import os
import numpy as np
import time

# Load the URDF file
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")
kuka = p.loadURDF("asset/technion/robotiq_arg85_technion_description.urdf", [0,0,1])

while 1:
    p.stepSimulation()
    time.sleep(1./240.)