import pybullet as p
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Connect to PyBullet and load the plane
p.connect(p.DIRECT)

# Use VHACD to decompose a concave object into convex parts
input_file = "scoop.obj"  # Replace with your concave mesh file path
output_prefix = "scoop_decomposed_convex.obj"
name_log = "log.txt"

# Decompose the mesh
p.vhacd(input_file, output_prefix, name_log)
