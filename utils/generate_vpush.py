import numpy as np
from stl import mesh
import math

def generate_v_shape_pusher_stl_obj(finger_length, angle, thickness, height, filename='v_pusher.stl', obj_filename='v_pusher.obj'):
    half_angle = angle / 2.0
    
    # Calculate the vertices for the first triangular prism
    vertices1 = [
        (0, 0, 0),
        (finger_length * math.cos(half_angle), finger_length * math.sin(half_angle), 0),
        (finger_length * math.cos(half_angle) - thickness * math.sin(half_angle), 
         finger_length * math.sin(half_angle) + thickness * math.cos(half_angle), 0),
        (-thickness * math.sin(half_angle), thickness * math.cos(half_angle), 0),
        (0, 0, height),
        (finger_length * math.cos(half_angle), finger_length * math.sin(half_angle), height),
        (finger_length * math.cos(half_angle) - thickness * math.sin(half_angle), 
         finger_length * math.sin(half_angle) + thickness * math.cos(half_angle), height),
        (-thickness * math.sin(half_angle), thickness * math.cos(half_angle), height)
    ]

    # Calculate the vertices for the second triangular prism, mirrored around the x-axis
    vertices2 = [
        (0, 0, 0),
        (finger_length * math.cos(half_angle), -finger_length * math.sin(half_angle), 0),
        (finger_length * math.cos(half_angle) - thickness * math.sin(half_angle), 
         -finger_length * math.sin(half_angle) - thickness * math.cos(half_angle), 0),
        (-thickness * math.sin(half_angle), -thickness * math.cos(half_angle), 0),
        (0, 0, height),
        (finger_length * math.cos(half_angle), -finger_length * math.sin(half_angle), height),
        (finger_length * math.cos(half_angle) - thickness * math.sin(half_angle), 
         -finger_length * math.sin(half_angle) - thickness * math.cos(half_angle), height),
        (-thickness * math.sin(half_angle), -thickness * math.cos(half_angle), height)
    ]

    # Combine all vertices
    vertices = np.array(vertices1 + vertices2)

    # Define the faces for the first triangular prism
    faces1 = [
        [0, 1, 2], [0, 2, 3],  # Bottom faces
        [4, 5, 6], [4, 6, 7],  # Top faces
        [0, 1, 5], [0, 5, 4],  # Side faces
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7]
    ]

    # Define the faces for the second triangular prism
    faces2 = [
        [8, 9, 10], [8, 10, 11],  # Bottom faces
        [12, 13, 14], [12, 14, 15],  # Top faces
        [8, 9, 13], [8, 13, 12],  # Side faces
        [9, 10, 14], [9, 14, 13],
        [10, 11, 15], [10, 15, 14],
        [11, 8, 12], [11, 12, 15]
    ]

    # Combine the faces
    faces = np.array(faces1 + faces2)

    # Center the vertices
    center = np.mean(vertices, axis=0)
    vertices -= center

    # Create mesh
    v_shape_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            v_shape_mesh.vectors[i][j] = vertices[face[j], :]

    # Save to STL
    v_shape_mesh.save(filename)
    print(f'V-shape pusher STL saved to {filename}')

    # Save to OBJ
    with open(obj_filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    print(f'V-shape pusher OBJ saved to {obj_filename}')

# Example usage
finger_length = .5
angle = math.radians(60)  # 90 degrees opening angle
thickness = 0.1
height = 0.1

generate_v_shape_pusher_stl_obj(finger_length, angle, thickness, height)


########## Convex decomposition of the scoop ##########
import pybullet as p
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Connect to PyBullet and load the plane
p.connect(p.DIRECT)

# Use VHACD to decompose a concave object into convex parts
input_file = "v_pusher.obj"  # Replace with your concave mesh file path
output_prefix = "v_pusher_vhacd.obj"
name_log = "log.txt"

# Decompose the mesh
p.vhacd(input_file, output_prefix, name_log)