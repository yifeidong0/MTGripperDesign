import numpy as np
from stl import mesh
import trimesh

def generate_scoop_stl_obj(coefficients, thickness, filename='scoop.stl', obj_filename='scoop.obj'):
    x = np.linspace(0, 1, 200)
    y = sum(c * x**i for i, c in enumerate(coefficients))
    y = y - y.min()
    y = y / y.max()  # Normalize to range [0, 1]

    depth = 1

    # Generate the surface by extruding the curve along the depth axis
    z = np.linspace(0, depth, 200)
    x_grid, z_grid = np.meshgrid(x, z)
    y_grid = np.meshgrid(y, z)[0]

    vertices = np.zeros((len(x_grid.ravel()), 3))
    vertices[:, 0] = x_grid.ravel()
    vertices[:, 1] = y_grid.ravel()
    vertices[:, 2] = z_grid.ravel()

    # Centering the vertices
    center = np.mean(vertices, axis=0)
    vertices -= center

    # Create faces for a single-layer thin scoop
    faces = []
    for i in range(len(z) - 1):
        for j in range(len(x) - 1):
            faces.append([i * len(x) + j, i * len(x) + j + 1, (i + 1) * len(x) + j])
            faces.append([i * len(x) + j + 1, (i + 1) * len(x) + j + 1, (i + 1) * len(x) + j])

    # Create mesh
    faces = np.array(faces)
    scoop_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            scoop_mesh.vectors[i][j] = vertices[f[j], :]

    # Save to STL
    scoop_mesh.save(filename)
    print(f'Scoop STL saved to {filename}')

    # Save to OBJ
    with open(obj_filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    print(f'Scoop OBJ saved to {obj_filename}')

# Parameters for the scoop
coefficients = [0, 0.5, 1, -1]  # Example coefficients for a cubic polynomial
thickness = 0.1

generate_scoop_stl_obj(coefficients, thickness)


########## Convex decomposition of the scoop ##########
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


########## Voxelize the scoop (making it jagged) ##########
# import numpy as np
# from stl import mesh
# import trimesh

# def load_stl(file_path):
#     return trimesh.load_mesh(file_path)

# def fill_mesh(mesh):
#     # Ensure the mesh is watertight
#     if not mesh.is_watertight:
#         mesh.fill_holes()
#     # Create a voxelized version of the mesh
#     voxelized = mesh.voxelized(pitch=0.005)
#     # Convert voxel grid back to a mesh
#     filled_mesh = voxelized.as_boxes()
#     return filled_mesh

# def save_stl(mesh, file_path):
#     mesh.export(file_path, file_type='stl')

# def repair_stl(input_file, output_file):
#     mesh = load_stl(input_file)
#     filled_mesh = fill_mesh(mesh)
#     save_stl(filled_mesh, output_file)
#     print(f"Repaired STL saved to {output_file}")

# # Use the function to repair an STL file
# input_stl = 'scoop.stl'
# output_stl = 'scoop.stl'
# repair_stl(input_stl, output_stl)
