import numpy as np
from stl import mesh

def generate_scoop(coefficients, filename='scoop.obj'):
    x = np.linspace(0, 1.5, 30)
    # y = sum(c * x**i for i, c in enumerate(coefficients))
    # y = y - y.min()
    # y = y / y.max()  # Normalize to range [0, 1]
    
    # the curve passes through the origin and (1,0)
    c = coefficients
    # z = c[0] * x * (1.5-x) * (c[1]-x)**2 * (c[2]-x)**2
    z = c[0] * x * (1.5-x) * (c[1]-x)**2

    depth = 1.5

    # Generate the surface by extruding the curve along the depth axis
    y = np.linspace(0, depth, 30)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.meshgrid(z, y)[0]

    vertices = np.zeros((len(x_grid.ravel()), 3))
    vertices[:, 0] = x_grid.ravel()
    vertices[:, 1] = y_grid.ravel()
    vertices[:, 2] = z_grid.ravel()

    # Centering the vertices (center x, y but not z)
    # center = np.mean(vertices, axis=0)
    # vertices -= center
    center = np.mean(vertices, axis=0)
    vertices[:, 0] -= center[0]
    vertices[:, 1] -= center[1]

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

    # Save to OBJ
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    print(f'Scoop OBJ saved to {filename}')

# # Parameters for the scoop
# coefficients = [0, 0.5, 1, -1]  # Example coefficients for a cubic polynomial

# generate_scoop(coefficients)