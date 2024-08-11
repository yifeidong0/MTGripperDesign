import numpy as np
from stl import mesh
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vhacd import decompose_mesh

def generate_v_shape_pusher(finger_length, base_angle, thickness, height, 
                            obj_filename='asset/vpusher/v_pusher.obj',
                            finger_angle=None,
                            distal_phalanx_length=None,
                            ):
    half_angle = base_angle / 2.0
    extended_fingers = finger_angle is not None and distal_phalanx_length is not None
    
    # Calculate the vertices for the first rectangular cuboid
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

    # Add distal phalanx
    if extended_fingers:
        vertices11 = [
            (0, 0, 0),
            (distal_phalanx_length * math.cos(finger_angle), distal_phalanx_length * math.sin(finger_angle), 0),
            (distal_phalanx_length * math.cos(finger_angle) - thickness * math.sin(finger_angle), 
             distal_phalanx_length * math.sin(finger_angle) + thickness * math.cos(finger_angle), 0),
            (-thickness * math.sin(finger_angle), thickness * math.cos(finger_angle), 0),
            (0, 0, height),
            (distal_phalanx_length * math.cos(finger_angle), distal_phalanx_length * math.sin(finger_angle), height),
            (distal_phalanx_length * math.cos(finger_angle) - thickness * math.sin(finger_angle), 
             distal_phalanx_length * math.sin(finger_angle) + thickness * math.cos(finger_angle), height),
            (-thickness * math.sin(finger_angle), thickness * math.cos(finger_angle), height)
        ]
        vertices11 = [(v[0] + finger_length * math.cos(half_angle), 
                       v[1] + finger_length * math.sin(half_angle),
                       v[2]) for v in vertices11]

    # Calculate the vertices for the second rectangular cuboid, mirrored around the x-axis
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

    # Add distal phalanx
    if extended_fingers:
        vertices21 = [
            (0, 0, 0),
            (distal_phalanx_length * math.cos(finger_angle), -distal_phalanx_length * math.sin(finger_angle), 0),
            (distal_phalanx_length * math.cos(finger_angle) - thickness * math.sin(finger_angle), 
             -distal_phalanx_length * math.sin(finger_angle) - thickness * math.cos(finger_angle), 0),
            (-thickness * math.sin(finger_angle), -thickness * math.cos(finger_angle), 0),
            (0, 0, height),
            (distal_phalanx_length * math.cos(finger_angle), -distal_phalanx_length * math.sin(finger_angle), height),
            (distal_phalanx_length * math.cos(finger_angle) - thickness * math.sin(finger_angle), 
             -distal_phalanx_length * math.sin(finger_angle) - thickness * math.cos(finger_angle), height),
            (-thickness * math.sin(finger_angle), -thickness * math.cos(finger_angle), height)
        ]
        vertices21 = [(v[0] + finger_length * math.cos(half_angle), 
                       v[1] - finger_length * math.sin(half_angle),
                       v[2]) for v in vertices21]
        
    # Combine all vertices
    if extended_fingers:
        vertices = np.array(vertices1 + vertices2 + vertices11 + vertices21)
    else:
        vertices = np.array(vertices1 + vertices2)

    # Define the faces for the first rectangular cuboid
    faces1 = [
        [0, 1, 2], [0, 2, 3],  # Bottom faces
        [4, 5, 6], [4, 6, 7],  # Top faces
        [0, 1, 5], [0, 5, 4],  # Side faces
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7]
    ]

    # Define the faces for the second rectangular cuboid
    faces2 = [
        [8, 9, 10], [8, 10, 11],  # Bottom faces
        [12, 13, 14], [12, 14, 15],  # Top faces
        [8, 9, 13], [8, 13, 12],  # Side faces
        [9, 10, 14], [9, 14, 13],
        [10, 11, 15], [10, 15, 14],
        [11, 8, 12], [11, 12, 15]
    ]

    if extended_fingers:
        faces11 = [
            [16, 17, 18], [16, 18, 19],  # Bottom faces
            [20, 21, 22], [20, 22, 23],  # Top faces
            [16, 17, 21], [16, 21, 20],  # Side faces
            [17, 18, 22], [17, 22, 21],
            [18, 19, 23], [18, 23, 22],
            [19, 16, 20], [19, 20, 23]
        ]
        faces21 = [
            [24, 25, 26], [24, 26, 27],  # Bottom faces
            [28, 29, 30], [28, 30, 31],  # Top faces
            [24, 25, 29], [24, 29, 28],  # Side faces
            [25, 26, 30], [25, 30, 29],
            [26, 27, 31], [26, 31, 30],
            [27, 24, 28], [27, 28, 31]
        ]

    # Combine the faces
    if extended_fingers:
        faces = np.array(faces1 + faces2 + faces11 + faces21)
    else:
        faces = np.array(faces1 + faces2)

    # # Center the vertices
    # center = np.mean(vertices, axis=0)
    # vertices -= center

    # Create mesh
    v_shape_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            v_shape_mesh.vectors[i][j] = vertices[face[j], :]

    # Save to OBJ
    with open(obj_filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


# TODO: produce v pushers in advance before runtime
if __name__ == '__main__':
    finger_length = .5
    base_angle = math.radians(40)  # 90 degrees opening base_angle
    thickness = 0.1
    height = 0.1

    generate_v_shape_pusher(finger_length, base_angle, thickness, height)

    decompose_mesh()