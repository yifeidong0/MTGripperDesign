import pybullet as p
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def decompose_mesh(
        pb_connected = False,
        input_file = "asset/vpusher/v_pusher.obj",  # Replace with your concave mesh file path
        name_log = "log.txt",
        resolution = int(1e5),
    ):
    output_file = input_file
    
    # Connect to PyBullet and load the plane
    if not pb_connected:
        p.connect(p.DIRECT)

    # Redirect stdout and stderr to null to suppress verbose output
    print(f"Decomposing {input_file}...")
    # devnull = open(os.devnull, 'w')
    # old_stdout = os.dup(1)
    # old_stderr = os.dup(2)
    # os.dup2(devnull.fileno(), 1)
    # os.dup2(devnull.fileno(), 2)

    # try:
    #     # Use VHACD to decompose a concave object into convex parts
    #     # Decompose the mesh
    #     p.vhacd(input_file, output_file, name_log, resolution=resolution)
    # finally:
    #     # Restore stdout and stderr
    #     os.dup2(old_stdout, 1)
    #     os.dup2(old_stderr, 2)
    #     devnull.close()
    p.vhacd(input_file, output_file, name_log, resolution=resolution)

    # Disconnect from PyBullet
    if not pb_connected:
        p.disconnect()

    # Remove log file
    if os.path.exists(name_log):
        os.remove(name_log)
    print(f"Decomposed mesh saved to {output_file}")