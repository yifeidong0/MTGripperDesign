import pybullet as p
import os
import sys
import concurrent.futures
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TODO: timeout does not work properly
def vhacd_with_timeout(input_file, output_file, name_log, resolution, timeout=5):
    def vhacd_function(input_file, output_file, name_log, resolution):
        p.vhacd(input_file, output_file, name_log, resolution=resolution)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(vhacd_function, input_file, output_file, name_log, resolution)
        try:
            # Wait for the function to complete with a timeout
            future.result(timeout=timeout)
            return True
        except concurrent.futures.TimeoutError:
            print(f"VHACD decomposition timed out after {timeout} seconds")
            return False

def decompose_mesh(
        pb_connected=False,
        input_file="asset/vpusher/v_pusher.obj",  # Replace with your concave mesh file path
        name_log="log.txt",
        resolution=int(1e5),
        timeout=5,
):
    output_file = input_file

    # Connect to PyBullet and load the plane
    if not pb_connected:
        p.connect(p.DIRECT)

    # Redirect stdout and stderr to null to suppress verbose output
    print(f"Decomposing {input_file}...")
    devnull = open(os.devnull, 'w')
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)

    # Add time out: 15 sec for the execution of the VHACD
    try:
        success = vhacd_with_timeout(input_file, output_file, name_log, resolution, timeout=timeout)
    finally:
        # Restore stdout and stderr
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        devnull.close()

    # Disconnect from PyBullet
    if not pb_connected:
        p.disconnect()

    # Remove log file
    if os.path.exists(name_log):
        os.remove(name_log)

    if success:
        print(f"Decomposed mesh saved to {output_file}")
    else:
        print("Failed to decompose mesh within the time limit.")
    return success

# Example usage
if __name__ == "__main__":
    decompose_mesh()
