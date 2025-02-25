# Intrinsic Matrix (K):
# [614.9035034179688, 0, 323.47271728515625]
# [0, 614.9575805664062, 237.75799560546875]
# [0, 0, 1]

import pyrealsense2 as rs

# Create pipeline and start the camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Adjust resolution as needed

# Start streaming
profile = pipeline.start(config)

# Get the intrinsics
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# Print the intrinsic matrix
K = [[intrinsics.fx, 0, intrinsics.ppx],
     [0, intrinsics.fy, intrinsics.ppy],
     [0, 0, 1]]

print("Intrinsic Matrix (K):")
for row in K:
    print(row)

# Stop the pipeline
pipeline.stop()
