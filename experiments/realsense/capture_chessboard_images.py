import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Chessboard settings
CHESSBOARD_SIZE = (6, 4)  # (number of INNNER corners per row, per column)
SAVE_PATH = "captured_chessboard_images"  # Folder to save images
os.makedirs(SAVE_PATH, exist_ok=True)

# Configure the pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

image_count = 6

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Duplicate the color image
        color_image_copy = color_image.copy()

        # Convert to grayscale for chessboard detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        print('ret:',ret, 'corners:',corners)

        if ret:
            # Draw the chessboard corners
            cv2.drawChessboardCorners(color_image, CHESSBOARD_SIZE, corners, ret)

            # Display the image
            cv2.imshow("Chessboard Detection", color_image)

            # Save image when 's' is pressed (press within the opencv window not the terminal)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                filename = os.path.join(SAVE_PATH, f"chessboard_{image_count}.bmp")
                cv2.imwrite(filename, color_image_copy)
                print(f"Saved: {filename}")
                image_count += 1

        else:
            # Show normal RGB image if no chessboard detected
            cv2.imshow("RGB Stream", color_image)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
