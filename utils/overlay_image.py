import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

# Load the video file
video_path = '/home/yif/Videos/vpush.mp4'
cap = cv2.VideoCapture(video_path)

# Get the total number of frames and the video FPS
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Generate non-linear frame indices (quadratic distribution for more frames at the end)
frame_indices = [int((i / 7) ** 0.9 * total_frames) for i in range(1, 8)]

# List to store the extracted frames
frames = []

# Extract the frames from the video based on the generated indices
for idx, frame_num in enumerate(frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

# Resize frames to the same size for overlaying
height, width, _ = frames[0].shape

# Initialize a transparent canvas to overlay images
overlay_image = np.zeros((height, width, 3), dtype=np.uint8)

# Current time for unique timestamp in file names
current_time = time.strftime("%Y%m%d_%H%M%S")

# Overlay frames with varying transparency and save intermediate images with timestamps
total_alpha = 0
for i in range(6):  # 6 intermediate frames
    alpha = (i + 1) / 10  # Use smaller alpha values to avoid overexposure
    total_alpha += alpha  # Keep track of the sum of alphas
    overlay_image = cv2.addWeighted(overlay_image, 1, frames[i], alpha, 0)

    # Calculate the time of the frame in seconds (using fps)
    timestamp = frame_indices[i] / fps

    # Save intermediate images with timestamps in filenames
    intermediate_output_path = f'/home/yif/Videos/overlay_step_{i + 1}_{current_time}_t{timestamp:.2f}s.png'
    cv2.imwrite(intermediate_output_path, overlay_image)
    print(f'Saved intermediate image: {intermediate_output_path}')

# Normalize the total alpha to avoid overexposure and add the final frame with full clarity
final_alpha = 1 - total_alpha / 6  # Ensure the final frame contributes without overexposure
final_alpha = max(0, min(1, final_alpha))  # Clamp alpha to [0, 1]

overlay_image = cv2.addWeighted(overlay_image, 1, frames[5], final_alpha, 0)

# Save the final overlaid image with timestamp
timestamp_final = frame_indices[6] / fps
final_output_path = f'/home/yif/Videos/catch1_final_overlay_{current_time}_t{timestamp_final:.2f}s.png'
cv2.imwrite(final_output_path, overlay_image)
print(f'Saved final overlaid image: {final_output_path}')

cap.release()

# Display the final image
overlay_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
plt.imshow(overlay_image_rgb)
plt.axis('off')  # Hide axes
plt.show()
