import pyrealsense2 as rs
import numpy as np
import cv2
import time

# 1. Configure the pipeline to ensure 30 FPS
pipeline = rs.pipeline()
config = rs.config()

# Explicitly request 640x480 at 30Hz for both streams
# You can change resolution to 848x480 or 1280x720 depending on your device capabilities
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Optional: Print actual FPS to console to verify
prev_time = 0
fps_counter = 0

try:
    print("Streaming started. Press 'q' or 'ESC' to exit...")
    while True:
        # 2. Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # 3. Convert images to numpy arrays
        # Depth is 16-bit integer (millimeters), Color is 8-bit RGB
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 4. Colorize depth image to make it visible
        # We apply a colormap (Jet) to the depth data. 
        # alpha=0.03 scales the depth so close objects are visible (approx 0-3 meters range)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 5. Stack both images horizontally
        # Ensure dimensions match before stacking
        images = np.hstack((color_image, depth_colormap))

        # 6. Calculate FPS for verification
        curr_time = time.time()
        fps_counter = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Overlay FPS on the image
        cv2.putText(images, f"FPS: {int(fps_counter)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 7. Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        # Press 'q' or ESC to close
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()