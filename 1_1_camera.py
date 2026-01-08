import sys
import os
import time
import argparse
import numpy as np
import zmq
import pickle
import math
import cv2
from PIL import Image

# --- Aria SDK Imports ---
import aria.sdk as aria
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord

# -----------------------------------------------------------------------------
# Aria Camera Class
# -----------------------------------------------------------------------------
class StreamingClientObserver:
    def __init__(self):
        self.rgb_image = None
        self.timestamp_ns = 0
    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.rgb_image = image
        self.timestamp_ns = record.capture_timestamp_ns

class AriaLiveStreamer:
    def __init__(self, interface="usb", device_ip=None, profile_name="profile28"):
        print(f"\n👓 Initializing Aria Glasses ({interface})...")
        aria.set_log_level(aria.Level.Info)
        self.device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        if device_ip:
            client_config.ip_v4_address = device_ip
        self.device_client.set_client_config(client_config)
        self.device = self.device_client.connect()
        self.streaming_manager = self.device.streaming_manager
        self.streaming_client = self.streaming_manager.streaming_client
        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = profile_name
        if interface == "usb":
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
        streaming_config.security_options.use_ephemeral_certs = True
        self.streaming_manager.streaming_config = streaming_config
        
        # Calibration
        sensors_calib_json = self.streaming_manager.sensors_calibration()
        sensors_calib = device_calibration_from_json_string(sensors_calib_json)
        self.rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
        target_height, target_width = 1408, 1408
        hfov_deg = 110
        focal_length = (target_width / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.dst_calib = get_linear_camera_calibration(target_width, target_height, focal_length, "camera-rgb")
        
        # Start
        self.streaming_manager.start_streaming()
        config = self.streaming_client.subscription_config
        config.subscriber_data_type = aria.StreamingDataType.Rgb
        self.streaming_client.subscription_config = config
        self.observer = StreamingClientObserver()
        self.streaming_client.set_streaming_client_observer(self.observer)
        self.streaming_client.subscribe()
        print("✅ Aria streaming started.")

    def get_latest_frame(self):
        raw_img = self.observer.rgb_image
        if raw_img is None: return None
        undistorted_img = distort_by_calibration(raw_img, self.dst_calib, self.rgb_calib)
        rotated_img = np.rot90(undistorted_img, -1)
        return Image.fromarray(np.ascontiguousarray(rotated_img))

    def stop(self):
        try:
            self.streaming_client.unsubscribe()
            self.streaming_manager.stop_streaming()
            self.device_client.disconnect(self.device)
        except: pass

# -----------------------------------------------------------------------------
# Main Loop (Publisher)
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", dest="streaming_interface", type=str, default="usb")
    parser.add_argument("--profile", dest="profile_name", type=str, default="profile28")
    parser.add_argument("--device-ip", help="Aria IP for wifi")
    parser.add_argument("--port", type=str, default="5555", help="ZMQ Port")
    parser.add_argument("--show", action="store_true", help="Show video stream locally")
    args = parser.parse_args()

    # 1. Setup ZMQ Publisher
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{args.port}")
    print(f"📡 Image Publisher started on port {args.port}")

    # 2. Setup Camera
    try:
        camera = AriaLiveStreamer(
            interface=args.streaming_interface,
            device_ip=args.device_ip,
            profile_name=args.profile_name
        )
    except Exception as e:
        print(f"❌ Failed to initialize Aria: {e}")
        return

    print("🚀 Streaming images... Press Ctrl+C to stop.")
    
    try:
        while True:
            # Get PIL Image
            img = camera.get_latest_frame()
            
            if img is not None:
                # Serialize and Send
                data = pickle.dumps(img)
                socket.send(data)

                # Visualization Logic
                if args.show:
                    # Convert PIL image (RGB) to numpy
                    open_cv_image = np.array(img)
                    # Convert RGB to BGR (OpenCV format)
                    open_cv_image = open_cv_image[:, :, ::-1].copy()
                    
                    # Resize to 240x240
                    open_cv_image = cv2.resize(open_cv_image, (240, 240))
                    
                    cv2.imshow("Aria Live Stream", open_cv_image)
                    cv2.waitKey(1)

                # Cap at ~30 FPS
                time.sleep(0.03)
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        camera.stop()
        socket.close()
        context.term()
        if args.show:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()