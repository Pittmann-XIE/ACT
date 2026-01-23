import aria.sdk as aria
import numpy as np
import cv2
import time
from PIL import Image
import sys

# Default settings (match your main script)
INTERFACE = "usb"          # or "wifi"
PROFILE_NAME = "profile28" # Ensure this matches your device config

class AriaCamera:
    def __init__(self, interface="usb", profile_name="profile28", device_ip=None):
        print(f"[Aria] Initializing SDK ({interface})...")
        aria.set_log_level(aria.Level.Info)
        
        # 1. Setup Client
        self.device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        if device_ip: 
            client_config.ip_v4_address = device_ip
        self.device_client.set_client_config(client_config)
        
        # 2. Connect
        try:
            print("[Aria] Connecting to device...")
            self.device = self.device_client.connect()
            print("[Aria] Device Connected!")
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            print("   -> Check USB cable or if 'rs-enumerate-devices' conflicts.")
            raise e
        
        # 3. Configure Stream
        self.streaming_manager = self.device.streaming_manager
        self.streaming_client = self.streaming_manager.streaming_client
        
        config = aria.StreamingConfig()
        config.profile_name = profile_name
        
        if interface == "usb": 
            config.streaming_interface = aria.StreamingInterface.Usb
        elif interface == "wifi":
            config.streaming_interface = aria.StreamingInterface.Wifi
            
        config.security_options.use_ephemeral_certs = True
        self.streaming_manager.streaming_config = config
        
        # 4. Start Streaming
        print("[Aria] Starting stream...")
        self.streaming_manager.start_streaming()
        
        # 5. Subscribe
        sub_config = self.streaming_client.subscription_config
        sub_config.subscriber_data_type = aria.StreamingDataType.Rgb
        self.streaming_client.subscription_config = sub_config
        
        self.observer = self._Observer()
        self.streaming_client.set_streaming_client_observer(self.observer)
        self.streaming_client.subscribe()
        print("[Aria] Subscribed to RGB.")

    class _Observer:
        def __init__(self): 
            self.image = None
            self.frame_count = 0
            
        def on_image_received(self, img, record): 
            self.image = img
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                print(f"[Aria] Received frame #{self.frame_count}")

    def get_frame(self):
        raw = self.observer.image
        if raw is None: return None
        # Rotate 90 deg clockwise (Raw format requirement)
        rot = np.rot90(raw, -1) 
        # Convert to BGR for OpenCV display
        bgr = cv2.cvtColor(rot, cv2.COLOR_RGB2BGR)
        return bgr

    def stop(self):
        print("[Aria] Stopping...")
        try:
            self.streaming_client.unsubscribe()
            self.streaming_manager.stop_streaming()
            self.device_client.disconnect(self.device)
        except Exception as e:
            print(f"Error stopping: {e}")

def main():
    try:
        cam = AriaCamera(interface=INTERFACE, profile_name=PROFILE_NAME)
    except Exception:
        sys.exit(1)

    print("\n✅ Stream started. Press 'q' to quit.\n")
    
    while True:
        frame = cam.get_frame()
        
        if frame is None:
            print("Waiting for data...")
            time.sleep(0.1)
            continue
            
        # Resize for easier viewing
        preview = cv2.resize(frame, (640, 480))
        cv2.imshow("Aria Test (Press q to quit)", preview)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()