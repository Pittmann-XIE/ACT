import sys
import os
import time
import numpy as np
import cv2
from multiprocessing import shared_memory
import struct

# Import Aria SDK
sys.path.append(os.getcwd())
import aria.sdk as aria

# --- CONFIG ---
SHM_NAME = "aria_stream_v1"
IMG_SHAPE = (480, 640, 3) 
IMG_SIZE = np.prod(IMG_SHAPE)
TOTAL_SIZE = IMG_SIZE + 8 

def main():
    print(f"[Sender] Process started. Press Ctrl+C to stop completely.")
    
    # --- 1. SETUP SHARED MEMORY ---
    try:
        shm = shared_memory.SharedMemory(create=True, size=TOTAL_SIZE, name=SHM_NAME)
        print(f"[Sender] Created Shared Memory '{SHM_NAME}'")
    except FileExistsError:
        shm = shared_memory.SharedMemory(create=False, name=SHM_NAME)
        print(f"[Sender] Attached to existing Shared Memory '{SHM_NAME}'")

    shm_img_buffer = np.ndarray(IMG_SHAPE, dtype=np.uint8, buffer=shm.buf, offset=8)
    global_frame_count = 0 

    # --- UPDATED OBSERVER (Lightweight) ---
    class Observer:
        def __init__(self):
            self.last_img = None
            self.last_update_time = time.time()
            self.new_frame_available = False

        def on_image_received(self, img, record):
            # ONLY store the data. Do NOT process here.
            self.last_img = img
            self.new_frame_available = True
            self.last_update_time = time.time()

    try:
        while True:
            device = None
            client = None
            manager = None
            
            try:
                print("[Sender] Initializing Aria Connection...")
                aria.set_log_level(aria.Level.Error) 
                
                device_client = aria.DeviceClient()
                client_config = aria.DeviceClientConfig()
                device_client.set_client_config(client_config)
                
                device = device_client.connect()
                manager = device.streaming_manager
                
                # Cleanup previous state if needed
                try:
                    manager.stop_streaming()
                    time.sleep(2.0) # Increased wait time for safety
                except:
                    pass

                # Configure
                config = aria.StreamingConfig()
                config.profile_name = "profile28"
                config.streaming_interface = aria.StreamingInterface.Usb
                config.security_options.use_ephemeral_certs = True
                manager.streaming_config = config
                
                manager.start_streaming()
                
                client = manager.streaming_client
                sub_config = client.subscription_config
                sub_config.subscriber_data_type = aria.StreamingDataType.Rgb
                client.subscription_config = sub_config

                observer = Observer()
                client.set_streaming_client_observer(observer)
                client.subscribe()

                print(f"[Sender] Streaming active.")
                
                # --- PROCESSING LOOP ---
                observer.last_update_time = time.time()
                
                while True:
                    # Check Watchdog
                    if time.time() - observer.last_update_time > 8.0:
                        raise RuntimeError("Watchdog: No frames received for 8 seconds.")

                    # PROCESS FRAME HERE (Decoupled from SDK thread)
                    if observer.new_frame_available and observer.last_img is not None:
                        # 1. Grab reference and clear flag
                        current_img = observer.last_img
                        observer.new_frame_available = False
                        
                        # 2. Heavy Processing (Rotate/Resize)
                        # Note: np.ascontiguousarray might be needed if rotation creates non-contiguous memory
                        rot = np.rot90(current_img, -1)
                        final_img = cv2.resize(rot, (640, 480))
                        
                        # 3. Write to Shared Memory
                        shm_img_buffer[:] = final_img[:]
                        global_frame_count += 1
                        struct.pack_into('Q', shm.buf, 0, global_frame_count)
                    
                    # Sleep briefly to prevent CPU hogging in the loop
                    time.sleep(0.001)

            except Exception as e:
                print(f"⚠️ [Sender] Connection lost/failed: {e}")
                print("[Sender] Cleaning up and waiting 5s before retry...")
            
            finally:
                if client:
                    try: client.unsubscribe()
                    except: pass
                if manager:
                    try: manager.stop_streaming()
                    except: pass
                if device_client and device:
                    try: device_client.disconnect(device)
                    except: pass
                
                # Critical wait for state 7 clearance
                time.sleep(5.0) 

    except KeyboardInterrupt:
        print("\n[Sender] Stopping...")
    
    finally:
        try:
            shm.close()
            shm.unlink()
        except:
            pass

if __name__ == "__main__":
    main()