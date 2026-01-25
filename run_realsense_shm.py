# import time
# import sys
# import numpy as np
# import pyrealsense2 as rs
# from multiprocessing import shared_memory
# import struct

# # --- CONFIG ---
# SHM_NAME = "realsense_stream_v1"
# IMG_WIDTH = 640
# IMG_HEIGHT = 480
# IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) 
# IMG_SIZE = np.prod(IMG_SHAPE)
# TOTAL_SIZE = IMG_SIZE + 8 

# def main():
#     print(f"[RS-Sender] Process started. Press Ctrl+C to stop completely.")
    
#     # --- 1. SETUP SHARED MEMORY (PERSISTENT) ---
#     # We do this OUTSIDE the retry loop so the memory stays valid even if the camera resets.
#     try:
#         shm = shared_memory.SharedMemory(create=True, size=TOTAL_SIZE, name=SHM_NAME)
#         print(f"[RS-Sender] Created Shared Memory '{SHM_NAME}'")
#     except FileExistsError:
#         shm = shared_memory.SharedMemory(create=False, name=SHM_NAME)
#         print(f"[RS-Sender] Attached to existing Shared Memory '{SHM_NAME}'")

#     # Buffer wrapper (skipping first 8 bytes)
#     shm_img_buffer = np.ndarray(IMG_SHAPE, dtype=np.uint8, buffer=shm.buf, offset=8)
    
#     frame_global_counter = 0

#     try:
#         # --- 2. RETRY LOOP (KEEPS SCRIPT ALIVE) ---
#         while True:
#             pipeline = None
#             try:
#                 print("[RS-Sender] Initializing RealSense Driver...")
                
#                 # Context & Reset
#                 ctx = rs.context()
#                 if len(ctx.query_devices()) > 0:
#                     for dev in ctx.query_devices():
#                         dev.hardware_reset()
#                 # Give the USB bus time to re-enumerate after reset
#                 time.sleep(2.0) 

#                 pipeline = rs.pipeline()
#                 config = rs.config()
#                 config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.rgb8, 30)
                
#                 pipeline.start(config)
#                 print("[RS-Sender] Streaming active.")

#                 # --- 3. STREAM LOOP ---
#                 while True:
#                     # Wait for frames (throws RuntimeError on timeout)
#                     frames = pipeline.wait_for_frames(timeout_ms=5000)
#                     color_frame = frames.get_color_frame()
#                     if not color_frame: continue

#                     img = np.asanyarray(color_frame.get_data())
                    
#                     frame_global_counter += 1
                    
#                     # Write Image
#                     shm_img_buffer[:] = img[:]
#                     # Update Counter
#                     struct.pack_into('Q', shm.buf, 0, frame_global_counter)

#             except Exception as e:
#                 # CATCH CRASHES HERE (Don't exit!)
#                 print(f"⚠️ [RS-Sender] Camera Error: {e}")
#                 print("[RS-Sender] Retrying in 3 seconds...")
#                 time.sleep(3)
            
#             finally:
#                 # Clean up camera resources before retrying
#                 if pipeline:
#                     try: 
#                         pipeline.stop()
#                     except: 
#                         pass

#     except KeyboardInterrupt:
#         print("\n[RS-Sender] Stopping by user request...")
    
#     finally:
#         # Only unlink SHM when the User explicitly kills the script
#         try:
#             shm.close()
#             shm.unlink()
#             print("[RS-Sender] Shared Memory Unlinked ✅")
#         except Exception:
#             pass

# if __name__ == "__main__":
#     main()




### two realsense cameras support with selection at start
import time
import sys
import numpy as np
import pyrealsense2 as rs
from multiprocessing import shared_memory
import struct

# --- CONFIG ---
SHM_NAME = "realsense_stream_v1"
IMG_WIDTH = 640
IMG_HEIGHT = 480
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) 
IMG_SIZE = np.prod(IMG_SHAPE)
TOTAL_SIZE = IMG_SIZE + 8 

def select_device():
    """
    Lists connected RealSense devices and asks user to select one.
    Returns the Serial Number of the selected device.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("❌ [RS-Sender] No RealSense devices found. Exiting.")
        sys.exit(1)

    print("\n--- Available RealSense Devices ---")
    device_serials = []
    
    for i, dev in enumerate(devices):
        try:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            print(f"[{i}] {name} (Serial: {serial})")
            device_serials.append(serial)
        except Exception as e:
            print(f"[{i}] Unknown Device ({e})")
            device_serials.append(None)

    print("-----------------------------------")
    
    while True:
        try:
            selection = input(f"Select camera index (0-{len(devices)-1}): ").strip()
            idx = int(selection)
            if 0 <= idx < len(devices):
                print(f"[RS-Sender] Selected Serial: {device_serials[idx]}")
                return device_serials[idx]
            else:
                print("Invalid index. Try again.")
        except ValueError:
            print("Please enter a number.")

def main():
    print(f"[RS-Sender] Process started. Press Ctrl+C to stop completely.")
    
    # --- 1. SETUP SHARED MEMORY (PERSISTENT) ---
    try:
        shm = shared_memory.SharedMemory(create=True, size=TOTAL_SIZE, name=SHM_NAME)
        print(f"[RS-Sender] Created Shared Memory '{SHM_NAME}'")
    except FileExistsError:
        shm = shared_memory.SharedMemory(create=False, name=SHM_NAME)
        print(f"[RS-Sender] Attached to existing Shared Memory '{SHM_NAME}'")

    # Buffer wrapper (skipping first 8 bytes)
    shm_img_buffer = np.ndarray(IMG_SHAPE, dtype=np.uint8, buffer=shm.buf, offset=8)
    
    frame_global_counter = 0

    # --- 2. SELECT CAMERA ---
    # We do this once at startup
    target_serial = select_device()

    try:
        # --- 3. RETRY LOOP (KEEPS SCRIPT ALIVE) ---
        while True:
            pipeline = None
            try:
                print(f"[RS-Sender] Initializing Driver for Serial {target_serial}...")
                
                # Context & Reset (Targeted)
                ctx = rs.context()
                devices = ctx.query_devices()
                
                device_found = False
                for dev in devices:
                    # Check if this is our target device
                    if dev.get_info(rs.camera_info.serial_number) == target_serial:
                        print("[RS-Sender] Performing Hardware Reset on target camera...")
                        dev.hardware_reset()
                        device_found = True
                        break
                
                if not device_found:
                    print(f"⚠️ [RS-Sender] Target camera {target_serial} not detected. Waiting...")
                    time.sleep(2)
                    continue

                # Give the USB bus time to re-enumerate after reset
                time.sleep(2.0) 

                pipeline = rs.pipeline()
                config = rs.config()
                
                # IMPORTANT: Select specific device by serial number
                config.enable_device(target_serial)
                
                config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.rgb8, 30)
                
                pipeline.start(config)
                print(f"[RS-Sender] Streaming active on {target_serial}.")

                # --- 4. STREAM LOOP ---
                while True:
                    # Wait for frames (throws RuntimeError on timeout)
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                    color_frame = frames.get_color_frame()
                    if not color_frame: continue

                    img = np.asanyarray(color_frame.get_data())
                    
                    frame_global_counter += 1
                    
                    # Write Image
                    shm_img_buffer[:] = img[:]
                    # Update Counter
                    struct.pack_into('Q', shm.buf, 0, frame_global_counter)

            except Exception as e:
                # CATCH CRASHES HERE (Don't exit!)
                print(f"⚠️ [RS-Sender] Camera Error: {e}")
                print("[RS-Sender] Retrying in 3 seconds...")
                time.sleep(3)
            
            finally:
                # Clean up camera resources before retrying
                if pipeline:
                    try: 
                        pipeline.stop()
                    except: 
                        pass

    except KeyboardInterrupt:
        print("\n[RS-Sender] Stopping by user request...")
    
    finally:
        # Only unlink SHM when the User explicitly kills the script
        try:
            shm.close()
            shm.unlink()
            print("[RS-Sender] Shared Memory Unlinked ✅")
        except Exception:
            pass

if __name__ == "__main__":
    main()


