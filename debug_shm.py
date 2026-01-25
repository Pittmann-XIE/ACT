import time
import cv2
import numpy as np
import struct
from multiprocessing import shared_memory, resource_tracker

# --- CONFIG ---
ARIA_SHM = "aria_stream_v1"
RS_SHM = "realsense_stream_v1"
SHAPE = (480, 640, 3)

class SharedMemoryViewer:
    def __init__(self, shm_name, shape=SHAPE):
        self.shm_name = shm_name
        self.shape = shape
        self.shm = None
        self.connected = False
        self.last_frame_id = -1
        self.frame_id = 0
        
        # FPS Tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Placeholder (Black with Text)
        self.placeholder = np.zeros(shape, dtype=np.uint8)
        cv2.putText(self.placeholder, f"Waiting for {shm_name}...", 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def _connect(self):
        try:
            self.shm = shared_memory.SharedMemory(create=False, name=self.shm_name)
            
            # --- PREVENT CRASH: Tell Python NOT to delete this file ---
            try:
                resource_tracker.unregister(self.shm._name, 'shared_memory')
            except KeyError: pass
            # ---------------------------------------------------------

            self.connected = True
            print(f"✅ Connected to {self.shm_name}")
        except FileNotFoundError:
            self.connected = False

    def get_frame(self):
        # Auto-reconnect
        if not self.connected:
            self._connect()
            if not self.connected:
                return self.placeholder, 0, False

        try:
            # 1. Read Frame ID (First 8 bytes)
            version_start = struct.unpack_from('Q', self.shm.buf, 0)[0]
            
            # 2. Copy Data
            img_buffer = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf, offset=8)
            img = img_buffer.copy()
            
            # 3. Read Frame ID Again (Integrity Check)
            version_end = struct.unpack_from('Q', self.shm.buf, 0)[0]
            
            if version_start != version_end or version_start == 0:
                return self.placeholder, 0, False # Torn frame
            
            self.frame_id = version_start
            return img, version_start, True

        except Exception:
            self.connected = False
            try: self.shm.close() 
            except: pass
            return self.placeholder, 0, False

    def update_fps(self):
        """Calculates FPS based on NEW frames received."""
        # Only count if we received a NEW frame ID
        if self.frame_id > self.last_frame_id:
            self.fps_counter += (self.frame_id - self.last_frame_id)
            self.last_frame_id = self.frame_id
        
        # Update every 1.0 second
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (time.time() - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = time.time()
        
        return self.current_fps

    def close(self):
        if self.shm: self.shm.close()

def main():
    print("========================================")
    print("   SHARED MEMORY STREAM DEBUGGER")
    print("   (Shows TRUE Camera FPS)")
    print("========================================")
    
    aria_viewer = SharedMemoryViewer(ARIA_SHM)
    rs_viewer = SharedMemoryViewer(RS_SHM)

    try:
        while True:
            # Get Data
            img_aria, id_aria, aria_ok = aria_viewer.get_frame()
            img_rs, id_rs, rs_ok = rs_viewer.get_frame()
            
            # Update FPS Logic
            fps_aria = aria_viewer.update_fps()
            fps_rs = rs_viewer.update_fps()

            # BGR Conversion
            if aria_ok: img_aria = cv2.cvtColor(img_aria, cv2.COLOR_RGB2BGR)
            if rs_ok:   img_rs = cv2.cvtColor(img_rs, cv2.COLOR_RGB2BGR)

            # --- DRAW LABELS ---
            # Aria Label
            color_a = (0, 255, 0) if fps_aria > 1.0 else (0, 0, 255)
            cv2.putText(img_aria, f"ARIA: {fps_aria:.1f} FPS", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_a, 2)
            
            # RealSense Label
            color_r = (0, 255, 0) if fps_rs > 1.0 else (0, 0, 255)
            cv2.putText(img_rs, f"REALSENSE: {fps_rs:.1f} FPS", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_r, 2)

            # Combine
            combined = np.hstack((img_aria, img_rs))
            
            cv2.imshow("Stream Debugger", combined)

            # Efficient Wait (Limit Viewer to ~60 FPS max to save CPU)
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        aria_viewer.close()
        rs_viewer.close()
        cv2.destroyAllWindows()
        print("Debugger closed.")

if __name__ == "__main__":
    main()