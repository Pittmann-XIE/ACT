import numpy as np
import argparse
import sys
import os
import time
import csv
from scipy.spatial.transform import Rotation as R

# --- Robot Interface Imports ---
# Ensure these files are in your python path or current directory
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

ROBOT_IP = "192.168.1.23"  # Robot IP

# -----------------------------------------------------------------------------
# Math & Transformation Constants
# -----------------------------------------------------------------------------

import numpy as np

# -----------------------------------------------------------------------------
# Math & Transformation Constants
# -----------------------------------------------------------------------------

# 1. Original Calibration Data (As provided previously)
R_TCP_DEVICE_RAW = np.array([
    [ 0.99920421, -0.03675654,  0.01548867],
    [ 0.03020608,  0.95091268,  0.30798161],
    [-0.02604871, -0.30726867,  0.95126622]
])
t_TCP_DEVICE_VEC = np.array([0.01260316, -0.09558874, 0.06506849])

# 2. Define the 90 degree Anti-Clockwise Rotation around Z
# Matrix for RotZ(90):
# [[ cos(90), -sin(90), 0],
#  [ sin(90),  cos(90), 0],
#  [       0,        0, 1]]
R_CORRECTION = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0]
])

# 3. Apply Correction
# R_new = R_old @ R_correction (Post-multiply rotates the local frame)
R_TCP_DEVICE_MAT = R_TCP_DEVICE_RAW @ R_CORRECTION

# Translation remains the same (origin hasn't moved, just twisted)
# t_TCP_DEVICE_VEC stays as is.

# 4. Construct Final 4x4 Matrix
T_TCP_DEVICE = np.eye(4)
T_TCP_DEVICE[:3, :3] = R_TCP_DEVICE_MAT
T_TCP_DEVICE[:3, 3] = t_TCP_DEVICE_VEC

# T_device_tcp (Inverse)
T_DEVICE_TCP = np.linalg.inv(T_TCP_DEVICE)

def rotvec_to_matrix(rx, ry, rz):
    return R.from_rotvec([rx, ry, rz]).as_matrix()

def matrix_to_rotvec(mat):
    return R.from_matrix(mat).as_rotvec()

# -----------------------------------------------------------------------------
# Main Execution Class
# -----------------------------------------------------------------------------

class TrajectoryExecutor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.T_base_device0 = None # Will be set after robot moves to start
        
        # 1. Robot Constants
        # Initial Pose [tx, ty, tz, qx, qy, qz, qw]
        self.initial_pose = [0.2662107156494109, 0.12091324814367804, 0.3483654814212416, np.float64(-0.7776310683923047), np.float64(0.2974687702795033), np.float64(-0.21657091021220123), np.float64(0.5098031904856045)]
        
        # 2. Initialize Robot Connection
        try:
            print(f"\n🔗 Connecting to robot at {ROBOT_IP}...")
            self.rtde_c = RTDEControlInterface(ROBOT_IP)
            self.rtde_r = RTDEReceiveInterface(ROBOT_IP)
            print("✅ Robot connected successfully!\n")
        except Exception as e:
            print(f"⚠️  Error: Could not connect to robot: {e}")
            sys.exit(1)

        # 3. Load Trajectory
        self.trajectory_steps = self._load_trajectory(self.csv_path)

    def _load_trajectory(self, path):
        """Reads CSV. Assumes format: x, y, z, qx, qy, qz, qw (Relative Poses)"""
        print(f"📂 Loading trajectory from {path}...")
        steps = []
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        vals = [float(x) for x in row]
                    except ValueError:
                        continue
                    
                    if len(vals) < 7:
                        continue

                    # Construct Relative Matrix (T_device0 -> Device)
                    t_rel = vals[:3]
                    quat_rel = vals[3:7] # [qx, qy, qz, qw]
                    
                    T_rel = np.eye(4)
                    T_rel[:3, :3] = R.from_quat(quat_rel).as_matrix()
                    T_rel[:3, 3] = t_rel
                    
                    steps.append(T_rel)
            
            print(f"✅ Loaded {len(steps)} steps.")
            return steps
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            sys.exit(1)

    def move_to_start(self):
        """Moves robot to the hardcoded initial pose."""
        print("\n" + "=" * 50)
        print("🚀 MOVING TO START POSE")
        print("=" * 50)
        
        tx, ty, tz, qx, qy, qz, qw = self.initial_pose
        rx, ry, rz = R.from_quat([qx, qy, qz, qw]).as_rotvec()
        target_pose = [tx, ty, tz, rx, ry, rz]
        
        try:
            self.rtde_c.moveL(target_pose, 0.2, 0.2)
            time.sleep(1.0) # Wait for settling
            print("✅ Reached Initial Pose.")
        except Exception as e:
            print(f"❌ Move failed: {e}")
            sys.exit(1)

    def record_reference_frame(self):
        """
        Reads the ACTUAL current TCP pose from the robot and 
        establishes T_base_device0.
        This ensures T_rel=Identity results in 0 movement.
        """
        print("📏 Reading actual robot pose to establish Reference Frame...")
        
        # 1. Get Actual TCP Pose [x, y, z, rx, ry, rz]
        actual_tcp = self.rtde_r.getActualTCPPose()
        
        # 2. Convert to Matrix (T_base_tcp)
        T_base_tcp = np.eye(4)
        T_base_tcp[:3, :3] = rotvec_to_matrix(actual_tcp[3], actual_tcp[4], actual_tcp[5])
        T_base_tcp[:3, 3] = actual_tcp[:3]

        # 3. Compute T_base_device0 = T_base_tcp * T_tcp_device
        self.T_base_device0 = T_base_tcp @ T_TCP_DEVICE
        print("✅ Reference Frame Established.")

    def run_trajectory(self):
        # 1. Move to start
        self.move_to_start()
        
        # 2. Establish Reference Frame based on ACTUAL position
        self.record_reference_frame()

        print("\n" + "=" * 50)
        print("▶️  EXECUTING TRAJECTORY")
        print("=" * 50)
        
        input("Press ENTER to start execution...")

        for i, T_rel in enumerate(self.trajectory_steps):
            
            # 1. Compute T_base_device (Apply relative transform to reference)
            # T_base_device = T_base_device0 * T_rel
            T_base_device = self.T_base_device0 @ T_rel

            # 2. Compute T_base_tcp (Transform back to Robot TCP frame)
            # T_base_tcp = T_base_device * T_device_tcp
            T_base_tcp = T_base_device @ T_DEVICE_TCP

            # 3. Convert to UR5 format
            x, y, z = T_base_tcp[:3, 3]
            rx, ry, rz = matrix_to_rotvec(T_base_tcp[:3, :3])
            
            target_pose = [x, y, z, rx, ry, rz]
            
            if i % 10 == 0:
                print(f"Step {i}/{len(self.trajectory_steps)}")

            try:
                # Use servoL or moveL. 
                # If the trajectory is high frequency (e.g. 30Hz), use servoL.
                # If it is sparse waypoints, use moveL.
                # Here assuming relatively dense CSV, using moveL with low blocking.
                self.rtde_c.moveL(target_pose, 0.5, 0.8)
            except Exception as e:
                print(f"❌ Execution error at step {i}: {e}")
                break

        print("\n✅ Trajectory Execution Complete.")

    def __del__(self):
        try:
            if hasattr(self, 'rtde_c') and self.rtde_c: 
                self.rtde_c.stopScript()
                self.rtde_c.disconnect()
            if hasattr(self, 'rtde_r') and self.rtde_r: 
                self.rtde_r.disconnect()
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Execute Relative Trajectory from CSV")
    parser.add_argument('--csv', type=str, default='trajectory_world.csv', help='Path to trajectory csv (x,y,z,qx,qy,qz,qw)')
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: File {args.csv} not found.")
        return

    executor = TrajectoryExecutor(csv_path=args.csv)
    executor.run_trajectory()

if __name__ == '__main__':
    main()