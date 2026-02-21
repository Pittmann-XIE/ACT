'''
This script is used to run inference of Action chunking transformer model on ur3 robot through pybullet in real world. The execution of actions should be collision free. 

Requiements:
- ur3 robot model in Pybullet, take the run_ur3_spacemouse.py as an example, keep the settings of the ur3 robot the same(home position, the size and positions of the obstacles).
- inference with ACT model, take evaluate.py as an example, which is designed for inference with ACT with another type of robot.
- args.mode: manual and auto  execution. maunal means, for each execution of the action, the robot should wait for user input to proceed to the next action. auto means, the robot will execute all the predicted actions without user input.
- activate action chunking of the predicted results, when args.action_chunking is True

Inputs:
- file path of tretrained ACT model checkpoint
- video stream from aria glasses, take inference_1_real_semiauto.py as an example of how to get the video stream.
- video stream from realsense D435, you have to google it how to get realsense video stream.
- robot current state, including 6 joint angles of ur3 robot and gripper state(open or closed), it an be gotten from pybullet directly.
- mean and std for normalization of joint angles. 
- ur3 robot IP address

ACT model 
-inputs:
    -- one image from aria glasses(use the distort images, profile28) and one image realsense D435 (the size is 640x480,30fps). both images should be normalized.
    -- robot current state, including 6 joint angles of ur3 robot and one gripper state(open or closed, open is 1,  close is 0). the 6 joint angles should be in radian and normalized by the provide statics(mean, std)
-outputs:
    -- predicted action sequence, including 6 joint angles and one gripper action(open or closed). the 6 joint angles should be de-normalized back to radian by the  provide statics(mean, std)
    -- execute the predicted action sequence with or without action chunking in pybullet, the execution should be collision free. when args.mode=manual, the execution of the action should first be verified in the pybullet that it's collision free and show it future state with ghost robot, then let user to confirm to execute the real robot. 
'''

import torch
import numpy as np
import os
import time
import argparse
import sys
import cv2
import pickle
import traceback
import threading
import copy
from PIL import Image
import pybullet as p
import pybullet_data
import mplib
import pyrealsense2 as rs
import torchvision.transforms as transforms
import math
from scipy.spatial.transform import Rotation as R

# --- Project Imports ---
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import aria.sdk as aria
from training.utils_joint_1_step import make_policy
from config.config_joint_1_step import POLICY_CONFIG, TASK_CONFIG

# --- Driver Imports ---
try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except ImportError:
    print("⚠️ WARNING: RTDE Drivers not found. Sim-Only.")
    RTDEControlInterface = None
    RTDEReceiveInterface = None

sys.path.append('/home/pengtao/ws_ros2humble-main_lab/ur3_vla_teleop/')

try:
    from drivers_new.wsg_driver import WSGGripperDriver
except ImportError:
    print("⚠️ WARNING: WSG Driver not found.")
    WSGGripperDriver = None

# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------

# --- SAFETY LIMITS ---
MANUAL_SPEED_RAD_S = 0.1    # Safe speed for manual steps
MANUAL_ACCEL_RAD_S2 = 0.05   # Safe accel for manual steps
SERVO_DT = 0.1             # Loop time for Auto mode (higher = slower/smoother)
SERVO_LOOKAHEAD = 0.1
SERVO_GAIN = 300
ACTIVE_HORIZON = 100       # Max past predictions to aggregate

# --- HARDWARE ---
ROBOT_IP = "10.0.0.1" 
GRIPPER_PORT = "/dev/ttyACM0" 

# --- CAMERAS ---
# Serial numbers for RealSense cameras: [Camera 1, Camera 2]
CAM_SERIALS = ['104122061227', '105422061000']
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# Image Normalization Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- ENV CONSTANTS (MATCHING ) ---
INIT_JOINTS_DEG = [47.22, -94.58, -125.38, -0.0, -271.32, 4.32]
SIM_ROBOT_WORLD_ROT_Z = np.pi 

# Obstacles
WALL_POS_WORLD = [-0.14, -0.14, 0.5]  
WALL_ROT_Z_DEG = 45
WALL_SIZE = [0.02, 1.0, 1.0] 

TABLE_SIZE = [1.0, 1.0, 0.1]
TABLE_POS_WORLD = [0, 0, -TABLE_SIZE[2]/2-0.01]
TABLE_ROT_Z_DEG = 0

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (MATCHING)
# -----------------------------------------------------------------------------
def get_object_transforms(pos_world_arr, rot_z_deg):
    pos_world = np.array(pos_world_arr)
    r_obj_local = R.from_euler('z', rot_z_deg, degrees=True)
    r_world_robot = R.from_euler('z', -SIM_ROBOT_WORLD_ROT_Z, degrees=False)
    pos_robot = r_world_robot.apply(pos_world)
    quat_robot = (r_world_robot * r_obj_local).as_quat()
    quat_world = r_obj_local.as_quat()
    return pos_robot, quat_robot, pos_world, quat_world

class ThreadedSensor:
    def __init__(self, sensor_class, *args, **kwargs):
        self.sensor_name = sensor_class.__name__
        try:
            self.sensor = sensor_class(*args, **kwargs)
        except Exception as e:
            print(f"❌ Critical Error Initializing {self.sensor_name}: {e}")
            raise e
            
        self.lock = threading.Lock()
        self.latest_data = None
        self.timestamp = 0
        self.running = False
        self.thread = None
        
        # --- FPS Counter Variables ---
        self.measured_fps = 0.0
        self._frame_counter = 0
        self._last_fps_time = time.time()
        # -----------------------------

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            try:
                data = self.sensor.get_frame()
                if data is not None:
                    with self.lock:
                        self.latest_data = data
                        self.timestamp = time.time()
                        
                        # --- FPS Calculation ---
                        self._frame_counter += 1
                        now = time.time()
                        elapsed = now - self._last_fps_time
                        if elapsed >= 1.0: # Update every 1 second
                            self.measured_fps = self._frame_counter / elapsed
                            self._frame_counter = 0
                            self._last_fps_time = now
                        # -----------------------
                        
            except Exception as e:
                print(f"❌ {self.sensor_name} Thread Error: {e}")
                time.sleep(1.0)
            time.sleep(0.001) 

    def get_latest(self):
        with self.lock:
            if self.latest_data is None: return None, 0, 0.0
            return self.latest_data.copy(), self.timestamp, self.measured_fps # Return FPS too

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
        try:
            self.sensor.stop()
        except: pass

# -----------------------------------------------------------------------------
# 2. Camera Classes
# -----------------------------------------------------------------------------

from multiprocessing import shared_memory, resource_tracker
import struct

class SharedMemoryReceiver:
    def __init__(self, shm_name, shape=(480, 640, 3)):
        self.shm_name = shm_name
        self.shape = shape
        self.size = np.prod(shape)
        self.connected = False
        self.shm = None
        self._connect()

    # def _connect(self):
    #     try:
    #         self.shm = shared_memory.SharedMemory(create=False, name=self.shm_name)
    #         self.connected = True
    #         print(f"[{self.shm_name}] Connected to Shared Memory.")
    #     except FileNotFoundError:
    #         print(f"[{self.shm_name}] Waiting for sender...")
    def _connect(self):
        try:
            self.shm = shared_memory.SharedMemory(create=False, name=self.shm_name)
            
            # --- FIX: Prevent resource tracker from destroying SHM on exit ---
            # This tells the process: "I am using this, but I don't own it. Do not unlink it."
            resource_tracker.unregister(self.shm._name, 'shared_memory')
            # ---------------------------------------------------------------

            self.connected = True
            print(f"[{self.shm_name}] Connected to Shared Memory.")
        except FileNotFoundError:
            print(f"[{self.shm_name}] Waiting for sender...")

    def get_frame(self):
        # Auto-reconnect logic
        if not self.connected:
            self._connect()
            if not self.connected: return None

        try:
            # 1. Read Version (Start)
            version_start = struct.unpack_from('Q', self.shm.buf, 0)[0]
            
            # 2. Copy Data
            img_buffer = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf, offset=8)
            img_copy = img_buffer.copy()
            
            # 3. Read Version (End)
            version_end = struct.unpack_from('Q', self.shm.buf, 0)[0]
            
            # 4. Validate (Optimistic Locking)
            if version_start != version_end or version_start == 0:
                return None
            return img_copy

        except Exception:
            self.connected = False
            return None

    def stop(self):
        if self.shm: self.shm.close()

class RealSenseCamera:
    """Direct Interface for RealSense Cameras via Serial Number"""
    def __init__(self, serial_number, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS):
        self.serial = serial_number
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        
        print(f"[RealSense] Initializing camera {serial_number}...")
        self.pipeline.start(config)
        print(f"[RealSense] Camera {serial_number} started.")

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def stop(self):
        self.pipeline.stop()

# -----------------------------------------------------------------------------
# 3. Robot System
# -----------------------------------------------------------------------------

class RobotSystem:
    def __init__(self, robot_ip, gripper_port):
        # PyBullet Setup
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # CAMERA: Exactly matching run_ur3_spacemouse.py
        p.resetDebugVisualizerCamera(1.2, 90, -30, [0, 0, 0])
        
        p.loadURDF("plane.urdf")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(current_dir, "ur3_gripper.urdf")
        self.srdf_path = os.path.join(current_dir, "ur3_gripper.srdf")
        
        # ROBOT LOAD: Orientation matching run_ur3_spacemouse.py
        sim_orn = p.getQuaternionFromEuler([0, 0, SIM_ROBOT_WORLD_ROT_Z])
        self.sim_robot_id = p.loadURDF(self.urdf_path, [0,0,0], baseOrientation=sim_orn, useFixedBase=1)
        self.ghost_robot_id = p.loadURDF(self.urdf_path, [0,0,0], baseOrientation=sim_orn, useFixedBase=1)
        self._make_ghost_transparent()
        
        # OBSTACLES: Exactly matching run_ur3_spacemouse.py
        self._setup_obstacles()
        
        self.planner = mplib.Planner(urdf=self.urdf_path, srdf=self.srdf_path, move_group="tool0")
        
        # Real Robot
        self.rtde_c = None
        self.rtde_r = None
        if RTDEControlInterface:
            try:
                print(f"[Robot] Connecting to {robot_ip}...")
                self.rtde_c = RTDEControlInterface(robot_ip)
                self.rtde_r = RTDEReceiveInterface(robot_ip)
                print("[Robot] Connected!")
            except Exception as e: print(f"❌ Robot Connection failed: {e}")
        
        # Gripper
        self.gripper = None
        if WSGGripperDriver:
            try:
                print(f"[Gripper] Connecting to {gripper_port}...")
                # Initialize driver (which starts the background serial listener)
                self.gripper = WSGGripperDriver(gripper_port)
                # Ensure gripper is referenced/open on startup
                self.gripper.move(1.0)
                print("[Gripper] Connected & Initialized!")
            except Exception as e: 
                print(f"❌ Gripper Connection Failed: {e}")

        # --- DYNAMIC JOINT INDEX LOOKUP (Fix for "Wrong Pose" issues) ---
        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        name_to_idx = {p.getJointInfo(self.sim_robot_id, i)[1].decode('utf-8'): i for i in range(p.getNumJoints(self.sim_robot_id))}
        
        # Check if all joints exist
        try:
            self.joint_indices = [name_to_idx[n] for n in joint_names]
            print(f"[Robot] Mapped Joints Indices: {self.joint_indices}")
        except KeyError as e:
            print(f"❌ Joint Mapping Error: {e}")
            print(f"Available Joints: {list(name_to_idx.keys())}")
            sys.exit(1)

        self.gripper_last_cmd = 1.0
        
        # FORCE SYNC: Ensure sim starts at HOME, not at arbitrary position
        self.go_home_sim_only()

    def go_home_sim_only(self):
        """Resets Simulation-Only Robot to Home Config immediately."""
        home_rad = np.deg2rad(INIT_JOINTS_DEG)
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.sim_robot_id, idx, home_rad[i])
            p.resetJointState(self.ghost_robot_id, idx, home_rad[i])
        p.stepSimulation()
        
        
    def go_home(self):
        """Moves robot to INIT_JOINTS_DEG position (Blocking)"""
        print(f"[Robot] Moving to Home: {INIT_JOINTS_DEG}")
        home_rad = np.deg2rad(INIT_JOINTS_DEG)
        
        # --- ADDED: Open Gripper First ---
        if self.gripper:
            try:
                print("[Gripper] Opening...")
                self.gripper.move(1.0) # 1.0 is Open
                # Optional: Update internal state to match
                self.gripper_last_cmd = 1.0 
                time.sleep(0.5) # Give it a moment to open
            except Exception as e:
                print(f"❌ Gripper Open Failed: {e}")
        # ---------------------------------

        # 1. Real Robot (Move slowly)
        if self.rtde_c:
            try:
                self.rtde_c.moveJ(home_rad, MANUAL_SPEED_RAD_S, MANUAL_ACCEL_RAD_S2)
            except Exception as e:
                print(f"❌ Move Home Failed: {e}")
    
        
        # 2. Sim & Ghost (Snap to home)
        # --- FIXED LINE BELOW ---
        for i, idx in enumerate(self.joint_indices): 
            p.resetJointState(self.sim_robot_id, idx, home_rad[i])
            p.resetJointState(self.ghost_robot_id, idx, home_rad[i])
        p.stepSimulation()
        
        # --- DEBUG LOGGING START ---
        # Capture ACTUAL joints from PyBullet to debug mismatch
        actual_sim_q = []
        for idx in self.joint_indices:
            actual_sim_q.append(p.getJointState(self.sim_robot_id, idx)[0])
        
        print("\n" + "-"*40)
        print("[Debug] PyBullet Home Check:")
        print(f"   Target (Deg): {INIT_JOINTS_DEG}")
        print(f"   Actual (Deg): {np.round(np.rad2deg(actual_sim_q), 2).tolist()}")
        print(f"   Actual (Rad): {np.round(actual_sim_q, 4).tolist()}")
        print("-"*40 + "\n")
        # --- DEBUG LOGGING END ---

        print("[Robot] Reached Home.")

    def _make_ghost_transparent(self):
        # 1. Visual Transparency
        for j in range(p.getNumJoints(self.ghost_robot_id)):
            p.changeVisualShape(self.ghost_robot_id, j, rgbaColor=[0, 1, 1, 0.4])
        
        # 2. DISABLE COLLISIONS (Crucial Fix)
        # Disable collision for the Base Link (-1)
        p.setCollisionFilterGroupMask(self.ghost_robot_id, -1, collisionFilterGroup=0, collisionFilterMask=0)
        
        # Disable collision for all other joints
        for j in range(p.getNumJoints(self.ghost_robot_id)):
            p.setCollisionFilterGroupMask(self.ghost_robot_id, j, collisionFilterGroup=0, collisionFilterMask=0)

    def _setup_obstacles(self):
        # Calculate transformed positions exactly like run_ur3_spacemouse.py
        wall_pos_r, wall_quat_r, wall_pos_w, wall_quat_w = get_object_transforms(WALL_POS_WORLD, WALL_ROT_Z_DEG)
        table_pos_r, table_quat_r, table_pos_w, table_quat_w = get_object_transforms(TABLE_POS_WORLD, TABLE_ROT_Z_DEG)
        
        # Create Wall (Red)
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE]), 
                          basePosition=wall_pos_w, baseOrientation=wall_quat_w,
                          baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE], rgbaColor=[0.8,0.2,0.2,0.5]))
        
        # Create Table (Wood/Brown)
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE]), 
                          basePosition=table_pos_w, baseOrientation=table_quat_w,
                          baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE], rgbaColor=[0.5,0.3,0.1,1.0]))

    def sync_sim_to_real(self):
        q_real = self.get_qpos_real()
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.sim_robot_id, idx, q_real[i])
            p.resetJointState(self.ghost_robot_id, idx, q_real[i])

    def get_qpos_real(self):
        if self.rtde_r:
            q = self.rtde_r.getActualQ()
        else:
            q = [p.getJointState(self.sim_robot_id, i)[0] for i in self.joint_indices]
        
        # UPDATED: Use get_pos() from wsg_driver.py
        if self.gripper:
            try:
                g_state = self.gripper.get_pos() # Returns mm
            except: 
                raise RuntimeError('Get not get gripper distance')
        else: 
            g_state = 110.0
            
        return np.array(q + [g_state])
    
    def check_collision(self, qpos_target):
            res = self.planner.check_for_self_collision(qpos_target[:6])
            if res: return True
            res = self.planner.check_for_env_collision(qpos_target[:6])
            return res

    def update_ghost(self, action):
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.ghost_robot_id, idx, action[i])

    def execute_action(self, action, mode='manual'):
            target_joints = action[:6]
            target_gripper = action[6]

            if self.check_collision(action):
                print("⚠️ COLLISION DETECTED! Stopping.")
                return False

            self.update_ghost(action)
            print(f"\n   [NEXT] Gripper: ({target_gripper:.1f}mm) | Joints: {np.round(target_joints[:], 2)}...")
            
            if mode == 'manual':
                # Visual Feedback
                for j in range(p.getNumJoints(self.ghost_robot_id)):
                    p.changeVisualShape(self.ghost_robot_id, j, rgbaColor=[0, 1, 0, 0.6])
                
                
                print(f"   >>> LOOK AT WINDOW: Press 'g' to Execute, 'q/c' to Skip/Quit.")

                # --- CHANGED: Use OpenCV waitKey instead of input() ---
                # This keeps the camera windows responsive while waiting
                user_approved = False
                while True:
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('g'):
                        user_approved = True
                        break
                    elif key == ord('q') or key == ord('c'):
                        user_approved = False
                        break
                # -----------------------------------------------------
                
                # Reset Ghost Color
                for j in range(p.getNumJoints(self.ghost_robot_id)):
                    p.changeVisualShape(self.ghost_robot_id, j, rgbaColor=[0, 1, 1, 0.4])

                if not user_approved:
                    print("   [Skipped]")
                    return False

            if self.rtde_c:
                try:
                    if mode == 'auto':
                        self.rtde_c.servoJ(target_joints, 0.0, 0.0, SERVO_DT, SERVO_LOOKAHEAD, SERVO_GAIN)
                    else:
                        self.rtde_c.moveJ(target_joints, MANUAL_SPEED_RAD_S, MANUAL_ACCEL_RAD_S2)
                except Exception as e:
                    print(f"Real Robot Error: {e}")
                    return False

            # if self.gripper:
            #     try:
            #         # UPDATED: wsg_driver.move() logic:
            #         # >0.5 -> driver.reference() (OPEN)
            #         # <=0.5 -> driver.close() (CLOSE)
            #         cmd_val = 1.0 if target_gripper > 0.5 else 0.0
                    
            #         # Only send command if it changes state, but use threading to not block
            #         # The driver handles repeated commands gracefully, but we check here too.
            #         if abs(cmd_val - self.gripper_last_cmd) > 0.1:
            #             # Launch in separate thread to avoid blocking main loop
            #             # Passing cmd_val (1.0 or 0.0) directly to move()
            #             t = threading.Thread(target=self.gripper.move, args=(cmd_val,), daemon=True)
            #             t.start()
                        
            #             self.gripper_last_cmd = cmd_val
            #     except Exception as e: print(f"Gripper Error: {e}")

            if self.gripper:
                try:
                    # --- FIXED: Hysteresis (State-Based) Logic ---
                    # Range: ~23mm (Closed) to ~50.3mm (Open)
                    # Use two thresholds to define "Context".
                    # 48mm coming from 50mm -> Should Close.
                    # 48mm coming from 23mm -> Should likely stay Closed (maintain grasp) until fully open.

                    # Thresholds (Tune these based on your object size/noise)
                    # If Open, wait until < 49.0 to Close (High sensitivity to start grasping)
                    TO_CLOSE_THRESH = 45 
                    # If Closed, wait until > 49.5 to Open (Ensure we really want to release)
                    TO_OPEN_THRESH = 45  

                    current_width = float(target_gripper)
                    is_currently_open = (self.gripper_last_cmd > 0.5)

                    if is_currently_open:
                        # Context: Gripper is Open.
                        # Action: Switch to CLOSE only if we drop below the close threshold.
                        # Example: 50.3 -> 48.0 (< 49.0) -> Triggers Close.
                        if current_width < TO_CLOSE_THRESH:
                            cmd_val = 0.0 # Close
                        else:
                            cmd_val = 1.0 # Maintain Open
                    else:
                        # Context: Gripper is Closed (holding object).
                        # Action: Switch to OPEN only if we rise above the open threshold.
                        # Example: 40.0 -> 48.0 (Not > 49.5) -> Maintains Grasp.
                        if current_width > TO_OPEN_THRESH:
                            cmd_val = 1.0 # Open
                        else:
                            cmd_val = 0.0 # Maintain Close

                    # Execute if state changed
                    if abs(cmd_val - self.gripper_last_cmd) > 0.1:
                        t = threading.Thread(target=self.gripper.move, args=(cmd_val,), daemon=True)
                        t.start()
                        self.gripper_last_cmd = cmd_val
                    # ---------------------------------------------

                except Exception as e: print(f"Gripper Error: {e}")
                
            for i, idx in enumerate(self.joint_indices):
                p.resetJointState(self.sim_robot_id, idx, target_joints[i])
            p.stepSimulation()
            
            return True

# -----------------------------------------------------------------------------
# 4. Main Loop
# -----------------------------------------------------------------------------

# def main(args):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # Define Normalizer
#     normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

#     # --- 0. Load Stats and Policy ---
#     checkpoint_dir = os.path.dirname(args.checkpoint)
#     stats_path = os.path.join(checkpoint_dir, 'dataset_stats.pkl')

#     if not os.path.exists(stats_path):
#         # Fallback to parent directory if checkpoint is in a subfolder
#         stats_path = os.path.join(os.path.dirname(checkpoint_dir), 'dataset_stats.pkl')

#     try:
#         with open(stats_path, 'rb') as f:
#             stats = pickle.load(f)
        
#         # Updated to match your flat dictionary structure
#         QPOS_MEAN = stats['qpos_mean']
#         QPOS_STD  = stats['qpos_std']
#         ACTION_MEAN = stats['action_mean']
#         ACTION_STD  = stats['action_std']
        
#         print(f"✅ Successfully loaded stats from: {stats_path}")
#         print(f"   Shape: Qpos Mean {QPOS_MEAN.shape} | Action Mean {ACTION_MEAN.shape}")
#     except Exception as e:
#         print(f"❌ Failed to load dataset_stats.pkl: {e}")
#         sys.exit(1)

#     def pre_process(s_qpos):
#         # Standard normalization: (x - mean) / std
#         return (s_qpos - QPOS_MEAN) / QPOS_STD

#     def post_process(a):
#         # De-normalization: (x * std) + mean
#         return a * ACTION_STD + ACTION_MEAN

#     policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
    
#     # Load checkpoint with weights_only=False to handle numpy scalars (epoch/loss) without error
#     print(f"Loading checkpoint: {args.checkpoint}")
#     payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
#     # Handle both "Full State" (dict with 'model_state_dict') and "Weights Only" checkpoints
#     if isinstance(payload, dict) and 'model_state_dict' in payload:
#         print(f"   [Checkpoint] Loading 'model_state_dict' from full checkpoint (Epoch: {payload.get('epoch', 'N/A')})")
#         state_dict = payload['model_state_dict']
#     else:
#         print(f"   [Checkpoint] Loading weights directly")
#         state_dict = payload
        
#     policy.load_state_dict(state_dict)
#     policy.to(device)
#     policy.eval()
    
#     # Debug: Check camera names order
#     required_cams = POLICY_CONFIG['camera_names']
#     print(f"\n[Config] Model Trained With Cameras: {required_cams}")
    
#     # 1. PHASE 1: Initialize Cameras (Robot Disconnected)
#     print("\n" + "="*60)
#     print("PHASE 1: SENSOR INITIALIZATION")
#     print("⚠️  Ensure Robot Ethernet is DISCONNECTED to avoid Aria conflicts.")
#     print("="*60 + "\n")
    
#     active_sensors = {}
    
#     try:
#         # Initialize only the cameras required by the config
#         # 'aria' uses SharedMemory. 'cam1' and 'cam2' use RealSense via USB.
        
#         if 'aria' in required_cams:
#             print("[Sensor] Starting Aria (Shared Memory)...")
#             active_sensors['aria'] = ThreadedSensor(SharedMemoryReceiver, shm_name="aria_stream_v1")
#             active_sensors['aria'].start()
        
#         if 'cam1_rgb' in required_cams:
#             print(f"[Sensor] Starting Cam 1 (Serial: {CAM_SERIALS[0]})...")
#             active_sensors['cam1_rgb'] = ThreadedSensor(RealSenseCamera, serial_number=CAM_SERIALS[0])
#             active_sensors['cam1_rgb'].start()
            
#         if 'cam2_rgb' in required_cams:
#             print(f"[Sensor] Starting Cam 2 (Serial: {CAM_SERIALS[1]})...")
#             active_sensors['cam2_rgb'] = ThreadedSensor(RealSenseCamera, serial_number=CAM_SERIALS[1])
#             active_sensors['cam2_rgb'].start()

#     except Exception as e:
#         print(f"❌ Camera Init Failed: {e}")
#         return

#     # Wait for streams
#     print("Waiting for video streams...")
#     while True:
#         all_ready = True
#         for name, sensor in active_sensors.items():
#             _, t_stamp,_ = sensor.get_latest()
#             if t_stamp == 0:
#                 all_ready = False
#                 break
        
#         if all_ready:
#             print(f"✅ All {len(active_sensors)} Cameras Ready.")
#             break
#         time.sleep(0.5)

#     # 2. PHASE 2: Connect Robot
#     print("\n" + "="*60)
#     print("PHASE 2: ROBOT CONNECTION")
#     print("👉 ACTION REQUIRED: Connect the Ethernet cable to the UR3 robot now.")
#     print("="*60 + "\n")
    
#     input("Press [Enter] ONLY after you have connected the robot cable...")

#     # 3. PHASE 3: Initialize Robot & Go Home
#     print("\n[System] Connecting to Robot & Gripper...")
#     env = RobotSystem(args.robot_ip, args.gripper_port)
    
#     # --- GO HOME ---
#     env.go_home()
#     # ---------------

#     print(f"\n✅ SYSTEM READY. Mode: {args.mode.upper()}")
    
#     # --- LOGIC: EXECUTE FIRST N ACTIONS ---
#     num_queries = POLICY_CONFIG['num_queries']
    
#     if args.action_chunking:
#         query_freq = 1
#     else:
#         # If execution steps is specified, use it. Otherwise execute all queries.
#         if args.steps_per_inference:
#             query_freq = min(args.steps_per_inference, num_queries)
#         else:
#             query_freq = num_queries
            
#     print(f"[Strategy] Chunking: {args.action_chunking}, Frequency: {query_freq}")
    
#     all_time_actions = torch.full(
#         [args.max_timesteps, args.max_timesteps + num_queries,TASK_CONFIG['state_dim']],
#         float('nan')
#     ).to(device)
    
#     # Initialize all_actions with None to track first run
#     all_actions = None
    
#     t = 0
#     try:
#         while t < args.max_timesteps:
#             loop_start = time.time()
            
#             # --- 1. Capture Data ---
#             qpos_raw = env.get_qpos_real()
#             qpos_norm = pre_process(qpos_raw)
#             qpos_tensor = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)
            
#             # Collect images dynamically based on config
#             latest_images = {}
#             latest_fps = {} # Store FPS

#             for cam_name, sensor in active_sensors.items():
#                 # UNPACK 3 VALUES NOW: img, timestamp, fps
#                 img_raw, ts, fps = sensor.get_latest()
#                 latest_images[cam_name] = img_raw
#                 latest_fps[cam_name] = fps
                
#                 # Check latency (Lag > 0.5s)
#                 if (time.time() - ts > 0.5) and ts > 0:
#                      # Make the warning visible in console
#                      print(f"⚠️ LAG WARNING: {cam_name} is {time.time() - ts:.2f}s behind!")

#             # --- 2. Visualization & Processing ---
#             tensors_to_stack = []
            
#             for cam_name in required_cams:
#                 if cam_name not in latest_images or latest_images[cam_name] is None:
#                     print(f"CRITICAL: Missing frame for {cam_name}")
#                     continue

#                 img_vis = latest_images[cam_name]
                
#                 # Visualize with FPS Overlay
#                 if args.visualize:
#                     bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
                    
#                     # --- DRAW FPS ---
#                     fps_val = latest_fps.get(cam_name, 0.0)
                    
#                     # Color Logic: Green = Good (>20), Red = Bad (<10)
#                     color = (0, 255, 0) if fps_val > 20 else (0, 0, 255)
                    
#                     text = f"FPS: {fps_val:.1f}"
#                     cv2.putText(bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
#                                 1.0, color, 2, cv2.LINE_AA)
#                     # ----------------
                    
#                     cv2.imshow(f"Input: {cam_name}", bgr)

#                 # Preprocess... (rest of your code remains the same)
#                 img_t = torch.from_numpy(img_vis).permute(2, 0, 1).float() / 255.0
#                 img_t = normalize(img_t)
#                 tensors_to_stack.append(img_t.to(device))
            
#             if args.visualize:
#                 cv2.waitKey(1)

#             # --- 3. User Verification (Pre-Inference) ---
#             if args.mode == 'manual':
#                 print("-" * 40)
#                 print(f"Current Joints:  {np.round(qpos_raw[:6], 3)}")
#                 print(f"Current Gripper: {qpos_raw[6]}")
#                 print(">>> Press 'y' in the IMAGE WINDOW to confirm, 'n' to skip.")
#                 print("-" * 40)

#                 user_approved = False
#                 while True:
#                     key = cv2.waitKey(30) & 0xFF
#                     if key == ord('y'):
#                         user_approved = True
#                         break
#                     elif key == ord('k'): # 'k' for skip
#                         user_approved = False
#                         break
                    
#                     # Check if windows closed
#                     if len(required_cams) > 0:
#                         first_cam = required_cams[0]
#                         if cv2.getWindowProperty(f"Input: {first_cam}", cv2.WND_PROP_VISIBLE) < 1:
#                             user_approved = False
#                             break

#                 if not user_approved:
#                     print("Skipping inference step...")
#                     continue

#             # --- 4. Inference ---
#             # Stack [Num_Cams, C, H, W] -> [1, Num_Cams, C, H, W]
#             img_all = torch.stack(tensors_to_stack, dim=0).unsqueeze(0)
#             # img_all_black = torch.zeros_like(img_all)

#             # Logic: If chunking is ON, run every step.
#             # If chunking is OFF, run only when t % query_freq == 0 (start of new N-step block).
#             if args.action_chunking or (t % query_freq == 0):
#                 with torch.inference_mode():
#                     all_actions, attn_weights = policy(qpos_tensor, img_all)

#             # --- 5. Action Selection ---
#             if args.action_chunking:
#                 # Fill temporal buffer with new prediction
#                 all_time_actions[[t], t : t + num_queries] = all_actions
                
#                 # Retrieve all overlapping predictions for the current timestep t
#                 actions_for_curr_step = all_time_actions[:, t]
#                 actions_valid = actions_for_curr_step[~torch.isnan(actions_for_curr_step[:, 0])]
                
#                 if len(actions_valid) > ACTIVE_HORIZON:
#                     actions_valid = actions_valid[-ACTIVE_HORIZON:]
                
#                 # Temporal Ensembling Weighting
#                 k = 0.01 
#                 # Reverse indices so the newest prediction (last index) gets the highest weight
#                 # np.arange(len)[::-1] -> [N-1, N-2, ..., 0]
#                 indices = np.arange(len(actions_valid))[::-1]
#                 exp_weights = np.exp(-k * indices)
#                 exp_weights = exp_weights / exp_weights.sum()
#                 exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                
#                 # Weighted average of valid actions
#                 raw_action = (actions_valid * exp_weights).sum(dim=0).cpu().numpy()
#             else:
#                 # Standard Open Loop Execution
#                 raw_action = all_actions[:, t % query_freq].squeeze(0).cpu().numpy()
                
#             # --- 6. Execution ---
#             action = post_process(raw_action)
#             success = env.execute_action(action, mode=args.mode)
            
#             if not success:
#                 print("Halting.")
#                 break

#             t += 1
#             if args.mode == 'auto':
#                 elapsed = time.time() - loop_start
#                 time.sleep(max(0, SERVO_DT - elapsed))
#     except KeyboardInterrupt:
#         print("Stopping...")
#         if env.rtde_c: env.rtde_c.stopScript()
#     finally:
#         # Stop all sensors
#         for name, sensor in active_sensors.items():
#             print(f"Stopping {name}...")
#             sensor.stop()
            
#         p.disconnect()
#         cv2.destroyAllWindows() 
#         if env.gripper: env.gripper.close_connection()
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define Normalizer
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # --- 0. Load Stats and Policy ---
    checkpoint_dir = os.path.dirname(args.checkpoint)
    stats_path = os.path.join(checkpoint_dir, 'dataset_stats.pkl')

    if not os.path.exists(stats_path):
        # Fallback to parent directory if checkpoint is in a subfolder
        stats_path = os.path.join(os.path.dirname(checkpoint_dir), 'dataset_stats.pkl')

    try:
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        
        QPOS_MEAN = stats['qpos_mean']
        QPOS_STD  = stats['qpos_std']
        ACTION_MEAN = stats['action_mean']
        ACTION_STD  = stats['action_std']
        
        print(f"✅ Successfully loaded stats from: {stats_path}")
        print(f"   Shape: Qpos Mean {QPOS_MEAN.shape} | Action Mean {ACTION_MEAN.shape}")
        print(f"   Shape: Qpos Mean {QPOS_MEAN} | Action Mean {ACTION_MEAN}")
    except Exception as e:
        print(f"❌ Failed to load dataset_stats.pkl: {e}")
        sys.exit(1)

    def pre_process(s_qpos):
        return (s_qpos - QPOS_MEAN) / QPOS_STD

    def post_process(a):
        return a * ACTION_STD + ACTION_MEAN

    policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if isinstance(payload, dict) and 'model_state_dict' in payload:
        print(f"   [Checkpoint] Loading 'model_state_dict' from full checkpoint (Epoch: {payload.get('epoch', 'N/A')})")
        state_dict = payload['model_state_dict']
    else:
        print(f"   [Checkpoint] Loading weights directly")
        state_dict = payload
        
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    
    required_cams = POLICY_CONFIG['camera_names']
    print(f"\n[Config] Model Trained With Cameras: {required_cams}")
    
    # 1. PHASE 1: Initialize Cameras
    print("\n" + "="*60)
    print("PHASE 1: SENSOR INITIALIZATION")
    print("⚠️  Ensure Robot Ethernet is DISCONNECTED to avoid Aria conflicts.")
    print("="*60 + "\n")
    
    active_sensors = {}
    
    try:
        if 'aria' in required_cams:
            print("[Sensor] Starting Aria (Shared Memory)...")
            active_sensors['aria'] = ThreadedSensor(SharedMemoryReceiver, shm_name="aria_stream_v1")
            active_sensors['aria'].start()
        
        if 'cam1_rgb' in required_cams:
            print(f"[Sensor] Starting Cam 1 (Serial: {CAM_SERIALS[0]})...")
            active_sensors['cam1_rgb'] = ThreadedSensor(RealSenseCamera, serial_number=CAM_SERIALS[0])
            active_sensors['cam1_rgb'].start()
            
        if 'cam2_rgb' in required_cams:
            print(f"[Sensor] Starting Cam 2 (Serial: {CAM_SERIALS[1]})...")
            active_sensors['cam2_rgb'] = ThreadedSensor(RealSenseCamera, serial_number=CAM_SERIALS[1])
            active_sensors['cam2_rgb'].start()

    except Exception as e:
        print(f"❌ Camera Init Failed: {e}")
        return

    print("Waiting for video streams...")
    while True:
        all_ready = True
        for name, sensor in active_sensors.items():
            _, t_stamp,_ = sensor.get_latest()
            if t_stamp == 0:
                all_ready = False
                break
        
        if all_ready:
            print(f"✅ All {len(active_sensors)} Cameras Ready.")
            break
        time.sleep(0.5)

    # 2. PHASE 2: Connect Robot
    print("\n" + "="*60)
    print("PHASE 2: ROBOT CONNECTION")
    print("👉 ACTION REQUIRED: Connect the Ethernet cable to the UR3 robot now.")
    print("="*60 + "\n")
    
    input("Press [Enter] ONLY after you have connected the robot cable...")

    # 3. PHASE 3: Initialize Robot & Go Home
    print("\n[System] Connecting to Robot & Gripper...")
    env = RobotSystem(args.robot_ip, args.gripper_port)
    
    env.go_home()

    print(f"\n✅ SYSTEM READY. Mode: {args.mode.upper()}")
    
    num_queries = POLICY_CONFIG['num_queries']
    
    if args.action_chunking:
        query_freq = 1
    else:
        if args.steps_per_inference:
            query_freq = min(args.steps_per_inference, num_queries)
        else:
            query_freq = num_queries
            
    print(f"[Strategy] Chunking: {args.action_chunking}, Frequency: {query_freq}")
    
    all_time_actions = torch.full(
        [args.max_timesteps, args.max_timesteps + num_queries, TASK_CONFIG['state_dim']],
        float('nan')
    ).to(device)
    
    all_actions = None
    attn_weights = None # Initialize attention weights variable
    t = 0
    
    try:
        while t < args.max_timesteps:
            loop_start = time.time()
            
            # --- 1. Capture Data ---
            qpos_raw = env.get_qpos_real()
            qpos_norm = pre_process(qpos_raw)
            qpos_tensor = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)
            
            latest_images = {}
            latest_fps = {} 

            for cam_name, sensor in active_sensors.items():
                img_raw, ts, fps = sensor.get_latest()
                latest_images[cam_name] = img_raw
                latest_fps[cam_name] = fps
                
                if (time.time() - ts > 0.5) and ts > 0:
                     print(f"⚠️ LAG WARNING: {cam_name} is {time.time() - ts:.2f}s behind!")

            # --- 2. Processing (To Tensors) ---
            tensors_to_stack = []
            for cam_name in required_cams:
                if cam_name not in latest_images or latest_images[cam_name] is None:
                    print(f"CRITICAL: Missing frame for {cam_name}")
                    continue

                img_vis = latest_images[cam_name]
                img_t = torch.from_numpy(img_vis).permute(2, 0, 1).float() / 255.0
                # img_t = normalize(img_t)
                tensors_to_stack.append(img_t.to(device))
            
            # Stack [Num_Cams, C, H, W] -> [1, Num_Cams, C, H, W]
            img_all = torch.stack(tensors_to_stack, dim=0).unsqueeze(0)

            # --- 3. Inference ---
            if args.action_chunking or (t % query_freq == 0):
                with torch.inference_mode():
                    # Get actions AND attention weights
                    all_actions, attn_weights = policy(qpos_tensor, img_all)

            # --- 4. VISUALIZATION (Overlaid Attention Map) ---
            if args.visualize and attn_weights is not None:
                # Extract attention for the current prediction query (index 0)
                # Ignore the first 2 tokens (latent, proprio)
                attn = attn_weights[0, 0, 2:].detach().cpu().numpy()
                
                # ResNet34 with 480x640 input -> 32x downsampling = 15x20
                # h, w = 30, 40
                h,w = 15,20
                num_cams = len(required_cams)
                
                # 【修复核心】：先还原为左右拼接的完整 2D 特征图 (15, 20 * num_cams)
                attn_2d_full = attn.reshape(h, w * num_cams)
                
                for cam_id, cam_name in enumerate(required_cams):
                    if cam_name in latest_images:
                        # Convert to BGR for OpenCV display
                        img_bgr = cv2.cvtColor(latest_images[cam_name], cv2.COLOR_RGB2BGR)
                        
                        # Add FPS text
                        fps_val = latest_fps.get(cam_name, 0.0)
                        color = (0, 255, 0) if fps_val > 20 else (0, 0, 255)
                        cv2.putText(img_bgr, f"FPS: {fps_val:.1f}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                        
                        # 【修复核心】：在宽度维度上切片，提取当前相机的特征
                        start_w = cam_id * w
                        end_w = (cam_id + 1) * w
                        cam_attn = attn_2d_full[:, start_w:end_w]
                        
                        # Normalize to 0-1 for heatmap rendering
                        cam_attn = (cam_attn - cam_attn.min()) / (cam_attn.max() - cam_attn.min() + 1e-8)
                        
                        # Resize to match camera dimensions (640x480)
                        cam_attn_resized = cv2.resize(cam_attn, (CAM_WIDTH, CAM_HEIGHT))
                        heatmap = cv2.applyColorMap(np.uint8(255 * cam_attn_resized), cv2.COLORMAP_JET)
                        
                        # Blend the original image and heatmap
                        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
                        cv2.imshow(f"Input & Attention: {cam_name}", overlay)
                cv2.waitKey(1)

            # --- 5. User Verification (Pre-Execution Loop) ---
            if args.mode == 'manual':
                print("-" * 40)
                print(f"Current Joints:  {np.round(qpos_raw[:6], 3)}")
                print(f"Current Gripper: {qpos_raw[6]}")
                print(">>> Press 'y' in the IMAGE WINDOW to confirm, 'k' to skip.")
                print("-" * 40)

                user_approved = False
                while True:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('y'):
                        user_approved = True
                        break
                    elif key == ord('k'): 
                        user_approved = False
                        break
                    
                    if len(required_cams) > 0:
                        first_cam = required_cams[0]
                        # Verify the window hasn't been closed manually
                        if cv2.getWindowProperty(f"Input & Attention: {first_cam}", cv2.WND_PROP_VISIBLE) < 1:
                            user_approved = False
                            break

                if not user_approved:
                    print("Skipping inference step...")
                    continue

            # --- 6. Action Selection ---
            if args.action_chunking:
                all_time_actions[[t], t : t + num_queries] = all_actions
                
                actions_for_curr_step = all_time_actions[:, t]
                actions_valid = actions_for_curr_step[~torch.isnan(actions_for_curr_step[:, 0])]
                
                if len(actions_valid) > ACTIVE_HORIZON:
                    actions_valid = actions_valid[-ACTIVE_HORIZON:]
                
                k = 0.01 
                indices = np.arange(len(actions_valid))[::-1]
                exp_weights = np.exp(-k * indices)
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                
                raw_action = (actions_valid * exp_weights).sum(dim=0).cpu().numpy()
            else:
                raw_action = all_actions[:, t % query_freq].squeeze(0).cpu().numpy()
                
            # --- 7. Execution ---
            action = post_process(raw_action)
            success = env.execute_action(action, mode=args.mode)
            
            if not success:
                print("Halting.")
                break

            t += 1
            if args.mode == 'auto':
                elapsed = time.time() - loop_start
                time.sleep(max(0, SERVO_DT - elapsed))

    except KeyboardInterrupt:
        print("Stopping...")
        if env.rtde_c: env.rtde_c.stopScript()
    finally:
        for name, sensor in active_sensors.items():
            print(f"Stopping {name}...")
            sensor.stop()
            
        p.disconnect()
        cv2.destroyAllWindows() 
        if env.gripper: env.gripper.close_connection()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/home/pengtao/ws_ros2humble-main_lab/ACT/checkpoints/joint/20260215_pick_and_place_66_realsense/pick/policy_step_66672.ckpt', help='Path to ACT policy.ckpt')
    parser.add_argument('--robot_ip', type=str, default=ROBOT_IP)
    parser.add_argument('--gripper_port', type=str, default=GRIPPER_PORT)
    parser.add_argument('--mode', type=str, default='auto', choices=['manual', 'auto'])
    
    # CHUNKING ARGUMENTS
    parser.add_argument('--action_chunking', action='store_true', default=True, help="Enable action chunking (temporal ensembling)")
    parser.add_argument('--steps_per_inference', type=int, default=1, help="If NO chunking, how many steps to execute before re-predicting. Default: Execute all.")
    
    parser.add_argument('--max_timesteps', type=int, default=10000)
    parser.add_argument('--visualize', action='store_true', help="Visualize active camera streams", default=True)
    
    args = parser.parse_args()
    main(args)
