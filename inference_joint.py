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
    -- one image from aria glasses(use the distort images, profile28) and one image realsense D435 (the size is 640x480,30fps). both images should be normalized , must be normalized by the transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])) . the input image size is (H*W): 480x640
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
from training.utils_joint import make_policy
from config.config_joint import POLICY_CONFIG

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
    from drivers.wsg_driver import WSGGripperDriver
except ImportError:
    print("⚠️ WARNING: WSG Driver not found.")
    WSGGripperDriver = None

# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------

# --- SAFETY LIMITS ---
MANUAL_SPEED_RAD_S = 0.1    # Safe speed for manual steps
MANUAL_ACCEL_RAD_S2 = 0.05   # Safe accel for manual steps
SERVO_DT = 0.15             # Loop time for Auto mode (higher = slower/smoother)
SERVO_LOOKAHEAD = 0.1
SERVO_GAIN = 300
ACTIVE_HORIZON = 2         # Max past predictions to aggregate

# --- HARDWARE ---
ROBOT_IP = "10.0.0.1" 
GRIPPER_PORT = "/dev/ttyACM0" 

# --- NORMALIZATION ---
NORM_STATS = {
    'qpos_mean': np.array([0.48919878, -1.99151537, -1.83667825, -0.15944986, -4.5422106, -0.22052902], dtype=np.float32),
    'qpos_std':  np.array([0.16403829, 0.34577541, 0.17637353, 0.22534245, 0.12832069, 0.13293354], dtype=np.float32),
}

# --- ENV CONSTANTS (MATCHING run_ur3_spacemouse.py) ---
INIT_JOINTS_DEG = [45, -80.0, -123.0, -24.00, -270.0, 0.0]
SIM_ROBOT_WORLD_ROT_Z = np.pi 

# Obstacles
WALL_POS_WORLD = [-0.14, -0.14, 0.5]  
WALL_ROT_Z_DEG = 45
WALL_SIZE = [0.02, 1.0, 1.0] 

TABLE_SIZE = [1.0, 1.0, 0.1]
TABLE_POS_WORLD = [0, 0, -TABLE_SIZE[2]/2-0.01]
TABLE_ROT_Z_DEG = 0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (MATCHING run_ur3_spacemouse.py)
# -----------------------------------------------------------------------------
def get_object_transforms(pos_world_arr, rot_z_deg):
    pos_world = np.array(pos_world_arr)
    r_obj_local = R.from_euler('z', rot_z_deg, degrees=True)
    r_world_robot = R.from_euler('z', -SIM_ROBOT_WORLD_ROT_Z, degrees=False)
    pos_robot = r_world_robot.apply(pos_world)
    quat_robot = (r_world_robot * r_obj_local).as_quat()
    quat_world = r_obj_local.as_quat()
    return pos_robot, quat_robot, pos_world, quat_world

# -----------------------------------------------------------------------------
# 1. Threaded Wrapper
# -----------------------------------------------------------------------------

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
            except Exception as e:
                print(f"❌ {self.sensor_name} Thread Error: {e}")
                time.sleep(1.0)
            time.sleep(0.001) 

    def get_latest(self):
        with self.lock:
            if self.latest_data is None: return None, 0
            return self.latest_data.copy(), self.timestamp

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
        try:
            self.sensor.stop()
        except: pass

# -----------------------------------------------------------------------------
# 2. Camera Classes
# -----------------------------------------------------------------------------

from multiprocessing import shared_memory
import struct
class SharedMemoryReceiver:
    def __init__(self, shm_name, shape=(480, 640, 3)):
        self.shm_name = shm_name
        self.shape = shape
        self.size = np.prod(shape)
        self.connected = False
        self.shm = None
        self._connect()

    def _connect(self):
        try:
            self.shm = shared_memory.SharedMemory(create=False, name=self.shm_name)
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
                self.gripper = WSGGripperDriver(gripper_port)
                print("[Gripper] Connected!")
            except: print("❌ Gripper Connection Failed")

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
        
        if self.gripper:
            try:
                g_state = np.clip(self.gripper.get_width() / 0.05, 0.0, 1.0)
            except: g_state = 1.0
        else: g_state = 1.0 
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
            
            if mode == 'manual':
                # Visual Feedback
                for j in range(p.getNumJoints(self.ghost_robot_id)):
                    p.changeVisualShape(self.ghost_robot_id, j, rgbaColor=[0, 1, 0, 0.6])
                
                grp_txt = "OPEN" if target_gripper > 0.5 else "CLOSE"
                print(f"\n   [NEXT] Gripper: {grp_txt} | Joints: {np.round(target_joints[:3], 2)}...")
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

            if self.gripper:
                try:
                    cmd_val = 1.0 if target_gripper > 0.5 else 0.0
                    if abs(cmd_val - self.gripper_last_cmd) > 0.1:
                        # --- FIX: THREADED GRIPPER CONTROL ---
                        # Launch in separate thread to avoid blocking main loop
                        t = threading.Thread(target=self.gripper.move, args=(cmd_val,), daemon=True)
                        t.start()
                        # -------------------------------------
                        self.gripper_last_cmd = cmd_val
                except Exception as e: print(f"Gripper Error: {e}")

            for i, idx in enumerate(self.joint_indices):
                p.resetJointState(self.sim_robot_id, idx, target_joints[i])
            p.stepSimulation()
            
            return True

# -----------------------------------------------------------------------------
# 4. Main Loop
# -----------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 0. Load Policy
    if 'qpos_mean' not in NORM_STATS: raise ValueError("Stats missing")

    def pre_process(s_qpos):
        """Normalize only the 6 joints, keep gripper (index 6) raw."""
        # s_qpos shape is (7,). Stats are (6,).
        joints_norm = (s_qpos[:6] - NORM_STATS['qpos_mean']) / NORM_STATS['qpos_std']
        # Concatenate: [normalized_joints, raw_gripper]
        return np.concatenate([joints_norm, s_qpos[6:]])

    def post_process(a):
        """Denormalize only the 6 joints, keep gripper (index 6) raw."""
        # a shape is (7,). Stats are (6,).
        joints_denorm = a[:6] * NORM_STATS['qpos_std'] + NORM_STATS['qpos_mean']
        # Concatenate: [denormalized_joints, raw_gripper]
        return np.concatenate([joints_denorm, a[6:]])


    policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
    
    # Load checkpoint with weights_only=False to handle numpy scalars (epoch/loss) without error
    print(f"Loading checkpoint: {args.checkpoint}")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Handle both "Full State" (dict with 'model_state_dict') and "Weights Only" checkpoints
    if isinstance(payload, dict) and 'model_state_dict' in payload:
        print(f"   [Checkpoint] Loading 'model_state_dict' from full checkpoint (Epoch: {payload.get('epoch', 'N/A')})")
        state_dict = payload['model_state_dict']
    else:
        print(f"   [Checkpoint] Loading weights directly")
        state_dict = payload
        
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    
    # Debug: Check camera names order
    required_cams = POLICY_CONFIG['camera_names']
    print(f"\n[Config] Model Trained With Cameras: {required_cams}")
    
    # 1. PHASE 1: Initialize Cameras (Robot Disconnected)
    print("\n" + "="*60)
    print("PHASE 1: SENSOR INITIALIZATION")
    print("⚠️  Ensure Robot Ethernet is DISCONNECTED to avoid Aria conflicts.")
    print("="*60 + "\n")
    
    aria_thread = None
    rs_thread = None

    try:
        aria_thread = ThreadedSensor(SharedMemoryReceiver, shm_name="aria_stream_v1")
        aria_thread.start()
        rs_thread = ThreadedSensor(SharedMemoryReceiver, shm_name="realsense_stream_v1")
        rs_thread.start()
    except Exception as e:
        print(f"❌ Camera Init Failed: {e}")
        return

    # Wait for streams
    print("Waiting for video streams...")
    while True:
        _, t1 = aria_thread.get_latest()
        _, t2 = rs_thread.get_latest()
        if t1 > 0 and t2 > 0:
            print("✅ Cameras Ready.")
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
    
    # --- GO HOME ---
    env.go_home()
    # ---------------

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    print(f"\n✅ SYSTEM READY. Mode: {args.mode.upper()}")
    
    query_freq = POLICY_CONFIG['num_queries']
    if args.action_chunking: query_freq = 1
    num_queries = POLICY_CONFIG['num_queries']
    
    all_time_actions = torch.full(
        [args.max_timesteps, args.max_timesteps + num_queries, POLICY_CONFIG['action_dim']],
        float('nan')
    ).to(device)
    
    t = 0
    try:
        while t < args.max_timesteps:
            loop_start = time.time()
            
            # --- 1. Capture Data ---
            qpos_raw = env.get_qpos_real()
            qpos_norm = pre_process(qpos_raw)
            qpos_tensor = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)
            
            img_aria_raw, ts_aria = aria_thread.get_latest()
            img_rs_raw, ts_rs = rs_thread.get_latest()
            
            # --- Safety Check ---
            if (time.time() - ts_aria > 0.5) or (time.time() - ts_rs > 0.5):
                print("⚠️ CRITICAL: Camera lag > 500ms! Stopping.")
                break

            # --- 2. Image Preprocessing ---
            # Ensure RealSense is 640x480 (Aria is already resized in get_frame)
            if img_rs_raw is not None and img_rs_raw.shape[:2] != (480, 640):
                img_rs_proc = cv2.resize(img_rs_raw, (640, 480))
            else:
                img_rs_proc = img_rs_raw

            # --- 3. Visualization (Inference Inputs Only) ---
            if args.aria and img_aria_raw is not None:
                vis_aria = cv2.cvtColor(img_aria_raw, cv2.COLOR_RGB2BGR)
                cv2.imshow("Inference Input: Aria", vis_aria)
            
            if args.realsense and img_rs_proc is not None:
                vis_rs = cv2.cvtColor(img_rs_proc, cv2.COLOR_RGB2BGR)
                cv2.imshow("Inference Input: RealSense", vis_rs)

            if args.aria or args.realsense: 
                cv2.waitKey(1)

            # --- 4. User Verification (Pre-Inference) ---
            if args.mode == 'manual':
                print("-" * 40)
                print(f"Current Joints:  {np.round(qpos_raw[:6], 3)}")
                print(f"Current Gripper: {qpos_raw[6]}")
                print(">>> Press 'y' in the IMAGE WINDOW to confirm, 'n' to skip.")
                print("-" * 40)

                # Loop until 'y' or 'n' is pressed in the OpenCV window
                # This prevents the "Force Quit" dialog by keeping the GUI active
                user_approved = False
                while True:
                    # Update window events every 30ms
                    key = cv2.waitKey(30) & 0xFF
                    
                    if key == ord('y'):
                        user_approved = True
                        break
                    elif key == ord('k'):
                        user_approved = False
                        break
                    
                    # Optional: Check if window was closed explicitly
                    if cv2.getWindowProperty("Inference Input: RealSense", cv2.WND_PROP_VISIBLE) < 1:
                        user_approved = False
                        break

                if not user_approved:
                    print("Skipping inference step...")
                    continue  # Skip to next loop iteration

            # --- 5. Prepare Tensors Dynamically ---
            t_aria = tf(Image.fromarray(img_aria_raw)).to(device)
            t_rs = tf(Image.fromarray(img_rs_proc)).to(device)
            t_aria_fake = torch.zeros_like(t_aria).to(device)
            t_rs_fake = torch.zeros_like(t_rs).to(device)
            
            # Map available tensors to names
            # 'aria' and 'realsense' are standard names in utils_joint.py
            camera_map = {
                'aria': t_aria,
                'realsense': t_rs,
            }

            # Construct input list based on Config order
            tensors_to_stack = []
            for cam_name in required_cams:
                if cam_name not in camera_map:
                    print(f"❌ CRITICAL ERROR: Model expects camera '{cam_name}' but it is not mapped in inference loop.")
                    raise ValueError(f"Missing camera mapping for {cam_name}")
                tensors_to_stack.append(camera_map[cam_name])
            
            # Stack [Num_Cams, C, H, W] -> [1, Num_Cams, C, H, W]
            img_all = torch.stack(tensors_to_stack, dim=0).unsqueeze(0)

            # --- 6. Inference ---
            with torch.inference_mode():
                all_actions = policy(qpos_tensor, img_all)

            # --- 7. Action Chunking ---
            if args.action_chunking:
                all_time_actions[[t], t : t + num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_valid = actions_for_curr_step[~torch.isnan(actions_for_curr_step[:, 0])]
                if len(actions_valid) > ACTIVE_HORIZON:
                    actions_valid = actions_valid[-ACTIVE_HORIZON:]
                
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_valid)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                
                raw_action = (actions_valid * exp_weights).sum(dim=0, keepdim=True).squeeze(0).cpu().numpy()
            else:
                raw_action = all_actions[:, t % query_freq].squeeze(0).cpu().numpy()

            # --- 8. Execution (Includes Ghost Visualization) ---
            # NOTE: env.execute_action handles the Ghost Robot display and 
            # final execution confirmation when mode='manual'.
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
        if aria_thread: aria_thread.stop()
        if rs_thread: rs_thread.stop()
        p.disconnect()
        cv2.destroyAllWindows() 
        if env.gripper: env.gripper.close_connection()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/home/pengtao/ws_ros2humble-main_lab/ACT/checkpoints/pick_aria_realsense/pick/policy_epoch_400_seed_42.ckpt', help='Path to ACT policy.ckpt')
    parser.add_argument('--robot_ip', type=str, default=ROBOT_IP)
    parser.add_argument('--gripper_port', type=str, default=GRIPPER_PORT)
    parser.add_argument('--aria_interface', type=str, default="usb")
    parser.add_argument('--mode', type=str, default='auto', choices=['manual', 'auto'])
    parser.add_argument('--action_chunking', action='store_true', default=True, help="Enable action chunking during execution")
    parser.add_argument('--max_timesteps', type=int, default=1000)
    parser.add_argument('--aria', action='store_true', help="Visualize Aria stream", default=True)
    parser.add_argument('--realsense', action='store_true', help="Visualize RealSense stream", default=True)
    
    args = parser.parse_args()
    main(args)