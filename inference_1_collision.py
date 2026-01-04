import torch
import numpy as np
import argparse
import sys
import os
import time
import cv2
import traceback
from PIL import Image
import math
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as transforms

# --- Physics & Planning Imports ---
import mplib
import pybullet as p
import pybullet_data

# --- Aria SDK Imports ---
import aria.sdk as aria
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord

# --- Project Imports ---
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config_1 import POLICY_CONFIG, TASK_CONFIG
from training.utils_1 import make_policy, get_norm_stats

# --- Robot Interface Imports ---
try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except ImportError:
    print("⚠️ RTDE modules not found. Running in simulation/dummy mode.")

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
ROBOT_IP = "192.168.1.102"  # <--- UPDATE THIS IP

# --- Environment Constants ---
INIT_JOINTS_DEG = [45.0, -52.39, -100, -90, -266.31, 0]
SIM_ROBOT_WORLD_ROT_Z = np.pi 

# Obstacles
WALL_POS_WORLD = [-0.14, -0.14, 0.5]  
WALL_ROT_Z_DEG = 45  
WALL_SIZE = [0.02, 1.0, 1.0] 
TABLE_SIZE = [1.0, 1.0, 0.1]
TABLE_POS_WORLD = [0, 0, -TABLE_SIZE[2]/2-0.01]
TABLE_ROT_Z_DEG = 0

# --- ACT Coordinate Transforms ---
R_TCP_DEVICE_RAW = np.array([
    [ 0.99920421, -0.03675654,  0.01548867],
    [ 0.03020608,  0.95091268,  0.30798161],
    [-0.02604871, -0.30726867,  0.95126622]
])
R_CORRECTION = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0]
])
R_TCP_DEVICE_MAT = R_TCP_DEVICE_RAW @ R_CORRECTION
t_TCP_DEVICE_VEC = np.array([0.01260316, -0.09558874, 0.06506849])

T_TCP_DEVICE = np.eye(4)
T_TCP_DEVICE[:3, :3] = R_TCP_DEVICE_MAT
T_TCP_DEVICE[:3, 3] = t_TCP_DEVICE_VEC
T_DEVICE_TCP = np.linalg.inv(T_TCP_DEVICE)

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def get_object_transforms(pos_world_arr, rot_z_deg):
    pos_world = np.array(pos_world_arr)
    r_obj_local = R.from_euler('z', rot_z_deg, degrees=True)
    r_world_robot = R.from_euler('z', -SIM_ROBOT_WORLD_ROT_Z, degrees=False)
    pos_robot = r_world_robot.apply(pos_world)
    quat_robot = (r_world_robot * r_obj_local).as_quat()
    quat_world = r_obj_local.as_quat()
    return pos_robot, quat_robot, pos_world, quat_world

def get_box_point_cloud(center, size, quat, resolution=0.04):
    x = np.arange(-size[0]/2, size[0]/2, resolution)
    y = np.arange(-size[1]/2, size[1]/2, resolution)
    z = np.arange(-size[2]/2, size[2]/2, resolution)
    if len(x)==0: x=[0]
    if len(y)==0: y=[0]
    if len(z)==0: z=[0]
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    points_rotated = R.from_quat(quat).apply(points)
    return points_rotated + center

def draw_coordinate_frame(pos, quat, axis_len=0.1, life_time=0):
    rot = R.from_quat(quat).as_matrix()
    p.addUserDebugLine(pos, pos + rot[:,0]*axis_len, [1,0,0], lineWidth=2, lifeTime=life_time) 
    p.addUserDebugLine(pos, pos + rot[:,1]*axis_len, [0,1,0], lineWidth=2, lifeTime=life_time) 
    p.addUserDebugLine(pos, pos + rot[:,2]*axis_len, [0,0,1], lineWidth=2, lifeTime=life_time) 

def rot6d_to_matrix(rot_6d):
    x_raw = rot_6d[..., 0:3]
    y_raw = rot_6d[..., 3:6]
    x = torch.nn.functional.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = torch.nn.functional.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)

def mat_to_rot6d_numpy(mat):
    return mat[:, :2].flatten(order='F')

def rotvec_to_matrix(rx, ry, rz):
    return R.from_rotvec([rx, ry, rz]).as_matrix()

def matrix_to_rotvec(mat):
    return R.from_matrix(mat).as_rotvec()

# -----------------------------------------------------------------------------
# 3. Aria Camera Helper
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
        sensors_calib_json = self.streaming_manager.sensors_calibration()
        sensors_calib = device_calibration_from_json_string(sensors_calib_json)
        self.rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
        target_height, target_width = 1408 ,1408
        hfov_deg = 110
        focal_length = (target_width / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.dst_calib = get_linear_camera_calibration(target_width, target_height, focal_length, "camera-rgb")
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
# 4. Main Inference Class
# -----------------------------------------------------------------------------
class ManualInferenceACT:
    def __init__(self, checkpoint_path, device=None, aria_args=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.checkpoint_path = checkpoint_path
        
        # Load Config & Stats
        self.policy_config = POLICY_CONFIG
        self.task_config = TASK_CONFIG
        self._load_stats()
        self._load_model()
        self.transform = self._get_transform()
        
        # Robot Connection
        self.rtde_c = None
        self.rtde_r = None
        try:
            print(f"\n🔗 Connecting to robot at {ROBOT_IP}...")
            self.rtde_c = RTDEControlInterface(ROBOT_IP)
            self.rtde_r = RTDEReceiveInterface(ROBOT_IP)
            print("✅ Robot connected successfully!\n")
        except Exception as e:
            print(f"⚠️ Warning: Could not connect to robot: {e}")

        # Setup Physics (Sim Environment)
        self._setup_physics_and_collision()

        # Setup Camera
        self.camera = AriaLiveStreamer(
            interface=aria_args.streaming_interface,
            device_ip=aria_args.device_ip,
            profile_name=aria_args.profile_name
        )

        # Reference Frame
        self.T_base_device0 = None 

    def _setup_physics_and_collision(self):
        """Replicates test_teleoperation.py"""
        print("🛡️ Initializing Physics and Collision Checkers...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(current_dir, "ur3_gripper.urdf") 
        self.srdf_path = os.path.join(current_dir, "ur3_gripper.srdf")
        
        p.connect(p.GUI) 
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.resetDebugVisualizerCamera(1.2, 90, -30, [0, 0, 0])
        p.loadURDF("plane.urdf")    

        # Obstacles
        wall_pos_r, wall_quat_r, wall_pos_w, wall_quat_w = get_object_transforms(WALL_POS_WORLD, WALL_ROT_Z_DEG)
        table_pos_r, table_quat_r, table_pos_w, table_quat_w = get_object_transforms(TABLE_POS_WORLD, TABLE_ROT_Z_DEG)

        p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE]), 
                          basePosition=wall_pos_w, baseOrientation=wall_quat_w,
                          baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE], rgbaColor=[0.8, 0.2, 0.2, 0.8]))
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE]), 
                          basePosition=table_pos_w, baseOrientation=table_quat_w,
                          baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE], rgbaColor=[0.5, 0.3, 0.1, 1.0]))

        # Robot
        sim_robot_orn = p.getQuaternionFromEuler([0, 0, SIM_ROBOT_WORLD_ROT_Z])
        self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0], baseOrientation=sim_robot_orn, useFixedBase=1)
        
        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        name_to_idx = {p.getJointInfo(self.robot_id, i)[1].decode('utf-8'): i for i in range(p.getNumJoints(self.robot_id))}
        self.active_indices = [name_to_idx.get(n, -1) for n in joint_names]

        # End Effector
        self.ee_link_idx = -1
        for i in range(p.getNumJoints(self.robot_id)):
            if p.getJointInfo(self.robot_id, i)[12].decode("utf-8") == "tool0": 
                self.ee_link_idx = i; break
        if self.ee_link_idx == -1: raise ValueError("Could not find link 'tool0'")

        # MPLib
        self.planner = mplib.Planner(urdf=self.urdf_path, srdf=self.srdf_path, move_group="tool0")
        combined_pts = np.vstack([
            get_box_point_cloud(wall_pos_r, WALL_SIZE, wall_quat_r, resolution=0.04), 
            get_box_point_cloud(table_pos_r, TABLE_SIZE, table_quat_r, resolution=0.04)
        ])
        self.planner.update_point_cloud(combined_pts)
        print("✅ Environment setup complete.")

    def _load_stats(self):
        dataset_dir = self.task_config['dataset_dir']
        stats = get_norm_stats(dataset_dir)
        self.tx_mean = torch.tensor([stats['tx']['mean']], device=self.device, dtype=torch.float32)
        self.tx_std  = torch.tensor([stats['tx']['std']],  device=self.device, dtype=torch.float32)
        self.ty_mean = torch.tensor([stats['ty']['mean']], device=self.device, dtype=torch.float32)
        self.ty_std  = torch.tensor([stats['ty']['std']],  device=self.device, dtype=torch.float32)
        self.tz_mean = torch.tensor([stats['tz']['mean']], device=self.device, dtype=torch.float32)
        self.tz_std  = torch.tensor([stats['tz']['std']],  device=self.device, dtype=torch.float32)
        
        self.qpos_mean = torch.zeros(10, device=self.device, dtype=torch.float32)
        self.qpos_std  = torch.ones(10,  device=self.device, dtype=torch.float32)
        self.qpos_mean[0] = self.tx_mean
        self.qpos_mean[1] = self.ty_mean
        self.qpos_mean[2] = self.tz_mean
        self.qpos_std[0]  = self.tx_std
        self.qpos_std[1]  = self.ty_std
        self.qpos_std[2]  = self.tz_std

    def _load_model(self):
        self.policy = make_policy(self.policy_config['policy_class'], self.policy_config)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.policy.to(self.device)
        self.policy.eval()
        print("✅ Model Loaded.")

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _normalize_qpos(self, qpos_raw):
        qpos_tensor = torch.tensor(qpos_raw, dtype=torch.float32, device=self.device).unsqueeze(0) 
        return (qpos_tensor - self.qpos_mean) / self.qpos_std

    def _denormalize_pos_numpy(self, norm_pos_array):
        tx_m, tx_s = self.tx_mean.cpu().numpy(), self.tx_std.cpu().numpy()
        ty_m, ty_s = self.ty_mean.cpu().numpy(), self.ty_std.cpu().numpy()
        tz_m, tz_s = self.tz_mean.cpu().numpy(), self.tz_std.cpu().numpy()
        x = norm_pos_array[..., 0] * tx_s + tx_m
        y = norm_pos_array[..., 1] * ty_s + ty_m
        z = norm_pos_array[..., 2] * tz_s + tz_m
        return np.stack([x, y, z], axis=-1)

    # -------------------------------------------------------------------------
    # COORDINATE TRANSFORMS
    # -------------------------------------------------------------------------
    def compute_T_base_device0(self, current_pose_vec):
        T_base_tcp = np.eye(4)
        T_base_tcp[:3, :3] = rotvec_to_matrix(*current_pose_vec[3:])
        T_base_tcp[:3, 3] = current_pose_vec[:3]
        self.T_base_device0 = T_base_tcp @ T_TCP_DEVICE
        print("✅ Initial Reference Frame Set (T_base_device0).")

    def get_current_pose_relative(self):
        if self.rtde_r is None: return [0,0,0, 1,0,0,0,1,0, 0] 
        pose_rtde = self.rtde_r.getActualTCPPose() 
        T_base_tcp = np.eye(4)
        T_base_tcp[:3, :3] = rotvec_to_matrix(*pose_rtde[3:])
        T_base_tcp[:3, 3] = pose_rtde[:3]
        T_base_device = T_base_tcp @ T_TCP_DEVICE
        T_rel = np.linalg.inv(self.T_base_device0) @ T_base_device
        pos_rel = T_rel[:3, 3].tolist()
        rot6d_rel = mat_to_rot6d_numpy(T_rel[:3, :3]).tolist()
        return pos_rel + rot6d_rel + [1.0]

    def convert_relative_to_absolute(self, rel_pos, rel_rot6d):
        r6d_tensor = torch.tensor(rel_rot6d).unsqueeze(0)
        rot_mat_rel = rot6d_to_matrix(r6d_tensor)[0].numpy()
        T_rel = np.eye(4)
        T_rel[:3, :3] = rot_mat_rel
        T_rel[:3, 3] = rel_pos
        T_base_device_target = self.T_base_device0 @ T_rel
        T_base_tcp_target = T_base_device_target @ T_DEVICE_TCP
        target_pos = T_base_tcp_target[:3, 3]
        target_quat = R.from_matrix(T_base_tcp_target[:3, :3]).as_quat() 
        return target_pos, target_quat

    # -------------------------------------------------------------------------
    # EXECUTION WITH moveJ and CONFIRMATION
    # -------------------------------------------------------------------------
    def execute_absolute_action(self, target_pos, target_quat):
        """
        1. Sync Sim to Real.
        2. PyBullet IK.
        3. Collision Check.
        4. Show Ghost Robot.
        5. ASK USER FOR CONFIRMATION.
        6. moveJ.
        """
        # 1. Visualization
        draw_coordinate_frame(target_pos, target_quat, life_time=5.0) # Show longer for user to see

        if self.rtde_r is None: return

        # 2. Sync Sim (Ghost Robot)
        actual_q = self.rtde_r.getActualQ()
        for i, angle in enumerate(actual_q):
            p.resetJointState(self.robot_id, self.active_indices[i], angle)
        
        # 3. Solve IK (PyBullet)
        ik_solution = p.calculateInverseKinematics(
            self.robot_id, 
            self.ee_link_idx, 
            targetPosition=target_pos, 
            targetOrientation=target_quat,
            maxNumIterations=50,
            residualThreshold=1e-4
        )
        q_proposal = list(ik_solution[:6]) 
        
        # 4. Collision Check (MPLib)
        is_self_collision = self.planner.check_for_self_collision(q_proposal)
        is_env_collision = self.planner.check_for_env_collision(q_proposal)

        if is_self_collision or is_env_collision:
            p.addUserDebugText("COLLISION!", [0,0,1], [1,0,0], lifeTime=1.0, textSize=2)
            print(f"🛑 COLLISION DETECTED! Stopping. (Self: {is_self_collision}, Env: {is_env_collision})")
            return # Skip move

        # 5. Visualize Ghost Robot
        p.setJointMotorControlArray(self.robot_id, self.active_indices, p.POSITION_CONTROL, 
                                    targetPositions=q_proposal, forces=[200.0]*6)
        p.stepSimulation()
        
        # 6. ASK FOR CONFIRMATION
        print(f"\n⚡ Target Calculated: {target_pos}")
        print("   Ghost Robot updated in PyBullet.")
        user_input = input(">> Press ENTER to Execute moveJ, 's' to Skip, 'q' to Quit: ").lower()
        
        if user_input == 'q':
            print("Quitting...")
            sys.exit(0)
        elif user_input == 's':
            print("Skipping step.")
            return

        # 7. Execute on Real Robot (Blocking moveJ)
        try:
            print("   Moving...")
            # Speed 0.5 rad/s, Acc 0.3 rad/s^2 (Adjust as needed)
            self.rtde_c.moveJ(q_proposal, 0.5, 0.3)
            print("   Done.")
        except Exception as e:
            print(f"Error in moveJ: {e}")

    def loop(self):
        print("\n🚀 Starting Inference Loop (Safe Mode - moveJ with Confirmation)...")
        
        # --- Initialization Move ---
        if self.rtde_c:
            print("Moving to initial config...")
            init_rad = np.deg2rad(INIT_JOINTS_DEG).tolist()
            self.rtde_c.moveJ(init_rad)
            time.sleep(1.0)
            
            # Set Reference Frame here
            init_pose_vec = self.rtde_r.getActualTCPPose()
            self.compute_T_base_device0(init_pose_vec)
        else:
            self.T_base_device0 = np.eye(4) 

        step_idx = 0
        while True:
            # 1. Get Inputs (Relative Frame)
            qpos_rel_raw = self.get_current_pose_relative()
            obs_pil = self.camera.get_latest_frame()
            if obs_pil is None: 
                time.sleep(0.1); continue

            # 2. Inference
            obs_tensor = self.transform(obs_pil).unsqueeze(0).unsqueeze(0).to(self.device)
            qpos_norm = self._normalize_qpos(qpos_rel_raw)
            
            with torch.inference_mode():
                action_pred = self.policy(qpos_norm, obs_tensor) 
            
            # 3. Process Chunk & Extract Prediction
            action_chunk = action_pred.squeeze(0).squeeze(0).cpu().numpy()
            pred_pos_rel = self._denormalize_pos_numpy(action_chunk[..., :3])[0]
            pred_rot6d_rel = action_chunk[..., 3:9][0]
            
            # 4. Convert to Absolute
            target_pos_abs, target_quat_abs = self.convert_relative_to_absolute(pred_pos_rel, pred_rot6d_rel)
            
            # 5. Execute (Validation + Confirmation + moveJ)
            self.execute_absolute_action(target_pos_abs, target_quat_abs)
            
            step_idx += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/policy_best.ckpt')
    parser.add_argument("--interface", dest="streaming_interface", type=str, default="usb")
    parser.add_argument("--profile", dest="profile_name", type=str, default="profile28")
    parser.add_argument("--device-ip", help="Aria IP for wifi")
    args = parser.parse_args()
    
    agent = ManualInferenceACT(args.checkpoint, aria_args=args)
    agent.loop()

if __name__ == '__main__':
    main()