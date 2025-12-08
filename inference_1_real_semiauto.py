import torch
import numpy as np
import argparse
import sys
import os
import time
import cv2
import traceback
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as transforms

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
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

ROBOT_IP = "10.0.0.10"  # Robot IP

# -----------------------------------------------------------------------------
# Math & Transformation Constants
# -----------------------------------------------------------------------------

# Transformation from TCP to Device (Fixed)
# R and t provided in instructions
R_TCP_DEVICE_MAT = np.array([
    [ 0.99920421, -0.03675654,  0.01548867],
    [ 0.03020608,  0.95091268,  0.30798161],
    [-0.02604871, -0.30726867,  0.95126622]
])
t_TCP_DEVICE_VEC = np.array([0.01260316, -0.09558874, 0.06506849])

# T_tcp_device (4x4)
T_TCP_DEVICE = np.eye(4)
T_TCP_DEVICE[:3, :3] = R_TCP_DEVICE_MAT
T_TCP_DEVICE[:3, 3] = t_TCP_DEVICE_VEC

# T_device_tcp (Inverse)
T_DEVICE_TCP = np.linalg.inv(T_TCP_DEVICE)

def rot6d_to_matrix(rot_6d):
    """Convert 6D rotation to 3x3 matrix (Torch)."""
    x_raw = rot_6d[..., 0:3]
    y_raw = rot_6d[..., 3:6]
    x = torch.nn.functional.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = torch.nn.functional.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)

def mat_to_rot6d_numpy(mat):
    """Convert 3x3 matrix to 6D rotation (Numpy). Returns 1D array of size 6."""
    # First two columns flattened
    return mat[:, :2].flatten(order='F')

def rotvec_to_matrix(rx, ry, rz):
    return R.from_rotvec([rx, ry, rz]).as_matrix()

def matrix_to_rotvec(mat):
    return R.from_matrix(mat).as_rotvec()

# -----------------------------------------------------------------------------
# Aria Camera Helper Classes
# -----------------------------------------------------------------------------

class StreamingClientObserver:
    def __init__(self):
        self.rgb_image = None
        self.timestamp_ns = 0

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.rgb_image = image
        self.timestamp_ns = record.capture_timestamp_ns

class AriaLiveStreamer:
    """Wrapper to handle Aria Glasses Streaming and Undistortion"""
    def __init__(self, interface="usb", device_ip=None, profile_name="profile18"):
        print(f"\n👓 Initializing Aria Glasses ({interface})...")
        
        # 1. Setup Client
        aria.set_log_level(aria.Level.Info)
        self.device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        if device_ip:
            client_config.ip_v4_address = device_ip
        self.device_client.set_client_config(client_config)

        # 2. Connect
        try:
            self.device = self.device_client.connect()
        except Exception as e:
            print(f"❌ Failed to connect to Aria Device: {e}")
            raise e

        # 3. Setup Streaming Manager
        self.streaming_manager = self.device.streaming_manager
        self.streaming_client = self.streaming_manager.streaming_client

        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = profile_name
        if interface == "usb":
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
        streaming_config.security_options.use_ephemeral_certs = True
        self.streaming_manager.streaming_config = streaming_config

        # 4. Get Calibration
        print("   Getting sensors calibration...")
        sensors_calib_json = self.streaming_manager.sensors_calibration()
        sensors_calib = device_calibration_from_json_string(sensors_calib_json)
        self.rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
        self.dst_calib = get_linear_camera_calibration(512, 512, 110, "camera-rgb")

        # 5. Start Streaming
        print("   Starting stream...")
        self.streaming_manager.start_streaming()

        # 6. Subscribe
        config = self.streaming_client.subscription_config
        config.subscriber_data_type = aria.StreamingDataType.Rgb
        self.streaming_client.subscription_config = config

        self.observer = StreamingClientObserver()
        self.streaming_client.set_streaming_client_observer(self.observer)
        self.streaming_client.subscribe()
        print("✅ Aria streaming started.")

    def get_latest_frame(self):
        """Returns the latest undistorted RGB PIL Image (rotated upright)"""
        raw_img = self.observer.rgb_image
        
        if raw_img is None:
            return None
            
        rgb_image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        undistorted_img = distort_by_calibration(rgb_image, self.dst_calib, self.rgb_calib)
        rotated_img = np.rot90(undistorted_img, -1)
        return Image.fromarray(np.ascontiguousarray(rotated_img))

    def stop(self):
        print("Stopping Aria stream...")
        try:
            self.streaming_client.unsubscribe()
            self.streaming_manager.stop_streaming()
            self.device_client.disconnect(self.device)
        except:
            pass

# -----------------------------------------------------------------------------
# Main Inference Class
# -----------------------------------------------------------------------------

class ManualInferenceACT:
    """Manual inference script using Standard ACT architecture with Relative Transforms."""
    
    def __init__(self, checkpoint_path, device=None, aria_args=None, mode='semi-auto'):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode 
        print(f"Using device: {self.device}")
        print(f"Operating Mode: {self.mode.upper()}")
        
        self.checkpoint_path = checkpoint_path
        
        # 1. Load Config & Stats
        self.policy_config = POLICY_CONFIG
        self.task_config = TASK_CONFIG
        self._load_stats()
        
        # 2. Load Model
        self._load_model()
        
        # 3. Setup Transforms
        self.transform = self._get_transform()
        
        # 4. Robot Constants & Frame Initialization
        # Initial Pose [tx, ty, tz, qx, qy, qz, qw]
        self.initial_pose = [-0.128127, 0.271105, 0.288622, -0.785852, 0.001169, 0.032670, 0.617549]
        
        # Compute T_base_device0 (Reference Frame) based on initial_pose
        self.T_base_device0 = self._compute_initial_device_frame()

        # 5. Initialize Robot Connection
        try:
            print("\n🔗 Connecting to robot...")
            self.rtde_c = RTDEControlInterface(ROBOT_IP)
            self.rtde_r = RTDEReceiveInterface(ROBOT_IP)
            print("✅ Robot connected successfully!\n")
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to robot at {ROBOT_IP}: {e}")
            print("   Continuing in simulation mode...")
            self.rtde_c = None
            self.rtde_r = None

        # 6. Initialize Aria Camera
        self.camera = AriaLiveStreamer(
            interface=aria_args.streaming_interface,
            device_ip=aria_args.device_ip,
            profile_name=aria_args.profile_name
        )

    def _compute_initial_device_frame(self):
        """Calculates T_base_device0 from self.initial_pose (TCP) and T_tcp_device."""
        tx, ty, tz, qx, qy, qz, qw = self.initial_pose
        
        # T_base_tcp_0
        T_base_tcp = np.eye(4)
        T_base_tcp[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T_base_tcp[:3, 3] = [tx, ty, tz]
        
        # T_base_device0 = T_base_tcp * T_tcp_device
        T_base_device0 = T_base_tcp @ T_TCP_DEVICE
        return T_base_device0

    def _load_stats(self):
        """
        Load normalization stats.
        Input Dim: 10
        [0:3] : Position (Relative)
        [3:9] : 6D Rotation (Relative)
        [9]   : Distance Class
        """
        dataset_dir = self.task_config['dataset_dir']
        print(f"Loading stats from: {dataset_dir}")
        stats = get_norm_stats(dataset_dir)
        
        # Stats for Denormalization (TX, TY, TZ only)
        self.tx_mean = torch.tensor([stats['tx']['mean']], device=self.device, dtype=torch.float32)
        self.tx_std  = torch.tensor([stats['tx']['std']],  device=self.device, dtype=torch.float32)
        self.ty_mean = torch.tensor([stats['ty']['mean']], device=self.device, dtype=torch.float32)
        self.ty_std  = torch.tensor([stats['ty']['std']],  device=self.device, dtype=torch.float32)
        self.tz_mean = torch.tensor([stats['tz']['mean']], device=self.device, dtype=torch.float32)
        self.tz_std  = torch.tensor([stats['tz']['std']],  device=self.device, dtype=torch.float32)

        print("Constructing 10D qpos stats...")
        # Initialize with Identity (Mean=0, Std=1)
        self.qpos_mean = torch.zeros(10, device=self.device, dtype=torch.float32)
        self.qpos_std  = torch.ones(10,  device=self.device, dtype=torch.float32)
        
        # Fill in specific means for Position
        self.qpos_mean[0] = self.tx_mean
        self.qpos_mean[1] = self.ty_mean
        self.qpos_mean[2] = self.tz_mean
        
        # Fill in specific stds for Position
        self.qpos_std[0] = self.tx_std
        self.qpos_std[1] = self.ty_std
        self.qpos_std[2] = self.tz_std

    def _load_model(self):
        print(f"\n{'='*80}\n📂 LOADING MODEL CHECKPOINT\n{'='*80}")
        try:
            self.policy = make_policy(self.policy_config['policy_class'], self.policy_config)
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.policy.load_state_dict(state_dict)
            self.policy.to(self.device)
            self.policy.eval()
            print(f"✅ MODEL LOADED SUCCESSFULLY\n{'='*80}\n")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            traceback.print_exc()
            raise

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _process_live_image(self, pil_image):
        image_tensor = self.transform(pil_image)
        return image_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
    
    def _normalize_qpos(self, qpos_raw):
        # qpos_raw is 10D
        qpos_tensor = torch.tensor(qpos_raw, dtype=torch.float32, device=self.device).unsqueeze(0) 
        qpos_norm = (qpos_tensor - self.qpos_mean) / self.qpos_std
        return qpos_norm

    def _denormalize_pos_numpy(self, norm_pos_array):
        tx_m, tx_s = self.tx_mean.cpu().numpy(), self.tx_std.cpu().numpy()
        ty_m, ty_s = self.ty_mean.cpu().numpy(), self.ty_std.cpu().numpy()
        tz_m, tz_s = self.tz_mean.cpu().numpy(), self.tz_std.cpu().numpy()

        x = norm_pos_array[..., 0] * tx_s + tx_m
        y = norm_pos_array[..., 1] * ty_s + ty_m
        z = norm_pos_array[..., 2] * tz_s + tz_m
        return np.stack([x, y, z], axis=-1)

    def run_inference(self, qpos_raw, obs_tensor):
        """
        qpos_raw: 10D vector [tx, ty, tz, r6d... , dist] (Relative Frame)
        Returns:
            pred_pos_real: Relative translation (denormalized)
            pred_rot6d: Relative rotation (6D)
            pred_dist: Distance class
        """
        qpos_norm = self._normalize_qpos(qpos_raw)
        
        print("\nRunning inference...")
        with torch.inference_mode():
            action_pred_raw = self.policy(qpos_norm, obs_tensor)
            
        # Denormalize Position (Relative)
        pred_pos_norm = action_pred_raw[..., :3].cpu().numpy()
        pred_pos_real = self._denormalize_pos_numpy(pred_pos_norm)
        
        # Rotation 6D (Relative)
        pred_rot6d = action_pred_raw[..., 3:9].cpu().numpy()
        
        # Distance Class
        pred_dist_prob = action_pred_raw[..., 9:]
        pred_dist = (pred_dist_prob > 0.5).float().cpu().numpy()
        
        return pred_pos_real, pred_rot6d, pred_dist

    def move_to_start(self):
        print("\n" + "=" * 50)
        print("🚀 INITIALIZATION SEQUENCE")
        print("=" * 50)
        print("Moving robot to INITIAL POSE...")
        if self.rtde_c:
            # Move to self.initial_pose
            tx, ty, tz, qx, qy, qz, qw = self.initial_pose
            rx, ry, rz = R.from_quat([qx, qy, qz, qw]).as_rotvec()
            target_pose = [tx, ty, tz, rx, ry, rz]
            try:
                self.rtde_c.moveL(target_pose, 0.2, 0.2)
                time.sleep(2.0)
            except Exception as e:
                print(f"❌ Move failed: {e}")
        else:
            print("   (Simulation Mode) Assuming robot is at initial pose.")

    def get_current_tcp_pose(self):
        """
        Calculates 10D input vector:
        1. Get T_base_tcp from RTDE.
        2. T_base_device = T_base_tcp * T_tcp_device
        3. T_device0_device = inv(T_base_device0) * T_base_device
        4. Vectorize -> [pos(3), rot6d(6), dist(1)]
        """
        # Default/Sim pose
        if self.rtde_r is None:
            # If sim, assume we are at initial pose -> relative is Identity
            pos = [0.0, 0.0, 0.0]
            rot6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] # Identity 6D
            return pos + rot6d + [1.0]

        try:
            # 1. Get Raw Pose (Base -> TCP)
            pose_rtde = self.rtde_r.getActualTCPPose() # [x, y, z, rx, ry, rz]
            
            T_base_tcp = np.eye(4)
            T_base_tcp[:3, :3] = rotvec_to_matrix(pose_rtde[3], pose_rtde[4], pose_rtde[5])
            T_base_tcp[:3, 3] = pose_rtde[:3]

            # 2. Transform to Device Frame
            T_base_device = T_base_tcp @ T_TCP_DEVICE

            # 3. Calculate Relative Frame (T_device0 -> Device)
            # T_device0_device = inv(T_base_device0) @ T_base_device
            T_rel = np.linalg.inv(self.T_base_device0) @ T_base_device

            # 4. Vectorize
            pos_rel = T_rel[:3, 3].tolist()
            rot_rel_mat = T_rel[:3, :3]
            rot_rel_6d = mat_to_rot6d_numpy(rot_rel_mat).tolist()
            
            # Distance Class (Fixed to 1 per instruction)
            dist_class = 1.0

            return pos_rel + rot_rel_6d + [dist_class]

        except Exception as e:
            print(f"❌ Error getting current pose: {e}")
            return None

    def execute_action_chunk(self, pos_chunk, rot6d_chunk, dist_chunk):
        """
        Executes first step of the chunk (Receding Horizon).
        Transforms predicted relative pose back to absolute TCP pose.
        
        Input: Relative Poses T_device0 -> Device
        Logic:
        1. Reconstruct T_rel from pos/rot6d.
        2. T_base_device = T_base_device0 * T_rel
        3. T_base_tcp = T_base_device * T_device_tcp
        4. Move robot.
        """
        if self.rtde_c is None:
            print(f"⚠️  (Sim) Would execute action.")
            return

        # Take first step of the chunk
        # pos_chunk: (1, Seq, 3) -> Take [0, 0]
        t_rel = pos_chunk[0][0] # [x, y, z]
        
        # rot6d_chunk: (1, Seq, 6) -> Take [0, 0]
        r6d = rot6d_chunk[0][0]
        
        # Reconstruct Rotation Matrix from 6D (Need Torch for helper, or custom numpy)
        # Using the torch helper for convenience inside numpy block
        r6d_tensor = torch.tensor(r6d).unsqueeze(0)
        rot_mat_rel = rot6d_to_matrix(r6d_tensor)[0].numpy()

        # 1. Reconstruct T_rel (T_device0 -> Device)
        T_rel = np.eye(4)
        T_rel[:3, :3] = rot_mat_rel
        T_rel[:3, 3] = t_rel

        # 2. Compute T_base_device
        T_base_device = self.T_base_device0 @ T_rel

        # 3. Compute T_base_tcp
        T_base_tcp = T_base_device @ T_DEVICE_TCP

        # 4. Convert to UR5 format (x,y,z, rx,ry,rz)
        x, y, z = T_base_tcp[:3, 3]
        rx, ry, rz = matrix_to_rotvec(T_base_tcp[:3, :3])
        
        target_pose = [x, y, z, rx, ry, rz]
        
        print(f"🤖 Moving robot to: {target_pose[:3]}")
        try:
            self.rtde_c.moveL(target_pose, 0.5, 0.2) 
            time.sleep(0.1) 
        except Exception as e:
            print(f"❌ Error executing action: {e}")

    def interactive_loop(self):
        print("\n" + "=" * 70)
        print("Standard ACT LIVE Camera Inference Script")
        print("=" * 70)
        
        self.move_to_start()
        inference_count = 0
        
        while True:
            inference_count += 1
            print(f"\n{'─' * 70}")
            print(f"Inference #{inference_count} | Mode: {self.mode.upper()}")
            print(f"{'─' * 70}")
            
            if self.mode == 'semi-auto':
                print("\n[SEMI-AUTO] Press ENTER to acquire state/image and predict (or 'q' to quit)...")
                user_input = input("   > ").strip().lower()
                if user_input == 'q':
                    break
            
            # 1. Acquire Robot State (10D Relative)
            print("\n[1/3] Acquiring robot state...")
            qpos_raw = self.get_current_tcp_pose()
            if qpos_raw is None:
                print("Failed to get robot pose.")
                break
                
            # 2. Acquire Image
            print("\n[2/3] Acquiring LIVE image from Aria...")
            obs_pil = None
            for _ in range(10):
                obs_pil = self.camera.get_latest_frame()
                if obs_pil is not None:
                    break
                time.sleep(0.1)
            
            if obs_pil is None:
                print("❌ Could not get frame from camera!")
                continue

            obs_tensor = self._process_live_image(obs_pil)
            
            try:
                print("\n[3/3] Running Model...")
                # Returns relative poses
                pred_pos, pred_rot6d, pred_dist = self.run_inference(qpos_raw, obs_tensor)
                
                # Execute (includes transformation back to absolute)
                self.execute_action_chunk(pred_pos, pred_rot6d, pred_dist)

            except Exception as e:
                print(f"✗ EXCEPTION CAUGHT during inference: {e}")
                traceback.print_exc()
                if self.mode == 'auto':
                    break
        
        print(f"\nCompleted {inference_count} inference(s). Goodbye!")    

    def __del__(self):
        try:
            if self.rtde_c: 
                self.rtde_c.stopScript()
                self.rtde_c.disconnect()
            if self.rtde_r: self.rtde_r.disconnect()
            if hasattr(self, 'camera'): self.camera.stop()
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Live Inference with Aria & ACT")
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/policy_best.ckpt', help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--mode', type=str, default='semi-auto', choices=['auto', 'semi-auto'])
    parser.add_argument("--interface", dest="streaming_interface", type=str, default="usb", choices=["usb", "wifi"])
    parser.add_argument("--profile", dest="profile_name", type=str, default="profile28")
    parser.add_argument("--device-ip", help="Aria IP for wifi")
    
    args = parser.parse_args()
    
    inferencer = ManualInferenceACT(
        checkpoint_path=args.checkpoint,
        device=args.device,
        aria_args=args,
        mode=args.mode
    )
    inferencer.interactive_loop()

if __name__ == '__main__':
    main()