##
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

ROBOT_IP = "192.168.1.23"  # Robot IP

# -----------------------------------------------------------------------------
# Math & Transformation Constants
# -----------------------------------------------------------------------------

R_TCP_DEVICE_RAW = np.array([
    [ 0.99920421, -0.03675654,  0.01548867],
    [ 0.03020608,  0.95091268,  0.30798161],
    [-0.02604871, -0.30726867,  0.95126622]
])
t_TCP_DEVICE_VEC = np.array([0.01260316, -0.09558874, 0.06506849])

R_CORRECTION = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0]
])

R_TCP_DEVICE_MAT = R_TCP_DEVICE_RAW @ R_CORRECTION

T_TCP_DEVICE = np.eye(4)
T_TCP_DEVICE[:3, :3] = R_TCP_DEVICE_MAT
T_TCP_DEVICE[:3, 3] = t_TCP_DEVICE_VEC

T_DEVICE_TCP = np.linalg.inv(T_TCP_DEVICE)

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
    def __init__(self, interface="usb", device_ip=None, profile_name="profile28"):
        print(f"\n👓 Initializing Aria Glasses ({interface})...")
        aria.set_log_level(aria.Level.Info)
        self.device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        if device_ip:
            client_config.ip_v4_address = device_ip
        self.device_client.set_client_config(client_config)

        try:
            self.device = self.device_client.connect()
        except Exception as e:
            print(f"❌ Failed to connect to Aria Device: {e}")
            raise e

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
        
        self.dst_calib = get_linear_camera_calibration(
            target_width, target_height, focal_length, "camera-rgb"
        )

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
        if raw_img is None:
            return None
        undistorted_img = distort_by_calibration(raw_img, self.dst_calib, self.rgb_calib)
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
    def __init__(self, checkpoint_path, device=None, aria_args=None, mode='semi-auto'):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode 
        print(f"Using device: {self.device}")
        print(f"Operating Mode: {self.mode.upper()}")
        
        self.checkpoint_path = checkpoint_path
        
        # 1. Create directory for saving images
        self.image_save_dir = "./inference_images"
        os.makedirs(self.image_save_dir, exist_ok=True)
        print(f"📂 Inference images will be saved to: {self.image_save_dir}")
        
        # 2. Load Config & Stats
        self.policy_config = POLICY_CONFIG
        self.task_config = TASK_CONFIG
        self._load_stats()
        
        # 3. Load Model
        self._load_model()
        
        # 4. Setup Transforms
        self.transform = self._get_transform()
        
        # 5. Robot Initialization
        self.initial_pose = [0.21237585739957632, 0.019320201601764633, 0.33314827243319656, -0.7465386191605052, 0.31175669465321637, -0.2768194313529758, 0.5185160131242258]
        try:
            print("\n🔗 Connecting to robot...")
            self.rtde_c = RTDEControlInterface(ROBOT_IP)
            self.rtde_r = RTDEReceiveInterface(ROBOT_IP)
            print("✅ Robot connected successfully!\n")
        except Exception as e:
            print(f"⚠️ Warning: Could not connect to robot: {e}")
            self.rtde_c = None
            self.rtde_r = None

        # 6. Initialize Aria Camera
        self.camera = AriaLiveStreamer(
            interface=aria_args.streaming_interface,
            device_ip=aria_args.device_ip,
            profile_name=aria_args.profile_name
        )

    def _compute_initial_device_frame(self):
        tx, ty, tz, qx, qy, qz, qw = self.initial_pose
        T_base_tcp = np.eye(4)
        T_base_tcp[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T_base_tcp[:3, 3] = [tx, ty, tz]
        return T_base_tcp @ T_TCP_DEVICE

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
        self.qpos_mean[0:3] = torch.cat([self.tx_mean, self.ty_mean, self.tz_mean])
        self.qpos_std[0:3]  = torch.cat([self.tx_std, self.ty_std, self.tz_std])

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
            traceback.print_exc()
            raise

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _process_live_image(self, pil_image):
        image_tensor = self.transform(pil_image)
        return image_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
    
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

    def run_inference(self, qpos_raw, obs_tensor):
        qpos_norm = self._normalize_qpos(qpos_raw)
        with torch.inference_mode():
            action_pred_raw = self.policy(qpos_norm, obs_tensor)
        pred_pos_real = self._denormalize_pos_numpy(action_pred_raw[..., :3].cpu().numpy())
        pred_rot6d = action_pred_raw[..., 3:9].cpu().numpy()
        pred_dist = (action_pred_raw[..., 9:] > 0.5).float().cpu().numpy()
        return pred_pos_real, pred_rot6d, pred_dist

    def move_to_start(self):
        print("Moving robot to INITIAL POSE...")
        if self.rtde_c:
            tx, ty, tz, qx, qy, qz, qw = self.initial_pose
            rx, ry, rz = R.from_quat([qx, qy, qz, qw]).as_rotvec()
            try:
                self.rtde_c.moveL([tx, ty, tz, rx, ry, rz], 0.2, 0.2)
                time.sleep(2.0)
            except Exception as e:
                print(f"❌ Move failed: {e}")

    def get_current_tcp_pose(self):
        if self.rtde_r is None:
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
        try:
            pose_rtde = self.rtde_r.getActualTCPPose()
            T_base_tcp = np.eye(4)
            T_base_tcp[:3, :3] = rotvec_to_matrix(pose_rtde[3], pose_rtde[4], pose_rtde[5])
            T_base_tcp[:3, 3] = pose_rtde[:3]
            T_base_device = T_base_tcp @ T_TCP_DEVICE
            T_rel = np.linalg.inv(self.T_base_device0) @ T_base_device
            return T_rel[:3, 3].tolist() + mat_to_rot6d_numpy(T_rel[:3, :3]).tolist() + [1.0]
        except Exception as e:
            print(f"❌ Error getting current pose: {e}")
            return None

    def execute_action_chunk(self, pos_chunk, rot6d_chunk, dist_chunk):
        if self.rtde_c is None:
            return
        t_rel = pos_chunk[0][0]
        r6d_tensor = torch.tensor(rot6d_chunk[0][0]).unsqueeze(0)
        rot_mat_rel = rot6d_to_matrix(r6d_tensor)[0].numpy()
        T_rel = np.eye(4)
        T_rel[:3, :3], T_rel[:3, 3] = rot_mat_rel, t_rel
        T_base_tcp = (self.T_base_device0 @ T_rel) @ T_DEVICE_TCP
        x, y, z = T_base_tcp[:3, 3]
        rx, ry, rz = matrix_to_rotvec(T_base_tcp[:3, :3])
        try:
            self.rtde_c.moveL([x, y, z, rx, ry, rz], 0.5, 0.2) 
            time.sleep(0.1) 
        except Exception as e:
            print(f"❌ Error executing action: {e}")

    def interactive_loop(self):
        self.move_to_start()
        self.T_base_device0 = self._compute_initial_device_frame()
        inference_count = 0
        
        while True:
            inference_count += 1
            print(f"\nInference #{inference_count} | Mode: {self.mode.upper()}")
            
            if self.mode == 'semi-auto':
                user_input = input("Press ENTER to run (or 'q' to quit): ").strip().lower()
                if user_input == 'q': break
            
            qpos_raw = self.get_current_tcp_pose()
            if qpos_raw is None: break
                
            obs_pil = None
            for _ in range(10):
                obs_pil = self.camera.get_latest_frame()
                if obs_pil is not None: break
                time.sleep(0.1)
            
            if obs_pil is None:
                print("❌ Frame timeout!"); continue

            # --- SAVE THE IMAGE ---
            save_path = os.path.join(self.image_save_dir, f"inference_{inference_count:04d}.png")
            try:
                obs_pil.save(save_path)
                print(f"   💾 Saved: {save_path}")
            except Exception as e:
                print(f"   ⚠️ Save Error: {e}")

            obs_tensor = self._process_live_image(obs_pil)
            try:
                pred_pos, pred_rot6d, pred_dist = self.run_inference(qpos_raw, obs_tensor)
                self.execute_action_chunk(pred_pos, pred_rot6d, pred_dist)
            except Exception as e:
                traceback.print_exc()
                if self.mode == 'auto': break

    def __del__(self):
        try:
            if self.rtde_c: self.rtde_c.disconnect()
            if self.rtde_r: self.rtde_r.disconnect()
            if hasattr(self, 'camera'): self.camera.stop()
        except: pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/policy_best.ckpt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='semi-auto')
    parser.add_argument("--interface", dest="streaming_interface", type=str, default="usb")
    parser.add_argument("--profile", dest="profile_name", type=str, default="profile28")
    parser.add_argument("--device-ip", help="Aria IP for wifi")
    args = parser.parse_args()
    
    inferencer = ManualInferenceACT(args.checkpoint, args.device, args, args.mode)
    inferencer.interactive_loop()

if __name__ == '__main__':
    main()