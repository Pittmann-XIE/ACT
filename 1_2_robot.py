# # --- START OF FILE 1_2_robot.py ---

# # --- Environment Constants ---
# import torch
# import numpy as np
# import argparse
# import sys
# import os
# import time
# import traceback
# import zmq
# import pickle
# from PIL import Image
# from scipy.spatial.transform import Rotation as R
# import torchvision.transforms as transforms

# # --- Physics & Planning Imports ---
# import mplib
# import pybullet as p
# import pybullet_data

# # --- Project Imports ---
# sys.path.append(os.getcwd())
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from config.config_1 import POLICY_CONFIG, TASK_CONFIG
# from training.utils_1 import make_policy, get_norm_stats

# # --- Robot Interface Imports ---
# try:
#     from rtde_control import RTDEControlInterface
#     from rtde_receive import RTDEReceiveInterface
# except ImportError:
#     print("⚠️ RTDE modules not found. Running in simulation/dummy mode.")

# # ---------------------------------------------------------
# # 1. CONFIGURATION
# # ---------------------------------------------------------
# ROBOT_IP = '10.0.0.1' 

# # --- MACRO VARIABLES FOR SPEED CONTROL ---
# ROBOT_SPEED = 0.10      
# ROBOT_ACCEL = 0.05      

# # --- Environment Constants ---

# # !!! CHANGED: Define Initial TCP Pose (Base Frame) [x, y, z, rx, ry, rz] !!!
# # You must update this to your desired starting position.
# # Example format: [-0.1, -0.4, 0.3, 2.2, -2.2, 0.0]
# INIT_TCP_POSE = [0.299, 0.146, 0.434, 3.932, -1.554, 1.595]

# SIM_ROBOT_WORLD_ROT_Z = np.pi 

# # Obstacles
# WALL_POS_WORLD = [-0.14, -0.14, 0.5]  
# WALL_ROT_Z_DEG = 45  
# WALL_SIZE = [0.02, 1.0, 1.0] 
# TABLE_SIZE = [1.0, 1.0, 0.1]
# TABLE_POS_WORLD = [0, 0, -TABLE_SIZE[2]/2-0.01]
# TABLE_ROT_Z_DEG = 0

# # --- ACT Coordinate Transforms ---
# R_TCP_DEVICE_RAW = np.array([
#     [ 0.99920421, -0.03675654,  0.01548867],
#     [ 0.03020608,  0.95091268,  0.30798161],
#     [-0.02604871, -0.30726867,  0.95126622]
# ])
# R_CORRECTION = np.array([
#     [0.0, -1.0, 0.0],
#     [1.0,  0.0, 0.0],
#     [0.0,  0.0, 1.0]
# ])
# R_TCP_DEVICE_MAT = R_TCP_DEVICE_RAW @ R_CORRECTION
# t_TCP_DEVICE_VEC = np.array([0.01260316, -0.09558874, 0.06506849])

# T_TCP_DEVICE = np.eye(4)
# T_TCP_DEVICE[:3, :3] = R_TCP_DEVICE_MAT
# T_TCP_DEVICE[:3, 3] = t_TCP_DEVICE_VEC
# T_DEVICE_TCP = np.linalg.inv(T_TCP_DEVICE)

# # ---------------------------------------------------------
# # 2. HELPER FUNCTIONS
# # ---------------------------------------------------------
# def get_object_transforms(pos_world_arr, rot_z_deg):
#     pos_world = np.array(pos_world_arr)
#     r_obj_local = R.from_euler('z', rot_z_deg, degrees=True)
#     r_world_robot = R.from_euler('z', -SIM_ROBOT_WORLD_ROT_Z, degrees=False)
#     pos_robot = r_world_robot.apply(pos_world)
#     quat_robot = (r_world_robot * r_obj_local).as_quat()
#     quat_world = r_obj_local.as_quat()
#     return pos_robot, quat_robot, pos_world, quat_world

# def get_box_point_cloud(center, size, quat, resolution=0.04):
#     x = np.arange(-size[0]/2, size[0]/2, resolution)
#     y = np.arange(-size[1]/2, size[1]/2, resolution)
#     z = np.arange(-size[2]/2, size[2]/2, resolution)
#     if len(x)==0: x=[0]
#     if len(y)==0: y=[0]
#     if len(z)==0: z=[0]
#     xx, yy, zz = np.meshgrid(x, y, z)
#     points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
#     points_rotated = R.from_quat(quat).apply(points)
#     return points_rotated + center

# def draw_coordinate_frame(pos, quat, axis_len=0.1, life_time=0):
#     rot = R.from_quat(quat).as_matrix()
#     p.addUserDebugLine(pos, pos + rot[:,0]*axis_len, [1,0,0], lineWidth=2, lifeTime=life_time) 
#     p.addUserDebugLine(pos, pos + rot[:,1]*axis_len, [0,1,0], lineWidth=2, lifeTime=life_time) 
#     p.addUserDebugLine(pos, pos + rot[:,2]*axis_len, [0,0,1], lineWidth=2, lifeTime=life_time) 

# def rot6d_to_matrix(rot_6d):
#     x_raw = rot_6d[..., 0:3]
#     y_raw = rot_6d[..., 3:6]
#     x = torch.nn.functional.normalize(x_raw, dim=-1)
#     z = torch.cross(x, y_raw, dim=-1)
#     z = torch.nn.functional.normalize(z, dim=-1)
#     y = torch.cross(z, x, dim=-1)
#     return torch.stack([x, y, z], dim=-1)

# def mat_to_rot6d_numpy(mat):
#     return mat[:, :2].flatten(order='F')

# def rotvec_to_matrix(rx, ry, rz):
#     return R.from_rotvec([rx, ry, rz]).as_matrix()

# # -----------------------------------------------------------------------------
# # 3. Remote Camera Helper
# # -----------------------------------------------------------------------------
# class RemoteCameraClient:
#     def __init__(self, port="5555"):
#         print(f"📡 Connecting to Camera Stream on localhost:{port}...")
#         self.context = zmq.Context()
#         self.socket = self.context.socket(zmq.SUB)
#         self.socket.connect(f"tcp://localhost:{port}")
#         self.socket.setsockopt(zmq.SUBSCRIBE, b"")
#         self.socket.setsockopt(zmq.CONFLATE, 1) 
#         print("✅ Camera Client Connected.")

#     def get_latest_frame(self):
#         try:
#             data = self.socket.recv(flags=zmq.NOBLOCK)
#             return pickle.loads(data)
#         except zmq.Again:
#             return None
#         except Exception as e:
#             print(f"Error receiving image: {e}")
#             return None

# # -----------------------------------------------------------------------------
# # 4. Main Inference Class
# # -----------------------------------------------------------------------------
# class ManualInferenceACT:
#     def __init__(self, checkpoint_path, device=None, zmq_port="5555"):
#         self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"Using device: {self.device}")
#         self.checkpoint_path = checkpoint_path
        
#         self.policy_config = POLICY_CONFIG
#         self.task_config = TASK_CONFIG
#         self._load_stats()
#         self._load_model()
#         self.transform = self._get_transform()
        
#         self.camera = RemoteCameraClient(port=zmq_port)

#         self.rtde_c = None
#         self.rtde_r = None
#         try:
#             print(f"\n🔗 Connecting to robot at {ROBOT_IP}...")
#             self.rtde_c = RTDEControlInterface(ROBOT_IP)
#             self.rtde_r = RTDEReceiveInterface(ROBOT_IP)
#             print("✅ Robot connected successfully!\n")
#         except Exception as e:
#             print(f"⚠️ Warning: Could not connect to robot: {e}")

#         self._setup_physics_and_collision()
#         self.T_base_device0 = None 

#     def _setup_physics_and_collision(self):
#         print("🛡️ Initializing Physics and Collision Checkers...")
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         self.urdf_path = os.path.join(current_dir, "ur3_gripper.urdf") 
#         self.srdf_path = os.path.join(current_dir, "ur3_gripper.srdf")
        
#         p.connect(p.GUI) 
#         p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         p.setGravity(0, 0, -9.81)
#         p.resetDebugVisualizerCamera(1.2, 90, -30, [0, 0, 0])
#         p.loadURDF("plane.urdf")    

#         # Obstacles
#         wall_pos_r, wall_quat_r, wall_pos_w, wall_quat_w = get_object_transforms(WALL_POS_WORLD, WALL_ROT_Z_DEG)
#         table_pos_r, table_quat_r, table_pos_w, table_quat_w = get_object_transforms(TABLE_POS_WORLD, TABLE_ROT_Z_DEG)

#         p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE]), 
#                           basePosition=wall_pos_w, baseOrientation=wall_quat_w,
#                           baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE], rgbaColor=[0.8, 0.2, 0.2, 0.8]))
#         p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE]), 
#                           basePosition=table_pos_w, baseOrientation=table_quat_w,
#                           baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE], rgbaColor=[0.5, 0.3, 0.1, 1.0]))

#         # Robot
#         sim_robot_orn = p.getQuaternionFromEuler([0, 0, SIM_ROBOT_WORLD_ROT_Z])
#         self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0], baseOrientation=sim_robot_orn, useFixedBase=1)
        
#         joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
#         name_to_idx = {p.getJointInfo(self.robot_id, i)[1].decode('utf-8'): i for i in range(p.getNumJoints(self.robot_id))}
#         self.active_indices = [name_to_idx.get(n, -1) for n in joint_names]

#         self.ee_link_idx = -1
#         for i in range(p.getNumJoints(self.robot_id)):
#             if p.getJointInfo(self.robot_id, i)[12].decode("utf-8") == "tool0": 
#                 self.ee_link_idx = i; break

#         # Initialize planner
#         self.planner = mplib.Planner(urdf=self.urdf_path, srdf=self.srdf_path, move_group="tool0")
#         combined_pts = np.vstack([
#             get_box_point_cloud(wall_pos_r, WALL_SIZE, wall_quat_r, resolution=0.04), 
#             get_box_point_cloud(table_pos_r, TABLE_SIZE, table_quat_r, resolution=0.04)
#         ])
#         self.planner.update_point_cloud(combined_pts)
#         print("✅ Environment setup complete.")

#     def _load_stats(self):
#         dataset_dir = self.task_config['dataset_dir']
#         stats = get_norm_stats(dataset_dir)
#         self.tx_mean = torch.tensor([stats['tx']['mean']], device=self.device, dtype=torch.float32)
#         self.tx_std  = torch.tensor([stats['tx']['std']],  device=self.device, dtype=torch.float32)
#         self.ty_mean = torch.tensor([stats['ty']['mean']], device=self.device, dtype=torch.float32)
#         self.ty_std  = torch.tensor([stats['ty']['std']],  device=self.device, dtype=torch.float32)
#         self.tz_mean = torch.tensor([stats['tz']['mean']], device=self.device, dtype=torch.float32)
#         self.tz_std  = torch.tensor([stats['tz']['std']],  device=self.device, dtype=torch.float32)
        
#         self.qpos_mean = torch.zeros(10, device=self.device, dtype=torch.float32)
#         self.qpos_std  = torch.ones(10,  device=self.device, dtype=torch.float32)
#         self.qpos_mean[0] = self.tx_mean
#         self.qpos_mean[1] = self.ty_mean
#         self.qpos_mean[2] = self.tz_mean
#         self.qpos_std[0]  = self.tx_std
#         self.qpos_std[1]  = self.ty_std
#         self.qpos_std[2]  = self.tz_std

#     def _load_model(self):
#         self.policy = make_policy(self.policy_config['policy_class'], self.policy_config)
#         state_dict = torch.load(self.checkpoint_path, map_location=self.device)
#         self.policy.load_state_dict(state_dict)
#         self.policy.to(self.device)
#         self.policy.eval()
#         print("✅ Model Loaded.")

#     def _get_transform(self):
#         return transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def _normalize_qpos(self, qpos_raw):
#         qpos_tensor = torch.tensor(qpos_raw, dtype=torch.float32, device=self.device).unsqueeze(0) 
#         return (qpos_tensor - self.qpos_mean) / self.qpos_std

#     def _denormalize_pos_numpy(self, norm_pos_array):
#         tx_m, tx_s = self.tx_mean.cpu().numpy(), self.tx_std.cpu().numpy()
#         ty_m, ty_s = self.ty_mean.cpu().numpy(), self.ty_std.cpu().numpy()
#         tz_m, tz_s = self.tz_mean.cpu().numpy(), self.tz_std.cpu().numpy()
#         x = norm_pos_array[..., 0] * tx_s + tx_m
#         y = norm_pos_array[..., 1] * ty_s + ty_m
#         z = norm_pos_array[..., 2] * tz_s + tz_m
#         return np.stack([x, y, z], axis=-1)

#     # -------------------------------------------------------------------------
#     # COORDINATE TRANSFORMS
#     # -------------------------------------------------------------------------
#     def compute_T_base_device0(self, current_pose_vec):
#         T_base_tcp = np.eye(4)
#         T_base_tcp[:3, :3] = rotvec_to_matrix(*current_pose_vec[3:])
#         T_base_tcp[:3, 3] = current_pose_vec[:3]
#         self.T_base_device0 = T_base_tcp @ T_TCP_DEVICE
#         print("✅ Initial Reference Frame Set (T_base_device0).")

#     def get_current_pose_relative(self):
#         if self.rtde_r is None: 
#             return [0,0,0, 1,0,0,0,1,0, 0] 
#         pose_rtde = self.rtde_r.getActualTCPPose() 
#         T_base_tcp = np.eye(4)
#         T_base_tcp[:3, :3] = rotvec_to_matrix(*pose_rtde[3:])
#         T_base_tcp[:3, 3] = pose_rtde[:3]
#         T_base_device = T_base_tcp @ T_TCP_DEVICE
#         T_rel = np.linalg.inv(self.T_base_device0) @ T_base_device
#         pos_rel = T_rel[:3, 3].tolist()
#         rot6d_rel = mat_to_rot6d_numpy(T_rel[:3, :3]).tolist()
#         return pos_rel + rot6d_rel + [1.0]

#     def convert_relative_to_absolute(self, rel_pos, rel_rot6d):
#         r6d_tensor = torch.tensor(rel_rot6d).unsqueeze(0)
#         rot_mat_rel = rot6d_to_matrix(r6d_tensor)[0].numpy()
#         T_rel = np.eye(4)
#         T_rel[:3, :3] = rot_mat_rel
#         T_rel[:3, 3] = rel_pos
#         T_base_device_target = self.T_base_device0 @ T_rel
#         T_base_tcp_target = T_base_device_target @ T_DEVICE_TCP
#         target_pos = T_base_tcp_target[:3, 3]
#         target_quat = R.from_matrix(T_base_tcp_target[:3, :3]).as_quat() 
#         return target_pos, target_quat

#     # -------------------------------------------------------------------------
#     # EXECUTION WITH moveL (Linear Cartesian Move)
#     # -------------------------------------------------------------------------
#     def execute_absolute_action(self, target_pos, target_quat):
#         # 1. Visualize target frame in PyBullet
#         draw_coordinate_frame(target_pos, target_quat, life_time=5.0)

#         # 2. Sync Simulation with Real Robot (for accurate collision check)
#         if self.rtde_r is not None:
#             actual_q = self.rtde_r.getActualQ()
#             for i, angle in enumerate(actual_q):
#                 p.resetJointState(self.robot_id, self.active_indices[i], angle)

#         # 3. Calculate IK (Only for collision checking now, not for execution)
#         ik_solution = p.calculateInverseKinematics(
#             self.robot_id, 
#             self.ee_link_idx, 
#             targetPosition=target_pos, 
#             targetOrientation=target_quat,
#             maxNumIterations=50,
#             residualThreshold=1e-4
#         )
#         q_proposal = list(ik_solution[:6]) 

#         # 4. Collision Check
#         is_self_collision = self.planner.check_for_self_collision(q_proposal)
#         is_env_collision = self.planner.check_for_env_collision(q_proposal)

#         if is_self_collision or is_env_collision:
#             p.addUserDebugText("COLLISION!", [0,0,1], [1,0,0], lifeTime=1.0, textSize=2)
#             print(f"🛑 COLLISION DETECTED! Stopping. (Self: {is_self_collision}, Env: {is_env_collision})")
#             return 

#         # 5. Prepare Cartesian Target for UR (moveL)
#         # Convert Quat (PyBullet) -> RotVec (UR Control)
#         rot_obj = R.from_quat(target_quat)
#         rot_vec = rot_obj.as_rotvec() # [rx, ry, rz]
#         target_pose_6d = list(target_pos) + list(rot_vec)

#         print(f"\n⚡ Target (Cartesian): {np.around(target_pose_6d, 3)}")
        
#         # 6. Execute
#         user_input = input(">> Press ENTER to Execute moveL, 's' to Skip, 'q' to Quit: ").lower()
        
#         if user_input == 'q':
#             print("Quitting...")
#             sys.exit(0)
#         elif user_input == 's':
#             print("Skipping step.")
#             return

#         # Move PyBullet visual
#         p.setJointMotorControlArray(self.robot_id, self.active_indices, p.POSITION_CONTROL, 
#                                     targetPositions=q_proposal, forces=[200.0]*6)
#         p.stepSimulation()

#         # Move Real Robot
#         if self.rtde_c is not None:
#             try:
#                 print(f"   Moving Real Robot (moveL)...")
#                 # moveL ensures linear path and prevents 'unwinding' rotation issues
#                 self.rtde_c.moveL(target_pose_6d, ROBOT_SPEED, ROBOT_ACCEL)
#                 print("   Done.")
#             except Exception as e:
#                 print(f"Error in moveL: {e}")
#                 print("Check if target is reachable or near a singularity.")
#         else:
#             print("   (Simulated) Move Complete.")

#     def loop(self):
#         print("\n🚀 Starting Inference Loop (Waiting for Camera)...")
        
#         if self.rtde_c:
#             print(f"Moving to initial Cartesian pose: {INIT_TCP_POSE} ...")
#             try:
#                 # Use moveL to go to start pose linearly
#                 self.rtde_c.moveL(INIT_TCP_POSE, ROBOT_SPEED, ROBOT_ACCEL)
#             except Exception as e:
#                 print(f"⚠️ Failed to move to INIT_TCP_POSE: {e}")
#                 return

#             time.sleep(1.0)
            
#             # Sync PyBullet to the real robot's new position
#             actual_q = self.rtde_r.getActualQ()
#             for i, angle in enumerate(actual_q):
#                 p.resetJointState(self.robot_id, self.active_indices[i], angle)
#             p.stepSimulation()
            
#             # Set Reference Frame
#             init_pose_vec = self.rtde_r.getActualTCPPose()
#             self.compute_T_base_device0(init_pose_vec)
#         else:
#             print("⚠️ Sim-only mode: Setting Identity Reference Frame.")
#             self.T_base_device0 = np.eye(4) 
#             # In sim mode, snap PyBullet to INIT_TCP_POSE via IK for visualization
#             t_pos = INIT_TCP_POSE[:3]
#             t_rot = rotvec_to_matrix(*INIT_TCP_POSE[3:])
#             t_quat = R.from_matrix(t_rot).as_quat()
#             ik_sol = p.calculateInverseKinematics(self.robot_id, self.ee_link_idx, t_pos, t_quat)
#             for i, angle in enumerate(ik_sol[:6]):
#                 p.resetJointState(self.robot_id, self.active_indices[i], angle)

#         step_idx = 0
#         while True:
#             obs_pil = self.camera.get_latest_frame()
#             if obs_pil is None: 
#                 time.sleep(0.01)
#                 continue

#             qpos_rel_raw = self.get_current_pose_relative()

#             obs_tensor = self.transform(obs_pil).unsqueeze(0).unsqueeze(0).to(self.device)
#             qpos_norm = self._normalize_qpos(qpos_rel_raw)
            
#             with torch.inference_mode():
#                 action_pred = self.policy(qpos_norm, obs_tensor) 
            
#             action_chunk = action_pred.squeeze(0).squeeze(0).cpu().numpy()
#             pred_pos_rel = self._denormalize_pos_numpy(action_chunk[..., :3])[0]
#             pred_rot6d_rel = action_chunk[..., 3:9][0]
            
#             target_pos_abs, target_quat_abs = self.convert_relative_to_absolute(pred_pos_rel, pred_rot6d_rel)
#             self.execute_absolute_action(target_pos_abs, target_quat_abs)
            
#             step_idx += 1

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--checkpoint', type=str, default='/home/pt/github/ws_ros2humble-main_lab/ACT/checkpoints/policy_epoch_60_seed_42.ckpt')
#     parser.add_argument("--port", type=str, default="5555")
#     args = parser.parse_args()
    
#     agent = ManualInferenceACT(args.checkpoint, zmq_port=args.port)
#     agent.loop()

# if __name__ == '__main__':
#     main()
# --- START OF FILE 1_2_robot.py ---

# # --- Environment Constants ---
# import torch
# import numpy as np
# import argparse
# import sys
# import os
# import time
# import traceback
# import zmq
# import pickle
# import csv  # <--- NEW
# from PIL import Image
# from scipy.spatial.transform import Rotation as R
# import torchvision.transforms as transforms

# # --- Physics & Planning Imports ---
# import mplib
# import pybullet as p
# import pybullet_data

# # --- Project Imports ---
# sys.path.append(os.getcwd())
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from config.config_1 import POLICY_CONFIG, TASK_CONFIG
# from training.utils_1 import make_policy, get_norm_stats

# # --- Robot Interface Imports ---
# try:
#     from rtde_control import RTDEControlInterface
#     from rtde_receive import RTDEReceiveInterface
# except ImportError:
#     print("⚠️ RTDE modules not found. Running in simulation/dummy mode.")

# # ---------------------------------------------------------
# # 1. CONFIGURATION
# # ---------------------------------------------------------
# ROBOT_IP = '10.0.0.1' 

# # --- MACRO VARIABLES FOR SPEED CONTROL ---
# ROBOT_SPEED = 0.08      
# ROBOT_ACCEL = 0.05      

# # --- Environment Constants ---

# # !!! Initial TCP Pose (Base Frame) [x, y, z, rx, ry, rz] !!!
# INIT_TCP_POSE = [0.299, 0.146, 0.434, 3.932, -1.554, 1.595]

# SIM_ROBOT_WORLD_ROT_Z = np.pi 

# # Obstacles
# WALL_POS_WORLD = [-0.14, -0.14, 0.5]  
# WALL_ROT_Z_DEG = 45  
# WALL_SIZE = [0.02, 1.0, 1.0] 
# TABLE_SIZE = [1.0, 1.0, 0.1]
# TABLE_POS_WORLD = [0, 0, -TABLE_SIZE[2]/2-0.01]
# TABLE_ROT_Z_DEG = 0

# # --- ACT Coordinate Transforms ---
# R_TCP_DEVICE_RAW = np.array([
#     [ 0.99920421, -0.03675654,  0.01548867],
#     [ 0.03020608,  0.95091268,  0.30798161],
#     [-0.02604871, -0.30726867,  0.95126622]
# ])
# R_CORRECTION = np.array([
#     [0.0, -1.0, 0.0],
#     [1.0,  0.0, 0.0],
#     [0.0,  0.0, 1.0]
# ])
# R_TCP_DEVICE_MAT = R_TCP_DEVICE_RAW @ R_CORRECTION
# t_TCP_DEVICE_VEC = np.array([0.01260316, -0.09558874, 0.06506849])

# T_TCP_DEVICE = np.eye(4)
# T_TCP_DEVICE[:3, :3] = R_TCP_DEVICE_MAT
# T_TCP_DEVICE[:3, 3] = t_TCP_DEVICE_VEC
# T_DEVICE_TCP = np.linalg.inv(T_TCP_DEVICE)

# # ---------------------------------------------------------
# # 2. HELPER FUNCTIONS
# # ---------------------------------------------------------
# def get_object_transforms(pos_world_arr, rot_z_deg):
#     pos_world = np.array(pos_world_arr)
#     r_obj_local = R.from_euler('z', rot_z_deg, degrees=True)
#     r_world_robot = R.from_euler('z', -SIM_ROBOT_WORLD_ROT_Z, degrees=False)
#     pos_robot = r_world_robot.apply(pos_world)
#     quat_robot = (r_world_robot * r_obj_local).as_quat()
#     quat_world = r_obj_local.as_quat()
#     return pos_robot, quat_robot, pos_world, quat_world

# def get_box_point_cloud(center, size, quat, resolution=0.04):
#     x = np.arange(-size[0]/2, size[0]/2, resolution)
#     y = np.arange(-size[1]/2, size[1]/2, resolution)
#     z = np.arange(-size[2]/2, size[2]/2, resolution)
#     if len(x)==0: x=[0]
#     if len(y)==0: y=[0]
#     if len(z)==0: z=[0]
#     xx, yy, zz = np.meshgrid(x, y, z)
#     points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
#     points_rotated = R.from_quat(quat).apply(points)
#     return points_rotated + center

# def draw_coordinate_frame(pos, quat, axis_len=0.1, life_time=0):
#     rot = R.from_quat(quat).as_matrix()
#     p.addUserDebugLine(pos, pos + rot[:,0]*axis_len, [1,0,0], lineWidth=2, lifeTime=life_time) 
#     p.addUserDebugLine(pos, pos + rot[:,1]*axis_len, [0,1,0], lineWidth=2, lifeTime=life_time) 
#     p.addUserDebugLine(pos, pos + rot[:,2]*axis_len, [0,0,1], lineWidth=2, lifeTime=life_time) 

# def rot6d_to_matrix(rot_6d):
#     x_raw = rot_6d[..., 0:3]
#     y_raw = rot_6d[..., 3:6]
#     x = torch.nn.functional.normalize(x_raw, dim=-1)
#     z = torch.cross(x, y_raw, dim=-1)
#     z = torch.nn.functional.normalize(z, dim=-1)
#     y = torch.cross(z, x, dim=-1)
#     return torch.stack([x, y, z], dim=-1)

# def mat_to_rot6d_numpy(mat):
#     return mat[:, :2].flatten(order='F')

# def rotvec_to_matrix(rx, ry, rz):
#     return R.from_rotvec([rx, ry, rz]).as_matrix()

# # -----------------------------------------------------------------------------
# # 3. Remote Camera Helper
# # -----------------------------------------------------------------------------
# class RemoteCameraClient:
#     def __init__(self, port="5555"):
#         print(f"📡 Connecting to Camera Stream on localhost:{port}...")
#         self.context = zmq.Context()
#         self.socket = self.context.socket(zmq.SUB)
#         self.socket.connect(f"tcp://localhost:{port}")
#         self.socket.setsockopt(zmq.SUBSCRIBE, b"")
#         self.socket.setsockopt(zmq.CONFLATE, 1) 
#         print("✅ Camera Client Connected.")

#     def get_latest_frame(self):
#         try:
#             data = self.socket.recv(flags=zmq.NOBLOCK)
#             return pickle.loads(data)
#         except zmq.Again:
#             return None
#         except Exception as e:
#             print(f"Error receiving image: {e}")
#             return None

# # -----------------------------------------------------------------------------
# # 4. Main Inference Class
# # -----------------------------------------------------------------------------
# class ManualInferenceACT:
#     def __init__(self, checkpoint_path, device=None, zmq_port="5555"):
#         self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"Using device: {self.device}")
#         self.checkpoint_path = checkpoint_path
        
#         self.policy_config = POLICY_CONFIG
#         self.task_config = TASK_CONFIG
#         self._load_stats()
#         self._load_model()
#         self.transform = self._get_transform()
        
#         self.camera = RemoteCameraClient(port=zmq_port)

#         self.rtde_c = None
#         self.rtde_r = None
#         try:
#             print(f"\n🔗 Connecting to robot at {ROBOT_IP}...")
#             self.rtde_c = RTDEControlInterface(ROBOT_IP)
#             self.rtde_r = RTDEReceiveInterface(ROBOT_IP)
#             print("✅ Robot connected successfully!\n")
#         except Exception as e:
#             print(f"⚠️ Warning: Could not connect to robot: {e}")

#         self._setup_physics_and_collision()
#         self.T_base_device0 = None 

#     def _setup_physics_and_collision(self):
#         print("🛡️ Initializing Physics and Collision Checkers...")
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         self.urdf_path = os.path.join(current_dir, "ur3_gripper.urdf") 
#         self.srdf_path = os.path.join(current_dir, "ur3_gripper.srdf")
        
#         p.connect(p.GUI) 
#         p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         p.setGravity(0, 0, -9.81)
#         p.resetDebugVisualizerCamera(1.2, 90, -30, [0, 0, 0])
#         p.loadURDF("plane.urdf")    

#         # Obstacles
#         wall_pos_r, wall_quat_r, wall_pos_w, wall_quat_w = get_object_transforms(WALL_POS_WORLD, WALL_ROT_Z_DEG)
#         table_pos_r, table_quat_r, table_pos_w, table_quat_w = get_object_transforms(TABLE_POS_WORLD, TABLE_ROT_Z_DEG)

#         p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE]), 
#                           basePosition=wall_pos_w, baseOrientation=wall_quat_w,
#                           baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE], rgbaColor=[0.8, 0.2, 0.2, 0.8]))
#         p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE]), 
#                           basePosition=table_pos_w, baseOrientation=table_quat_w,
#                           baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE], rgbaColor=[0.5, 0.3, 0.1, 1.0]))

#         # --- MAIN ROBOT ---
#         sim_robot_orn = p.getQuaternionFromEuler([0, 0, SIM_ROBOT_WORLD_ROT_Z])
#         self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0], baseOrientation=sim_robot_orn, useFixedBase=1)
        
#         # --- GHOST ROBOT ---
#         self.ghost_id = p.loadURDF(self.urdf_path, [0, 0, 0], baseOrientation=sim_robot_orn, useFixedBase=1)
        
#         num_joints = p.getNumJoints(self.ghost_id)
#         for i in range(num_joints):
#             p.changeVisualShape(self.ghost_id, i, rgbaColor=[0, 1, 0, 0.5])
#             p.setCollisionFilterGroupMask(self.ghost_id, i, collisionFilterGroup=0, collisionFilterMask=0)

#         joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
#         name_to_idx = {p.getJointInfo(self.robot_id, i)[1].decode('utf-8'): i for i in range(p.getNumJoints(self.robot_id))}
#         self.active_indices = [name_to_idx.get(n, -1) for n in joint_names]

#         self.ee_link_idx = -1
#         for i in range(p.getNumJoints(self.robot_id)):
#             if p.getJointInfo(self.robot_id, i)[12].decode("utf-8") == "tool0": 
#                 self.ee_link_idx = i; break

#         # Initialize planner
#         self.planner = mplib.Planner(urdf=self.urdf_path, srdf=self.srdf_path, move_group="tool0")
#         combined_pts = np.vstack([
#             get_box_point_cloud(wall_pos_r, WALL_SIZE, wall_quat_r, resolution=0.04), 
#             get_box_point_cloud(table_pos_r, TABLE_SIZE, table_quat_r, resolution=0.04)
#         ])
#         self.planner.update_point_cloud(combined_pts)
#         print("✅ Environment setup complete (Ghost Robot Loaded).")

#     def _load_stats(self):
#         dataset_dir = self.task_config['dataset_dir']
#         stats = get_norm_stats(dataset_dir)
#         self.tx_mean = torch.tensor([stats['tx']['mean']], device=self.device, dtype=torch.float32)
#         self.tx_std  = torch.tensor([stats['tx']['std']],  device=self.device, dtype=torch.float32)
#         self.ty_mean = torch.tensor([stats['ty']['mean']], device=self.device, dtype=torch.float32)
#         self.ty_std  = torch.tensor([stats['ty']['std']],  device=self.device, dtype=torch.float32)
#         self.tz_mean = torch.tensor([stats['tz']['mean']], device=self.device, dtype=torch.float32)
#         self.tz_std  = torch.tensor([stats['tz']['std']],  device=self.device, dtype=torch.float32)
        
#         self.qpos_mean = torch.zeros(10, device=self.device, dtype=torch.float32)
#         self.qpos_std  = torch.ones(10,  device=self.device, dtype=torch.float32)
#         self.qpos_mean[0] = self.tx_mean
#         self.qpos_mean[1] = self.ty_mean
#         self.qpos_mean[2] = self.tz_mean
#         self.qpos_std[0]  = self.tx_std
#         self.qpos_std[1]  = self.ty_std
#         self.qpos_std[2]  = self.tz_std

#     def _load_model(self):
#         self.policy = make_policy(self.policy_config['policy_class'], self.policy_config)
#         state_dict = torch.load(self.checkpoint_path, map_location=self.device)
#         self.policy.load_state_dict(state_dict)
#         self.policy.to(self.device)
#         self.policy.eval()
#         print("✅ Model Loaded.")

#     def _get_transform(self):
#         return transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def _normalize_qpos(self, qpos_raw):
#         qpos_tensor = torch.tensor(qpos_raw, dtype=torch.float32, device=self.device).unsqueeze(0) 
#         return (qpos_tensor - self.qpos_mean) / self.qpos_std

#     def _denormalize_pos_numpy(self, norm_pos_array):
#         tx_m, tx_s = self.tx_mean.cpu().numpy(), self.tx_std.cpu().numpy()
#         ty_m, ty_s = self.ty_mean.cpu().numpy(), self.ty_std.cpu().numpy()
#         tz_m, tz_s = self.tz_mean.cpu().numpy(), self.tz_std.cpu().numpy()
#         x = norm_pos_array[..., 0] * tx_s + tx_m
#         y = norm_pos_array[..., 1] * ty_s + ty_m
#         z = norm_pos_array[..., 2] * tz_s + tz_m
#         return np.stack([x, y, z], axis=-1)

#     # -------------------------------------------------------------------------
#     # COORDINATE TRANSFORMS
#     # -------------------------------------------------------------------------
#     def compute_T_base_device0(self, current_pose_vec):
#         T_base_tcp = np.eye(4)
#         T_base_tcp[:3, :3] = rotvec_to_matrix(*current_pose_vec[3:])
#         T_base_tcp[:3, 3] = current_pose_vec[:3]
#         self.T_base_device0 = T_base_tcp @ T_TCP_DEVICE
#         print("✅ Initial Reference Frame Set (T_base_device0).")

#     def get_current_pose_relative(self):
#         if self.rtde_r is None: 
#             return [0,0,0, 1,0,0,0,1,0, 0] 
#         pose_rtde = self.rtde_r.getActualTCPPose() 
#         T_base_tcp = np.eye(4)
#         T_base_tcp[:3, :3] = rotvec_to_matrix(*pose_rtde[3:])
#         T_base_tcp[:3, 3] = pose_rtde[:3]
#         T_base_device = T_base_tcp @ T_TCP_DEVICE
#         T_rel = np.linalg.inv(self.T_base_device0) @ T_base_device
#         pos_rel = T_rel[:3, 3].tolist()
#         rot6d_rel = mat_to_rot6d_numpy(T_rel[:3, :3]).tolist()
#         return pos_rel + rot6d_rel + [1.0]

#     def convert_relative_to_absolute(self, rel_pos, rel_rot6d):
#         r6d_tensor = torch.tensor(rel_rot6d).unsqueeze(0)
#         rot_mat_rel = rot6d_to_matrix(r6d_tensor)[0].numpy()
#         T_rel = np.eye(4)
#         T_rel[:3, :3] = rot_mat_rel
#         T_rel[:3, 3] = rel_pos
#         T_base_device_target = self.T_base_device0 @ T_rel
#         T_base_tcp_target = T_base_device_target @ T_DEVICE_TCP
#         target_pos = T_base_tcp_target[:3, 3]
#         target_quat = R.from_matrix(T_base_tcp_target[:3, :3]).as_quat() 
#         return target_pos, target_quat

#     # -------------------------------------------------------------------------
#     # EXECUTION WITH moveL & GHOST PREVIEW
#     # -------------------------------------------------------------------------
#     def execute_absolute_action(self, target_pos, target_quat):
#         # 1. Visualize the Target Coordinate Frame
#         draw_coordinate_frame(target_pos, target_quat, life_time=5.0)

#         # 2. Sync MAIN Sim Robot with Real Robot
#         if self.rtde_r is not None:
#             actual_q = self.rtde_r.getActualQ()
#             for i, angle in enumerate(actual_q):
#                 p.resetJointState(self.robot_id, self.active_indices[i], angle)
        
#         # 3. Calculate IK for the FUTURE target
#         ik_solution = p.calculateInverseKinematics(
#             self.robot_id, 
#             self.ee_link_idx, 
#             targetPosition=target_pos, 
#             targetOrientation=target_quat,
#             maxNumIterations=50,
#             residualThreshold=1e-4
#         )
#         q_proposal = list(ik_solution[:6]) 

#         # 4. Check Collisions
#         is_self_collision = self.planner.check_for_self_collision(q_proposal)
#         is_env_collision = self.planner.check_for_env_collision(q_proposal)

#         if is_self_collision or is_env_collision:
#             p.addUserDebugText("COLLISION!", [0,0,1], [1,0,0], lifeTime=1.0, textSize=2)
#             print(f"🛑 COLLISION DETECTED! Stopping. (Self: {is_self_collision}, Env: {is_env_collision})")
#             return 

#         # 5. VISUALIZE FUTURE (Ghost)
#         for i, angle in enumerate(q_proposal):
#             p.resetJointState(self.ghost_id, self.active_indices[i], angle)
#         p.stepSimulation()

#         # 6. User Confirmation
#         rot_obj = R.from_quat(target_quat)
#         rot_vec = rot_obj.as_rotvec() # [rx, ry, rz]
#         target_pose_6d = list(target_pos) + list(rot_vec)

#         print(f"\n🔮 Policy Prediction (Ghost Shown):")
#         print(f"   Target Pos: {np.around(target_pos, 3)}")
        
#         user_input = input(">> Press ENTER to Execute moveL, 's' to Skip, 'q' to Quit: ").lower()
        
#         if user_input == 'q':
#             print("Quitting...")
#             sys.exit(0)
#         elif user_input == 's':
#             print("Skipping step.")
#             return

#         # 7. Execute on Real Robot
#         if self.rtde_c is not None:
#             try:
#                 print(f"   🚀 Moving Real Robot (moveL)...")
#                 self.rtde_c.moveL(target_pose_6d, ROBOT_SPEED, ROBOT_ACCEL)
#                 print("   ✅ Done.")
#             except Exception as e:
#                 print(f"Error in moveL: {e}")
#                 print("Check if target is reachable or near a singularity.")
#         else:
#             print("   (Simulated) Real Robot Move Skipped.")
#             for i, angle in enumerate(q_proposal):
#                 p.resetJointState(self.robot_id, self.active_indices[i], angle)
#             p.stepSimulation()

#     def loop(self):
#         print("\n🚀 Starting Inference Loop (Waiting for Camera)...")
        
#         # --- LOGGING SETUP START ---
#         image_save_dir = "./inference_images"
#         csv_path = "inference_log.csv"
        
#         # Create directory
#         os.makedirs(image_save_dir, exist_ok=True)
#         print(f"📂 Saving images to: {image_save_dir}")
#         print(f"📂 Saving stats to: {csv_path}")

#         # Initialize CSV with Headers
#         with open(csv_path, 'w', newline='') as f:
#             writer = csv.writer(f)
#             # qpos has 10 dims based on _load_stats
#             header = ["step_idx"] + [f"q_{i}" for i in range(10)]
#             writer.writerow(header)
#         # --- LOGGING SETUP END ---
        
#         if self.rtde_c:
#             print(f"Moving to initial Cartesian pose: {INIT_TCP_POSE} ...")
#             try:
#                 self.rtde_c.moveL(INIT_TCP_POSE, ROBOT_SPEED, ROBOT_ACCEL)
#             except Exception as e:
#                 print(f"⚠️ Failed to move to INIT_TCP_POSE: {e}")
#                 return

#             time.sleep(1.0)
            
#             # Sync PyBullet
#             actual_q = self.rtde_r.getActualQ()
#             for i, angle in enumerate(actual_q):
#                 p.resetJointState(self.robot_id, self.active_indices[i], angle)
#             p.stepSimulation()
            
#             # Set Reference Frame
#             init_pose_vec = self.rtde_r.getActualTCPPose()
#             self.compute_T_base_device0(init_pose_vec)
#         else:
#             print("⚠️ Sim-only mode: Setting Identity Reference Frame.")
#             self.T_base_device0 = np.eye(4) 
#             t_pos = INIT_TCP_POSE[:3]
#             t_rot = rotvec_to_matrix(*INIT_TCP_POSE[3:])
#             t_quat = R.from_matrix(t_rot).as_quat()
#             ik_sol = p.calculateInverseKinematics(self.robot_id, self.ee_link_idx, t_pos, t_quat)
#             for i, angle in enumerate(ik_sol[:6]):
#                 p.resetJointState(self.robot_id, self.active_indices[i], angle)

#         step_idx = 0
#         while True:
#             obs_pil = self.camera.get_latest_frame()
#             if obs_pil is None: 
#                 time.sleep(0.01)
#                 continue

#             # --- LOGGING IMAGE ---
#             img_filename = f"{step_idx:04d}.png"
#             obs_pil.save(os.path.join(image_save_dir, img_filename))
#             # ---------------------

#             qpos_rel_raw = self.get_current_pose_relative()

#             obs_tensor = self.transform(obs_pil).unsqueeze(0).unsqueeze(0).to(self.device)
#             qpos_norm = self._normalize_qpos(qpos_rel_raw)

#             # --- LOGGING CSV ---
#             # Move tensor to CPU and convert to list
#             q_values = qpos_norm.cpu().squeeze().tolist()
#             with open(csv_path, 'a', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow([step_idx] + q_values)
#             # -------------------
            
#             with torch.inference_mode():
#                 action_pred = self.policy(qpos_norm, obs_tensor) 
            
#             action_chunk = action_pred.squeeze(0).squeeze(0).cpu().numpy()
#             pred_pos_rel = self._denormalize_pos_numpy(action_chunk[..., :3])[0]
#             pred_rot6d_rel = action_chunk[..., 3:9][0]
            
#             target_pos_abs, target_quat_abs = self.convert_relative_to_absolute(pred_pos_rel, pred_rot6d_rel)
#             self.execute_absolute_action(target_pos_abs, target_quat_abs)
            
#             step_idx += 1

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--checkpoint', type=str, default='/home/pt/github/ws_ros2humble-main_lab/ACT/checkpoints/policy_epoch_60_seed_42.ckpt')
#     parser.add_argument("--port", type=str, default="5555")
#     args = parser.parse_args()
    
#     agent = ManualInferenceACT(args.checkpoint, zmq_port=args.port)
#     agent.loop()

# if __name__ == '__main__':
#     main()

##

# --- START OF FILE 1_2_robot.py ---

# --- Environment Constants ---
import torch
import numpy as np
import argparse
import sys
import os
import time
import traceback
import zmq
import pickle
import csv
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as transforms

# --- Physics & Planning Imports ---
import mplib
import pybullet as p
import pybullet_data

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
ROBOT_IP = '10.0.0.1' 

# --- MACRO VARIABLES FOR SPEED CONTROL ---
ROBOT_SPEED = 0.08      
ROBOT_ACCEL = 0.05      

# --- Environment Constants ---

# !!! Initial TCP Pose (Base Frame) [x, y, z, rx, ry, rz] !!!
INIT_TCP_POSE = [0.299, 0.146, 0.434, 3.932, -1.554, 1.595]

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

# -----------------------------------------------------------------------------
# 3. Remote Camera Helper
# -----------------------------------------------------------------------------
class RemoteCameraClient:
    def __init__(self, port="5555"):
        print(f"📡 Connecting to Camera Stream on localhost:{port}...")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1) 
        print("✅ Camera Client Connected.")

    def get_latest_frame(self):
        # 1. Flush the buffer (remove stale images sitting in the queue)
        try:
            while True:
                self.socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            pass 

        # 2. Blocking wait for a NEW frame
        # This ensures we get a frame that arrived *after* we called this function.
        try:
            data = self.socket.recv() # Blocking call
            return pickle.loads(data)
        except Exception as e:
            print(f"Error receiving image: {e}")
            return None

# -----------------------------------------------------------------------------
# 4. Main Inference Class
# -----------------------------------------------------------------------------
class ManualInferenceACT:
    def __init__(self, checkpoint_path, device=None, zmq_port="5555"):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.checkpoint_path = checkpoint_path
        
        self.policy_config = POLICY_CONFIG
        self.task_config = TASK_CONFIG
        self._load_stats()
        self._load_model()
        self.transform = self._get_transform()
        
        self.camera = RemoteCameraClient(port=zmq_port)

        self.rtde_c = None
        self.rtde_r = None
        try:
            print(f"\n🔗 Connecting to robot at {ROBOT_IP}...")
            self.rtde_c = RTDEControlInterface(ROBOT_IP)
            self.rtde_r = RTDEReceiveInterface(ROBOT_IP)
            print("✅ Robot connected successfully!\n")
        except Exception as e:
            print(f"⚠️ Warning: Could not connect to robot: {e}")

        self._setup_physics_and_collision()
        self.T_base_device0 = None 

    def _setup_physics_and_collision(self):
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

        # --- MAIN ROBOT ---
        sim_robot_orn = p.getQuaternionFromEuler([0, 0, SIM_ROBOT_WORLD_ROT_Z])
        self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0], baseOrientation=sim_robot_orn, useFixedBase=1)
        
        # --- GHOST ROBOT ---
        self.ghost_id = p.loadURDF(self.urdf_path, [0, 0, 0], baseOrientation=sim_robot_orn, useFixedBase=1)
        
        num_joints = p.getNumJoints(self.ghost_id)
        for i in range(num_joints):
            p.changeVisualShape(self.ghost_id, i, rgbaColor=[0, 1, 0, 0.5])
            p.setCollisionFilterGroupMask(self.ghost_id, i, collisionFilterGroup=0, collisionFilterMask=0)

        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        name_to_idx = {p.getJointInfo(self.robot_id, i)[1].decode('utf-8'): i for i in range(p.getNumJoints(self.robot_id))}
        self.active_indices = [name_to_idx.get(n, -1) for n in joint_names]

        self.ee_link_idx = -1
        for i in range(p.getNumJoints(self.robot_id)):
            if p.getJointInfo(self.robot_id, i)[12].decode("utf-8") == "tool0": 
                self.ee_link_idx = i; break

        # Initialize planner
        self.planner = mplib.Planner(urdf=self.urdf_path, srdf=self.srdf_path, move_group="tool0")
        combined_pts = np.vstack([
            get_box_point_cloud(wall_pos_r, WALL_SIZE, wall_quat_r, resolution=0.04), 
            get_box_point_cloud(table_pos_r, TABLE_SIZE, table_quat_r, resolution=0.04)
        ])
        self.planner.update_point_cloud(combined_pts)
        print("✅ Environment setup complete (Ghost Robot Loaded).")

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
        if self.rtde_r is None: 
            return [0,0,0, 1,0,0,0,1,0, 0] 
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
    # EXECUTION WITH moveL & GHOST PREVIEW
    # -------------------------------------------------------------------------
    def execute_absolute_action(self, target_pos, target_quat):
        # 1. Visualize the Target Coordinate Frame
        draw_coordinate_frame(target_pos, target_quat, life_time=5.0)

        # 2. Sync MAIN Sim Robot with Real Robot
        if self.rtde_r is not None:
            actual_q = self.rtde_r.getActualQ()
            for i, angle in enumerate(actual_q):
                p.resetJointState(self.robot_id, self.active_indices[i], angle)
        
        # 3. Calculate IK for the FUTURE target
        ik_solution = p.calculateInverseKinematics(
            self.robot_id, 
            self.ee_link_idx, 
            targetPosition=target_pos, 
            targetOrientation=target_quat,
            maxNumIterations=50,
            residualThreshold=1e-4
        )
        q_proposal = list(ik_solution[:6]) 

        # 4. Check Collisions
        is_self_collision = self.planner.check_for_self_collision(q_proposal)
        is_env_collision = self.planner.check_for_env_collision(q_proposal)

        if is_self_collision or is_env_collision:
            p.addUserDebugText("COLLISION!", [0,0,1], [1,0,0], lifeTime=1.0, textSize=2)
            print(f"🛑 COLLISION DETECTED! Stopping. (Self: {is_self_collision}, Env: {is_env_collision})")
            return 

        # 5. VISUALIZE FUTURE (Ghost)
        for i, angle in enumerate(q_proposal):
            p.resetJointState(self.ghost_id, self.active_indices[i], angle)
        p.stepSimulation()

        # 6. User Confirmation
        rot_obj = R.from_quat(target_quat)
        rot_vec = rot_obj.as_rotvec() # [rx, ry, rz]
        target_pose_6d = list(target_pos) + list(rot_vec)

        print(f"\n🔮 Policy Prediction (Ghost Shown):")
        print(f"   Target Pos: {np.around(target_pos, 3)}")
        
        user_input = input(">> Press ENTER to Execute moveL, 's' to Skip, 'q' to Quit: ").lower()
        
        if user_input == 'q':
            print("Quitting...")
            sys.exit(0)
        elif user_input == 's':
            print("Skipping step.")
            return

        # 7. Execute on Real Robot
        if self.rtde_c is not None:
            try:
                print(f"   🚀 Moving Real Robot (moveL)...")
                self.rtde_c.moveL(target_pose_6d, ROBOT_SPEED, ROBOT_ACCEL)
                print("   ✅ Done.")
                # Important: Wait a little bit for the camera to capture the static scene
                time.sleep(0.5) 
            except Exception as e:
                print(f"Error in moveL: {e}")
                print("Check if target is reachable or near a singularity.")
        else:
            print("   (Simulated) Real Robot Move Skipped.")
            for i, angle in enumerate(q_proposal):
                p.resetJointState(self.robot_id, self.active_indices[i], angle)
            p.stepSimulation()
            time.sleep(0.1)

    def loop(self):
        print("\n🚀 Starting Inference Loop (Waiting for Camera)...")
        
        # --- LOGGING SETUP START ---
        image_save_dir = "./inference_images"
        csv_path = "inference_log.csv"
        
        # Create directory
        os.makedirs(image_save_dir, exist_ok=True)
        print(f"📂 Saving images to: {image_save_dir}")
        print(f"📂 Saving stats to: {csv_path}")

        # Initialize CSV with Headers
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # qpos has 10 dims based on _load_stats
            header = ["step_idx"] + [f"q_{i}" for i in range(10)]
            writer.writerow(header)
        # --- LOGGING SETUP END ---
        
        if self.rtde_c:
            print(f"Moving to initial Cartesian pose: {INIT_TCP_POSE} ...")
            try:
                self.rtde_c.moveL(INIT_TCP_POSE, ROBOT_SPEED, ROBOT_ACCEL)
            except Exception as e:
                print(f"⚠️ Failed to move to INIT_TCP_POSE: {e}")
                return

            time.sleep(1.0)
            
            # Sync PyBullet
            actual_q = self.rtde_r.getActualQ()
            for i, angle in enumerate(actual_q):
                p.resetJointState(self.robot_id, self.active_indices[i], angle)
            p.stepSimulation()
            
            # Set Reference Frame
            init_pose_vec = self.rtde_r.getActualTCPPose()
            self.compute_T_base_device0(init_pose_vec)
        else:
            print("⚠️ Sim-only mode: Setting Identity Reference Frame.")
            self.T_base_device0 = np.eye(4) 
            t_pos = INIT_TCP_POSE[:3]
            t_rot = rotvec_to_matrix(*INIT_TCP_POSE[3:])
            t_quat = R.from_matrix(t_rot).as_quat()
            ik_sol = p.calculateInverseKinematics(self.robot_id, self.ee_link_idx, t_pos, t_quat)
            for i, angle in enumerate(ik_sol[:6]):
                p.resetJointState(self.robot_id, self.active_indices[i], angle)

        step_idx = 0
        while True:
            # Flushes buffer and waits for a fresh frame to arrive
            obs_pil = self.camera.get_latest_frame()
            if obs_pil is None: 
                print("Wait for camera...")
                time.sleep(0.5)
                continue

            # --- LOGGING IMAGE ---
            img_filename = f"{step_idx:04d}.png"
            obs_pil.save(os.path.join(image_save_dir, img_filename))
            # ---------------------

            qpos_rel_raw = self.get_current_pose_relative()

            obs_tensor = self.transform(obs_pil).unsqueeze(0).unsqueeze(0).to(self.device)
            qpos_norm = self._normalize_qpos(qpos_rel_raw)

            # --- LOGGING CSV ---
            # Move tensor to CPU and convert to list
            q_values = qpos_norm.cpu().squeeze().tolist()
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step_idx] + q_values)
            # -------------------
            
            with torch.inference_mode():
                action_pred = self.policy(qpos_norm, obs_tensor) 
            
            action_chunk = action_pred.squeeze(0).squeeze(0).cpu().numpy()
            pred_pos_rel = self._denormalize_pos_numpy(action_chunk[..., :3])[0]
            pred_rot6d_rel = action_chunk[..., 3:9][0]
            
            target_pos_abs, target_quat_abs = self.convert_relative_to_absolute(pred_pos_rel, pred_rot6d_rel)
            self.execute_absolute_action(target_pos_abs, target_quat_abs)
            
            step_idx += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/home/pt/github/ws_ros2humble-main_lab/ACT/checkpoints/policy_epoch_60_seed_42.ckpt')
    parser.add_argument("--port", type=str, default="5555")
    args = parser.parse_args()
    
    agent = ManualInferenceACT(args.checkpoint, zmq_port=args.port)
    agent.loop()

if __name__ == '__main__':
    main()