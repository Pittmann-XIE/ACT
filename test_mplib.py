# import mplib
# import pybullet as p
# import pybullet_data
# import numpy as np
# import time
# import os
# from scipy.spatial.transform import Rotation as R

# # ---------------------------------------------------------
# # HELPER: Generate SRDF
# # ---------------------------------------------------------
# def generate_srdf(srdf_path):
#     srdf_content = """<?xml version="1.0" ?>
# <robot name="ur3_mplib">
#     <group name="manipulator">
#         <chain base_link="base_link" tip_link="tool0"/>
#     </group>
#     <disable_collisions link1="base_link_inertia" link2="shoulder_link" reason="Adjacent" />
#     <disable_collisions link1="base_link" link2="base_link_inertia" reason="Adjacent" />
#     <disable_collisions link1="shoulder_link" link2="upper_arm_link" reason="Adjacent" />
#     <disable_collisions link1="upper_arm_link" link2="forearm_link" reason="Adjacent" />
#     <disable_collisions link1="forearm_link" link2="wrist_1_link" reason="Adjacent" />
#     <disable_collisions link1="wrist_1_link" link2="wrist_2_link" reason="Adjacent" />
#     <disable_collisions link1="wrist_2_link" link2="wrist_3_link" reason="Adjacent" />
#     <disable_collisions link1="wrist_3_link" link2="tool0" reason="Adjacent" />
# </robot>
# """
#     with open(srdf_path, "w") as f:
#         f.write(srdf_content)
#     print(f"✅ Generated SRDF file at: {srdf_path}")

# # ---------------------------------------------------------
# # HELPER: Map Joint Names
# # ---------------------------------------------------------
# def get_pybullet_joint_indices(robot_id, joint_names):
#     joint_indices = []
#     num_joints = p.getNumJoints(robot_id)
#     name_to_index = {}
#     for i in range(num_joints):
#         info = p.getJointInfo(robot_id, i)
#         name = info[1].decode("utf-8")
#         name_to_index[name] = i

#     for name in joint_names:
#         if name in name_to_index:
#             joint_indices.append(name_to_index[name])
#         else:
#             joint_indices.append(-1) 
#     return joint_indices

# def get_link_index(robot_id, link_name):
#     num_joints = p.getNumJoints(robot_id)
#     for i in range(num_joints):
#         info = p.getJointInfo(robot_id, i)
#         if info[12].decode("utf-8") == link_name:
#             return i
#     return -1

# # ---------------------------------------------------------
# # HELPER: Generate Point Cloud for Obstacles
# # ---------------------------------------------------------
# def get_box_point_cloud(center, size, resolution=0.02):
#     x = np.arange(center[0] - size[0]/2, center[0] + size[0]/2, resolution)
#     y = np.arange(center[1] - size[1]/2, center[1] + size[1]/2, resolution)
#     z = np.arange(center[2] - size[2]/2, center[2] + size[2]/2, resolution)
#     xx, yy, zz = np.meshgrid(x, y, z)
#     points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
#     return points

# # ---------------------------------------------------------
# # HELPER: Create & Attach "U-Shape" Gripper
# # ---------------------------------------------------------
# def attach_procedural_gripper(robot_id, ee_link_name, planner):
#     """
#     Creates a lightweight U-shaped gripper using a Compound Shape.
#     Dimensions: 140mm (Thick/X) x 80mm (Width/Y) x 160mm (Length/Z)
#     """
    
#     # --- 1. Define Geometry Parts ---
#     # We will build the gripper out of 3 boxes: 1 Base + 2 Fingers
    
#     # Total Dimensions
#     total_len = 0.16
#     total_thick = 0.14
#     total_width = 0.08
    
#     # Part A: Base Plate (Connects to robot)
#     # Takes up the first 4cm of length
#     base_len = 0.04 
#     base_size = [total_thick, total_width, base_len]
#     base_pos  = [0, 0, base_len/2] # Centered in Z for this part
    
#     # Part B: Finger 1 (Left)
#     finger_len = total_len - base_len
#     finger_thick = 0.02 # 2cm thick fingers
#     finger_size = [finger_thick, total_width, finger_len]
#     # Position: Shifted X to the left edge, Shifted Z to be after base
#     finger1_pos = [-(total_thick/2 - finger_thick/2), 0, base_len + finger_len/2]

#     # Part C: Finger 2 (Right)
#     finger2_pos = [(total_thick/2 - finger_thick/2), 0, base_len + finger_len/2]

#     # Arrays for Compound Shape
#     halfExtents = [
#         [s/2 for s in base_size], 
#         [s/2 for s in finger_size], 
#         [s/2 for s in finger_size]
#     ]
#     positions = [base_pos, finger1_pos, finger2_pos]
    
#     # --- 2. Create PyBullet Visual/Collision ---
#     # We make it VERY light (0.01kg) so it doesn't drag the arm down
    
#     vis_shapes = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX]*3, 
#                                           halfExtents=halfExtents, 
#                                           visualFramePositions=positions,
#                                           rgbaColors=[[0.2,0.2,0.2,1]]*3)
                                          
#     col_shapes = p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX]*3, 
#                                              halfExtents=halfExtents, 
#                                              collisionFramePositions=positions)

#     gripper_id = p.createMultiBody(baseMass=0.01,  # <--- VERY LIGHT
#                                    baseCollisionShapeIndex=col_shapes, 
#                                    baseVisualShapeIndex=vis_shapes, 
#                                    basePosition=[0,0,0])

#     # --- 3. Attach to Robot ---
#     ee_idx = get_link_index(robot_id, ee_link_name)
#     if ee_idx == -1: return

#     # Fixed constraint at the end of the tool link
#     # p.createConstraint(robot_id, ee_idx, gripper_id, -1, p.JOINT_FIXED, 
#     #                    jointAxis=[0, 0, 0], parentFramePosition=[0,0,0], childFramePosition=[0, 0, 0])
#     # 1. Attach the gripper with a Fixed Joint
#     p.createConstraint(robot_id, ee_idx, gripper_id, -1, p.JOINT_FIXED, 
#                     jointAxis=[0, 0, 0], parentFramePosition=[0,0,0], childFramePosition=[0, 0, 0])

#     # 2. Disable collision ONLY between Gripper and the Link it attaches to (tool0)
#     # This stops the shaking.
#     p.setCollisionFilterPair(bodyUniqueIdA=robot_id, bodyUniqueIdB=gripper_id, 
#                             linkIndexA=ee_idx, linkIndexB=-1, 
#                             enableCollision=0)

#     # 3. (Optional but recommended) Disable collision with the immediate parent (Wrist 3)
#     # because the gripper base often clips slightly into the wrist mesh too.
#     p.setCollisionFilterPair(bodyUniqueIdA=robot_id, bodyUniqueIdB=gripper_id, 
#                             linkIndexA=ee_idx-1, linkIndexB=-1, 
#                             enableCollision=0)
#     print(f"🔧 Attached U-Shaped Gripper (Mass: 0.01kg)")

#     # --- 4. Update Mplib Planner ---
#     # For collision planning, we can simply approximate the gripper as one big box
#     # OR add the 3 specific boxes. For simplicity and safety, let's add the full bounding box.
#     # This ensures the planner is conservative and won't scrape the "empty space" between fingers against a wall.
    
#     full_box_size = [total_thick, total_width, total_len]
#     center_offset = [0, 0, total_len/2]
    
#     tool_pose = mplib.Pose(p=center_offset, q=[1, 0, 0, 0])
#     planner.update_attached_box(size=full_box_size, pose=tool_pose, link_id=-1)
#     print("✅ Added Gripper collision box to Mplib planner.")

# # ---------------------------------------------------------
# # HELPER: GUI Sliders
# # ---------------------------------------------------------
# def setup_debug_sliders():
#     p.addUserDebugText("Left Click to Rotate Camera", [0, 0, 0.7], [0, 0, 0], textSize=1.2)
#     sx = p.addUserDebugParameter("Target X", -0.6, 0.6, 0.3)
#     sy = p.addUserDebugParameter("Target Y", -0.6, 0.6, 0.1)
#     sz = p.addUserDebugParameter("Target Z", 0.0, 0.8, 0.4) 
#     sqx = p.addUserDebugParameter("Target Qx", -1.0, 1.0, 1.0)
#     sqy = p.addUserDebugParameter("Target Qy", -1.0, 1.0, 0.0)
#     sqz = p.addUserDebugParameter("Target Qz", -1.0, 1.0, 0.0)
#     sqw = p.addUserDebugParameter("Target Qw", -1.0, 1.0, 0.0)
#     btn = p.addUserDebugParameter(">>> PLAN & MOVE <<<", 1, 0, 0)
#     return [sx, sy, sz, sqx, sqy, sqz, sqw, btn]

# def main():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     urdf_path = os.path.join(current_dir, "ur3_mesh_.urdf") 
#     srdf_path = os.path.join(current_dir, "ur3_mplib.srdf")
    
#     if not os.path.exists(urdf_path):
#         print(f"❌ CRITICAL ERROR: Could not find {urdf_path}")
#         return

#     generate_srdf(srdf_path)

#     print("🔮 Initializing PyBullet Visualization...")
#     p.connect(p.GUI)
#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     p.setGravity(0, 0, -9.81)
#     p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
#     p.resetDebugVisualizerCamera(1.5, 45, -30, [0, 0, 0])

#     p.loadURDF("plane.urdf")
    
#     # --- OBSTACLES ---
#     table_pos  = [0, 0, -0.05]
#     table_size = [1.0, 1.0, 0.1]
#     t_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in table_size]), 
#                                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in table_size], rgbaColor=[0.6, 0.4, 0.2, 1]), 
#                                basePosition=table_pos)
    
#     wall_pos  = [0, -0.225, 0.5]
#     wall_size = [1.0, 0.05, 1.0]
#     w_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in wall_size]), 
#                                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in wall_size], rgbaColor=[0.8, 0.2, 0.2, 0.8]), 
#                                basePosition=wall_pos)

#     robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=1)
    
#     # --- MPLIB SETUP ---
#     planner = mplib.Planner(urdf=urdf_path, srdf=srdf_path, move_group="tool0")
#     pts_table = get_box_point_cloud(table_pos, table_size)
#     pts_wall  = get_box_point_cloud(wall_pos, wall_size)
#     planner.update_point_cloud(np.vstack([pts_table, pts_wall]))
    
#     # --- ATTACH LIGHTWEIGHT GRIPPER ---
#     attach_procedural_gripper(robot_id, "tool0", planner)

#     mplib_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
#                          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
#     pb_joint_indices = get_pybullet_joint_indices(robot_id, mplib_joint_names)
    
#     # Filter out invalid indices
#     active_joint_indices = [i for i in pb_joint_indices if i >= 0]

#     # Home Pose
#     home_qpos = [0, -1.57, 0, -1.57, 0, 0]
#     for i, angle in enumerate(home_qpos):
#         idx = pb_joint_indices[i]
#         if idx >= 0: 
#             p.resetJointState(robot_id, idx, angle)

#     # ---------------------------------------------------------
#     # MAIN LOOP
#     # ---------------------------------------------------------
#     sliders = setup_debug_sliders()
#     [sid_x, sid_y, sid_z, sid_qx, sid_qy, sid_qz, sid_qw, sid_btn] = sliders
#     old_btn_val = p.readUserDebugParameter(sid_btn)
    
#     last_mouse_pos = None
#     is_rotating = False
    
#     # Variable to track where we want the robot to stay
#     target_joint_positions = home_qpos 

#     print("\n✅ System Ready!")

#     while True:
#         # --- PHYSICS HOLD LOOP (Fixes "Falling Down") ---
#         # PyBullet joints have 0 friction by default. 
#         # We must actively command them to stay at 'target_joint_positions'.
#         p.setJointMotorControlArray(
#             bodyUniqueId=robot_id,
#             jointIndices=active_joint_indices,
#             controlMode=p.POSITION_CONTROL,
#             targetPositions=target_joint_positions,
#             forces=[100.0] * len(active_joint_indices) # Sufficient force to hold
#         )
        
#         p.stepSimulation()

#         # Camera Logic
#         events = p.getMouseEvents()
#         for e in events:
#             if e[0] == 2 and e[3] == 0: 
#                 if e[4] == 1: is_rotating, last_mouse_pos = True, (e[1], e[2])
#                 else: is_rotating, last_mouse_pos = False, None
#             if e[0] == 1 and is_rotating and last_mouse_pos:
#                 cam_info = p.getDebugVisualizerCamera()
#                 dx, dy = e[1] - last_mouse_pos[0], e[2] - last_mouse_pos[1]
#                 p.resetDebugVisualizerCamera(cam_info[10], cam_info[8] - dx*0.3, cam_info[9] - dy*0.3, cam_info[11])
#                 last_mouse_pos = (e[1], e[2])

#         # Planner Logic
#         btn_val = p.readUserDebugParameter(sid_btn)
#         if btn_val > old_btn_val:
#             old_btn_val = btn_val
            
#             tx, ty, tz = [p.readUserDebugParameter(s) for s in [sid_x, sid_y, sid_z]]
#             q_raw = np.array([p.readUserDebugParameter(s) for s in [sid_qx, sid_qy, sid_qz, sid_qw]])
#             norm = np.linalg.norm(q_raw)
#             q_norm = q_raw / norm if norm > 1e-6 else np.array([0.0, 0.0, 0.0, 1.0])
            
#             target_pose = mplib.Pose(p=[tx, ty, tz], q=[q_norm[3], q_norm[0], q_norm[1], q_norm[2]])

#             # Read current state directly from robot
#             current_qpos = [p.getJointState(robot_id, idx)[0] for idx in active_joint_indices]

#             print(f"🤔 Planning to: [{tx:.2f}, {ty:.2f}, {tz:.2f}]...")
#             result = planner.plan_pose(target_pose, current_qpos, time_step=0.01)
            
#             if result["status"] == "Success":
#                 print("🚀 Path Found!")
#                 for waypoint in result["position"]:
#                     # 1. Move visually/physically
#                     # We set the "target" for the physics engine to follow
#                     target_joint_positions = waypoint
                    
#                     # 2. Teleporting (resetJointState) is smoother for visualization 
#                     # but since we added MotorControl, we can just update the target.
#                     # However, to keep it snappy like MPLIB usually is, we can do both:
#                     for i, angle in enumerate(waypoint):
#                         idx = pb_joint_indices[i]
#                         if idx >= 0: p.resetJointState(robot_id, idx, angle)
                        
#                     p.stepSimulation()
#                     time.sleep(0.02)
                
#                 # Update the holding target to the end of the path
#                 target_joint_positions = result["position"][-1]
                
#             else:
#                 print(f"❌ Planning Failed: {result['status']}")
                
#         time.sleep(0.01)

# if __name__ == "__main__":
#     main()


# ##
# import mplib
# import pybullet as p
# import pybullet_data
# import numpy as np
# import time
# import os
# from scipy.spatial.transform import Rotation as R

# # --- UR RTDE IMPORTS ---
# try:
#     import rtde_control
#     import rtde_receive
# except ImportError:
#     print("❌ ERROR: ur_rtde library not found. Please install: pip install ur_rtde")
#     exit()

# # ---------------------------------------------------------
# # CONFIGURATION
# # ---------------------------------------------------------
# ROBOT_IP = "192.168.1.23"
# # Initial joints in DEGREES as requested
# # [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
# INIT_JOINTS_DEG = [49.5, -99.57, -6.18, -71.6, -266.31, 358.6]

# # ---------------------------------------------------------
# # HELPER: Generate SRDF
# # ---------------------------------------------------------
# def generate_srdf(srdf_path):
#     srdf_content = """<?xml version="1.0" ?>
# <robot name="ur3_mplib">
#     <group name="manipulator">
#         <chain base_link="base_link" tip_link="tool0"/>
#     </group>
#     <disable_collisions link1="base_link_inertia" link2="shoulder_link" reason="Adjacent" />
#     <disable_collisions link1="base_link" link2="base_link_inertia" reason="Adjacent" />
#     <disable_collisions link1="shoulder_link" link2="upper_arm_link" reason="Adjacent" />
#     <disable_collisions link1="upper_arm_link" link2="forearm_link" reason="Adjacent" />
#     <disable_collisions link1="forearm_link" link2="wrist_1_link" reason="Adjacent" />
#     <disable_collisions link1="wrist_1_link" link2="wrist_2_link" reason="Adjacent" />
#     <disable_collisions link1="wrist_2_link" link2="wrist_3_link" reason="Adjacent" />
#     <disable_collisions link1="wrist_3_link" link2="tool0" reason="Adjacent" />
# </robot>
# """
#     with open(srdf_path, "w") as f:
#         f.write(srdf_content)

# # ---------------------------------------------------------
# # HELPER: Map Joint Names
# # ---------------------------------------------------------
# def get_pybullet_joint_indices(robot_id, joint_names):
#     joint_indices = []
#     num_joints = p.getNumJoints(robot_id)
#     name_to_index = {}
#     for i in range(num_joints):
#         info = p.getJointInfo(robot_id, i)
#         name = info[1].decode("utf-8")
#         name_to_index[name] = i

#     for name in joint_names:
#         if name in name_to_index:
#             joint_indices.append(name_to_index[name])
#         else:
#             joint_indices.append(-1) 
#     return joint_indices

# def get_link_index(robot_id, link_name):
#     num_joints = p.getNumJoints(robot_id)
#     for i in range(num_joints):
#         info = p.getJointInfo(robot_id, i)
#         if info[12].decode("utf-8") == link_name:
#             return i
#     return -1

# # ---------------------------------------------------------
# # HELPER: Environment & Gripper
# # ---------------------------------------------------------
# def get_box_point_cloud(center, size, resolution=0.02):
#     x = np.arange(center[0] - size[0]/2, center[0] + size[0]/2, resolution)
#     y = np.arange(center[1] - size[1]/2, center[1] + size[1]/2, resolution)
#     z = np.arange(center[2] - size[2]/2, center[2] + size[2]/2, resolution)
#     xx, yy, zz = np.meshgrid(x, y, z)
#     points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
#     return points

# def attach_procedural_gripper(robot_id, ee_link_name, planner):
#     total_len = 0.16
#     total_thick = 0.14
#     total_width = 0.08
#     base_len = 0.04 
#     base_size = [total_thick, total_width, base_len]
#     base_pos  = [0, 0, base_len/2] 
    
#     finger_len = total_len - base_len
#     finger_thick = 0.02 
#     finger_size = [finger_thick, total_width, finger_len]
#     finger1_pos = [-(total_thick/2 - finger_thick/2), 0, base_len + finger_len/2]
#     finger2_pos = [(total_thick/2 - finger_thick/2), 0, base_len + finger_len/2]

#     halfExtents = [[s/2 for s in base_size], [s/2 for s in finger_size], [s/2 for s in finger_size]]
#     positions = [base_pos, finger1_pos, finger2_pos]
    
#     vis_shapes = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX]*3, 
#                                           halfExtents=halfExtents, visualFramePositions=positions,
#                                           rgbaColors=[[0.2,0.2,0.2,1]]*3)
#     col_shapes = p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX]*3, 
#                                              halfExtents=halfExtents, collisionFramePositions=positions)

#     gripper_id = p.createMultiBody(baseMass=0.01, baseCollisionShapeIndex=col_shapes, 
#                                    baseVisualShapeIndex=vis_shapes, basePosition=[0,0,0])

#     ee_idx = get_link_index(robot_id, ee_link_name)
#     if ee_idx != -1:
#         p.createConstraint(robot_id, ee_idx, gripper_id, -1, p.JOINT_FIXED, 
#                         jointAxis=[0, 0, 0], parentFramePosition=[0,0,0], childFramePosition=[0, 0, 0])
#         p.setCollisionFilterPair(robot_id, gripper_id, ee_idx, -1, 0)
#         p.setCollisionFilterPair(robot_id, gripper_id, ee_idx-1, -1, 0)

#     full_box_size = [total_thick, total_width, total_len]
#     center_offset = [0, 0, total_len/2]
#     tool_pose = mplib.Pose(p=center_offset, q=[1, 0, 0, 0])
#     planner.update_attached_box(size=full_box_size, pose=tool_pose, link_id=-1)
#     print("✅ Gripper attached and collision model updated.")

# # ---------------------------------------------------------
# # HELPER: Sync Simulation to Real Robot
# # ---------------------------------------------------------
# def sync_sim_to_real(robot_id, active_indices, real_q):
#     """Snaps the PyBullet robot to the exact joint angles of the Real Robot."""
#     for i, angle_rad in enumerate(real_q):
#         idx = active_indices[i]
#         if idx >= 0:
#             p.resetJointState(robot_id, idx, angle_rad)

# # ---------------------------------------------------------
# # HELPER: GUI Sliders
# # ---------------------------------------------------------
# def setup_debug_sliders():
#     p.addUserDebugText("Left Click to Rotate Camera", [0, 0, 0.7], [0, 0, 0], textSize=1.2)
#     sx = p.addUserDebugParameter("Target X", -0.6, 0.6, 0.3)
#     sy = p.addUserDebugParameter("Target Y", -0.6, 0.6, 0.1)
#     sz = p.addUserDebugParameter("Target Z", 0.0, 0.8, 0.4) 
#     sqx = p.addUserDebugParameter("Target Qx", -1.0, 1.0, 1.0)
#     sqy = p.addUserDebugParameter("Target Qy", -1.0, 1.0, 0.0)
#     sqz = p.addUserDebugParameter("Target Qz", -1.0, 1.0, 0.0)
#     sqw = p.addUserDebugParameter("Target Qw", -1.0, 1.0, 0.0)
    
#     # Workflow Buttons
#     btn_plan = p.addUserDebugParameter("[1] PLAN (Sim Only)", 1, 0, 0)
#     btn_exec = p.addUserDebugParameter("[2] EXECUTE (Real Robot)", 1, 0, 0)
    
#     return [sx, sy, sz, sqx, sqy, sqz, sqw, btn_plan, btn_exec]

# def main():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     urdf_path = os.path.join(current_dir, "ur3_mesh_.urdf") 
#     srdf_path = os.path.join(current_dir, "ur3_mplib.srdf")
    
#     if not os.path.exists(urdf_path):
#         print(f"❌ CRITICAL ERROR: Could not find {urdf_path}")
#         return

#     generate_srdf(srdf_path)

#     # -----------------------------------------------------
#     # 1. CONNECT TO REAL ROBOT
#     # -----------------------------------------------------
#     print(f"🔌 Connecting to UR3 at {ROBOT_IP}...")
#     try:
#         rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
#         rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
#         print("✅ Connected to Real Robot via RTDE!")
#     except Exception as e:
#         print(f"❌ Connection Failed: {e}")
#         return

#     # Move to Initial Configuration (Safety first!)
#     # Convert degrees to radians
#     init_q_rad = np.deg2rad(INIT_JOINTS_DEG).tolist()
    
#     print(f"⚠️ Moving Real Robot to Start Pose: {INIT_JOINTS_DEG} (deg)...")
#     rtde_c.moveJ(init_q_rad, 0.5, 0.3) # speed 0.5 rad/s, accel 0.3 rad/s^2
#     print("✅ Real Robot at Start Pose.")

#     # -----------------------------------------------------
#     # 2. SETUP PYBULLET
#     # -----------------------------------------------------
#     p.connect(p.GUI)
#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     p.setGravity(0, 0, -9.81)
#     p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
#     p.resetDebugVisualizerCamera(1.5, 45, -30, [0, 0, 0])
#     p.loadURDF("plane.urdf")
    
#     # Obstacles
#     table_pos, table_size = [0, 0, -0.05], [1.0, 1.0, 0.1]
#     p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in table_size]), 
#                       basePosition=table_pos, baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in table_size], rgbaColor=[0.6, 0.4, 0.2, 1]))
    
#     wall_pos, wall_size = [0, -0.225, 0.5], [1.0, 0.05, 1.0]
#     p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in wall_size]), 
#                       basePosition=wall_pos, baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in wall_size], rgbaColor=[0.8, 0.2, 0.2, 0.8]))

#     robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=1)
    
#     # Setup MPLIB
#     planner = mplib.Planner(urdf=urdf_path, srdf=srdf_path, move_group="tool0")
#     planner.update_point_cloud(np.vstack([get_box_point_cloud(table_pos, table_size), get_box_point_cloud(wall_pos, wall_size)]))
#     attach_procedural_gripper(robot_id, "tool0", planner)

#     # Joint Mapping
#     mplib_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
#                          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
#     pb_joint_indices = get_pybullet_joint_indices(robot_id, mplib_joint_names)
#     active_indices = [i for i in pb_joint_indices if i >= 0]

#     # Sync Sim to Real immediately
#     actual_q = rtde_r.getActualQ() # Get current real joints (rad)
#     sync_sim_to_real(robot_id, active_indices, actual_q)
    
#     # -----------------------------------------------------
#     # 3. MAIN LOOP
#     # -----------------------------------------------------
#     sliders = setup_debug_sliders()
#     [sid_x, sid_y, sid_z, sid_qx, sid_qy, sid_qz, sid_qw, sid_plan, sid_exec] = sliders
    
#     old_btn_plan = p.readUserDebugParameter(sid_plan)
#     old_btn_exec = p.readUserDebugParameter(sid_exec)
    
#     pending_path = None # Stores the calculated trajectory
#     sim_hold_pos = actual_q # Sim robot holds this position

#     print("\n✅ System Ready! Use GUI sliders.")
#     print("👉 Step 1: Adjust Sliders & Click PLAN.")
#     print("👉 Step 2: If satisfied, Click EXECUTE to move Real Robot.")

#     try:
#         while True:
#             # Keep Sim Robot holding position (Physics fix)
#             p.setJointMotorControlArray(robot_id, active_indices, p.POSITION_CONTROL, 
#                                         targetPositions=sim_hold_pos, forces=[100.0]*6)
#             p.stepSimulation()
            
#             # --- HANDLE "PLAN" BUTTON ---
#             btn_plan_val = p.readUserDebugParameter(sid_plan)
#             if btn_plan_val > old_btn_plan:
#                 old_btn_plan = btn_plan_val
                
#                 # 1. Get Target from Sliders
#                 tx, ty, tz = [p.readUserDebugParameter(s) for s in [sid_x, sid_y, sid_z]]
#                 q_raw = np.array([p.readUserDebugParameter(s) for s in [sid_qx, sid_qy, sid_qz, sid_qw]])
#                 norm = np.linalg.norm(q_raw)
#                 q_norm = q_raw / norm if norm > 1e-6 else [0,0,0,1]
#                 target_pose = mplib.Pose(p=[tx, ty, tz], q=[q_norm[3], q_norm[0], q_norm[1], q_norm[2]])

#                 # 2. Get Current Start State (From Real Robot)
#                 # MPLIB needs accurate start state to plan correctly
#                 current_q_real = rtde_r.getActualQ()
#                 sync_sim_to_real(robot_id, active_indices, current_q_real) # Update sim visually too

#                 print(f"\n🤔 Planning to: [{tx:.2f}, {ty:.2f}, {tz:.2f}]...")
#                 result = planner.plan_pose(target_pose, current_q_real, time_step=0.01)

#                 if result["status"] == "Success":
#                     print("✅ Path Found! Visualizing in Sim...")
#                     pending_path = result["position"]
                    
#                     # Visualize: Fast replay in Sim
#                     for waypoint in pending_path:
#                         sync_sim_to_real(robot_id, active_indices, waypoint)
#                         p.stepSimulation()
#                         time.sleep(0.01)
                    
#                     # Reset Sim to start so user knows we haven't moved real robot yet
#                     print("⚠️ Path Pending. Press EXECUTE to move real robot.")
#                     sim_hold_pos = current_q_real 
#                 else:
#                     print(f"❌ Planning Failed: {result['status']}")
#                     pending_path = None

#             # --- HANDLE "EXECUTE" BUTTON ---
#             btn_exec_val = p.readUserDebugParameter(sid_exec)
#             if btn_exec_val > old_btn_exec:
#                 old_btn_exec = btn_exec_val

#                 if pending_path is not None:
#                     print("\n🚀 EXECUTING ON REAL ROBOT...")
                    
#                     # Prepare path for RTDE movePath
#                     # movePath expects list of [q0...q5, vel, acc, blend]
#                     # We add default velocity/accel and small blend for smoothness
#                     rtde_path = []
#                     vel = 0.5
#                     acc = 0.5
#                     blend = 0.02 # 2cm blend radius
                    
#                     for i, wp in enumerate(pending_path):
#                         # RTDE requires appending [v, a, blend] to joint list
#                         # Last point should have 0 blend to stop
#                         b_val = 0.0 if i == len(pending_path)-1 else blend
#                         wp_list = list(wp) + [vel, acc, b_val]
#                         rtde_path.append(wp_list)

#                     # Send to Robot (Blocking call)
#                     try:
#                         rtde_c.movePath(rtde_path)
#                         print("✅ Real Execution Complete.")
                        
#                         # Sync Sim Final State
#                         final_real_q = rtde_r.getActualQ()
#                         sim_hold_pos = final_real_q
#                         sync_sim_to_real(robot_id, active_indices, final_real_q)
                        
#                         # Clear pending path
#                         pending_path = None
#                     except Exception as e:
#                         print(f"❌ RTDE Execution Error: {e}")
#                 else:
#                     print("⚠️ No pending plan! Please Click PLAN first.")

#             time.sleep(0.01)
            
#     except KeyboardInterrupt:
#         print("Stopping...")
#     finally:
#         rtde_c.stopScript()
#         rtde_c.disconnect()
#         print("🔌 Disconnected from Robot.")

# if __name__ == "__main__":
#     main()

# ##
# import mplib
# import pybullet as p
# import pybullet_data
# import numpy as np
# import time
# import os
# from scipy.spatial.transform import Rotation as R

# # ---------------------------------------------------------
# # CONFIGURATION
# # ---------------------------------------------------------
# ROBOT_IP = "192.168.1.23"
# INIT_JOINTS_DEG = [49.5, -99.57, -6.18, -71.6, -266.31, 358.6]
# SIM_ROBOT_WORLD_ROT_Z = np.pi 

# # Obstacles
# WALL_POS_WORLD = [-0.14, -0.14, 0.5]  
# WALL_ROT_Z_DEG = 45  
# WALL_SIZE = [0.02, 1.0, 1.0] 
# TABLE_SIZE = [1.0, 1.0, 0.1]
# TABLE_POS_WORLD = [0, 0, -TABLE_SIZE[2]/2]
# TABLE_ROT_Z_DEG = 0

# # Gripper Settings
# GRIPPER_RX = -1.5708
# GRIPPER_DIMS = [0.14, 0.08, 0.16] # Thick, Width, Length

# # --- UR RTDE IMPORTS ---
# try:
#     import rtde_control
#     import rtde_receive
# except ImportError:
#     import types
#     rtde_control = types.ModuleType("rtde_control")
#     rtde_control.Path = type("Path", (), {"addEntry": lambda *args: None})
#     rtde_control.PathEntry = type("PathEntry", (), {
#         "MoveJ": 1, "PositionJoints": 1,
#         "__init__": lambda *args: None
#     })
#     RTDEControlInterface = type('RTDEControlInterface', (object,), {
#         '__init__': lambda *args: None,
#         'moveJ': lambda *args: None,
#         'movePath': lambda *args: None,
#         'stopScript': lambda *args: None,
#         'disconnect': lambda *args: None
#     })
#     RTDEReceiveInterface = type('RTDEReceiveInterface', (object,), {
#         '__init__': lambda *args: None,
#         'getActualQ': lambda *args: np.deg2rad(INIT_JOINTS_DEG).tolist()
#     })
#     print("⚠️ WARNING: ur_rtde not found. Running in SIMULATION-ONLY mode.")

# # ---------------------------------------------------------
# # HELPER: Transform Calculations
# # ---------------------------------------------------------
# def get_object_transforms(pos_world_arr, rot_z_deg):
#     pos_world = np.array(pos_world_arr)
#     r_obj_local = R.from_euler('z', rot_z_deg, degrees=True)
#     r_world_robot = R.from_euler('z', -SIM_ROBOT_WORLD_ROT_Z, degrees=False)
#     pos_robot = r_world_robot.apply(pos_world)
#     quat_robot = (r_world_robot * r_obj_local).as_quat()
#     quat_world = r_obj_local.as_quat()
#     return pos_robot, quat_robot, pos_world, quat_world

# def world_to_robot_frame(pos_world, quat_world):
#     r_base_world = R.from_euler('z', SIM_ROBOT_WORLD_ROT_Z, degrees=False)
#     r_target_world = R.from_quat(quat_world)
#     r_target_robot = r_base_world.inv() * r_target_world
#     p_target_robot = r_base_world.inv().apply(pos_world)
#     return p_target_robot, r_target_robot.as_quat()

# def normalize_angle(angle_rad):
#     return (angle_rad + np.pi) % (2 * np.pi) - np.pi

# # ---------------------------------------------------------
# # HELPER: Environment & Point Cloud
# # ---------------------------------------------------------
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

# def visualize_point_cloud_in_world(points_robot_frame):
#     r_robot_world = R.from_euler('z', SIM_ROBOT_WORLD_ROT_Z, degrees=False)
#     points_world = r_robot_world.apply(points_robot_frame)
#     if len(points_world) > 3000: 
#         indices = np.random.choice(len(points_world), 3000, replace=False)
#         vis_points = points_world[indices]
#     else:
#         vis_points = points_world
#     p.addUserDebugPoints(vis_points, [[0, 0, 1]] * len(vis_points), pointSize=2)

# # ---------------------------------------------------------
# # HELPER: Visualizers
# # ---------------------------------------------------------
# def load_ghost_robot(urdf_path, base_pos, base_orn):
#     ghost_id = p.loadURDF(urdf_path, base_pos, baseOrientation=base_orn, useFixedBase=1)
#     for i in range(-1, p.getNumJoints(ghost_id)):
#         p.changeVisualShape(ghost_id, i, rgbaColor=[0, 1, 0, 0.4]) 
#         p.setCollisionFilterGroupMask(ghost_id, i, 0, 0)
#     return ghost_id

# def set_ghost_config(ghost_id, active_indices, q_pos):
#     for i, angle in enumerate(q_pos):
#         idx = active_indices[i]
#         if idx >= 0: p.resetJointState(ghost_id, idx, angle)

# def draw_coordinate_frame(pos, quat, axis_len=0.15, life_time=0.04, line_width=3):
#     rot = R.from_quat(quat).as_matrix()
#     p.addUserDebugLine(pos, pos + rot[:,0]*axis_len, [1,0,0], lineWidth=line_width, lifeTime=life_time) 
#     p.addUserDebugLine(pos, pos + rot[:,1]*axis_len, [0,1,0], lineWidth=line_width, lifeTime=life_time) 
#     p.addUserDebugLine(pos, pos + rot[:,2]*axis_len, [0,0,1], lineWidth=line_width, lifeTime=life_time) 

# def draw_mplib_debug_box(robot_id, ee_link_idx, gripper_dims, offset_pos, offset_quat):
#     """
#     Draws a PINK wireframe box representing exactly where MPLIB thinks the gripper is.
#     """
#     if ee_link_idx == -1: return
    
#     ls = p.getLinkState(robot_id, ee_link_idx)
#     tool_pos, tool_orn = ls[4], ls[5]
    
#     r_tool = R.from_quat(tool_orn)
#     # MPLIB (w,x,y,z) -> Scipy (x,y,z,w)
#     r_offset = R.from_quat([offset_quat[1], offset_quat[2], offset_quat[3], offset_quat[0]]) 
    
#     box_orn_world = (r_tool * r_offset).as_matrix()
#     box_center_world = np.array(tool_pos) + r_tool.apply(offset_pos)
    
#     hx, hy, hz = np.array(gripper_dims) / 2
    
#     corners_local = [
#         [ hx,  hy,  hz], [ hx,  hy, -hz], [ hx, -hy,  hz], [ hx, -hy, -hz],
#         [-hx,  hy,  hz], [-hx,  hy, -hz], [-hx, -hy,  hz], [-hx, -hy, -hz]
#     ]
    
#     corners_world = [box_center_world + box_orn_world.dot(c) for c in corners_local]
    
#     edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
#     for i, j in edges:
#         p.addUserDebugLine(corners_world[i], corners_world[j], [1, 0, 1], lineWidth=2, lifeTime=0.05) 

# # ---------------------------------------------------------
# # HELPER: Procedural Gripper
# # ---------------------------------------------------------
# def attach_procedural_gripper(robot_id, ee_link_name, planner):
#     total_len, total_thick, total_width = GRIPPER_DIMS[2], GRIPPER_DIMS[0], GRIPPER_DIMS[1]
    
#     # 1. VISUALS (PyBullet)
#     base_len = 0.04 
#     base_size = [total_thick, total_width, base_len]
#     finger_size = [0.02, total_width, total_len - base_len]
#     base_pos  = [0, 0, base_len/2] 
#     finger1_pos = [-(total_thick - 0.02)/2, 0, base_len + finger_size[2]/2]
#     finger2_pos = [(total_thick - 0.02)/2, 0, base_len + finger_size[2]/2]

#     vis_shapes = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX]*3, 
#                                           halfExtents=[[s/2 for s in base_size], [s/2 for s in finger_size], [s/2 for s in finger_size]], 
#                                           visualFramePositions=[base_pos, finger1_pos, finger2_pos],
#                                           rgbaColors=[[0.2,0.2,0.2,1]]*3)
#     col_shapes = p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX]*3, 
#                                              halfExtents=[[s/2 for s in base_size], [s/2 for s in finger_size], [s/2 for s in finger_size]], 
#                                              collisionFramePositions=[base_pos, finger1_pos, finger2_pos])

#     ee_idx = -1
#     for i in range(p.getNumJoints(robot_id)):
#         if p.getJointInfo(robot_id, i)[12].decode("utf-8") == ee_link_name:
#             ee_idx = i; break
            
#     gripper_id, q_corr = None, [0,0,0,1]

#     if ee_idx != -1:
#         ls = p.getLinkState(robot_id, ee_idx)
#         tool_pos, tool_orn = ls[4], ls[5]
        
#         # This rotation (GRIPPER_RX) is used for the constraint.
#         # The physical child body will rotate by INVERSE of this.
#         q_corr = p.getQuaternionFromEuler([GRIPPER_RX, 0, 0]) 
        
#         # Calculate visual quaternion (Inverse of q_corr)
#         # This is the ACTUAL rotation of the visible gripper relative to tool0
#         q_visual_rel = R.from_quat(q_corr).inv().as_quat()
        
#         spawn_orn = (R.from_quat(tool_orn) * R.from_quat(q_visual_rel)).as_quat()
        
#         gripper_id = p.createMultiBody(baseMass=0.01, baseCollisionShapeIndex=col_shapes, 
#                                        baseVisualShapeIndex=vis_shapes, 
#                                        basePosition=tool_pos, baseOrientation=spawn_orn)
        
#         p.createConstraint(robot_id, ee_idx, gripper_id, -1, p.JOINT_FIXED, 
#             jointAxis=[0, 0, 0], parentFramePosition=[0,0,0], childFramePosition=[0, 0, 0],
#             childFrameOrientation=q_corr)
            
#         p.setCollisionFilterPair(robot_id, gripper_id, ee_idx, -1, 0)
#         p.setCollisionFilterPair(robot_id, gripper_id, ee_idx-1, -1, 0)
    
#         # 2. COLLISION (MPLIB) - FIXED ALIGNMENT
#         # We must apply the ACTUAL visual rotation (q_visual_rel) to the collision box
#         safe_dims = [total_thick-0.01, total_width-0.01, total_len]
#         center_offset_local = [0, 0, total_len/2 + 0.015] 
        
#         r_visual = R.from_quat(q_visual_rel)
        
#         # Rotate the offset vector
#         offset_vec = r_visual.apply(center_offset_local)
        
#         # Rotate the box orientation
#         # MPLIB uses [w, x, y, z]
#         q_vis = r_visual.as_quat()
#         quat_mplib_order = [q_vis[3], q_vis[0], q_vis[1], q_vis[2]] 
        
#         tool_pose = mplib.Pose(p=offset_vec, q=quat_mplib_order) 
#         planner.update_attached_box(size=safe_dims, pose=tool_pose, link_id=-1)
        
#         # Return the corrected MPLIB data for debug visualization
#         return gripper_id, q_visual_rel, safe_dims, offset_vec, quat_mplib_order

#     return None, None, None, None, None

# def sync_sim_to_real(robot_id, active_indices, real_q, gripper_id=None, gripper_offset=None):
#     if len(active_indices) != len(real_q): return
#     for i, angle_rad in enumerate(real_q):
#         idx = active_indices[i]
#         if idx >= 0: p.resetJointState(robot_id, idx, normalize_angle(angle_rad))
            
#     if gripper_id is not None and gripper_offset is not None:
#         ee_idx = -1 
#         for i in range(p.getNumJoints(robot_id)):
#             if p.getJointInfo(robot_id, i)[12].decode("utf-8") == "tool0":
#                 ee_idx = i; break
#         if ee_idx != -1:
#             ls = p.getLinkState(robot_id, ee_idx)
#             tool_pos, tool_orn = ls[4], ls[5]
            
#             # The gripper follows the Tool Frame rotated by the Visual Offset
#             r_tool = R.from_quat(tool_orn)
#             r_vis = R.from_quat(gripper_offset)
            
#             gripper_new_orn = (r_tool * r_vis).as_quat()
#             p.resetBasePositionAndOrientation(gripper_id, tool_pos, gripper_new_orn)

# def setup_debug_sliders():
#     sx = p.addUserDebugParameter("Target X", -0.6, 0.6, 0.3)
#     sy = p.addUserDebugParameter("Target Y", -0.6, 0.6, 0.1)
#     sz = p.addUserDebugParameter("Target Z", 0.0, 0.8, 0.3) 
#     sr = p.addUserDebugParameter("Roll (Deg)", -180, 180, 180) 
#     sp = p.addUserDebugParameter("Pitch (Deg)", -180, 180, 0)
#     sya = p.addUserDebugParameter("Yaw (Deg)", -180, 180, 0)
#     btn_plan = p.addUserDebugParameter("[1] PLAN", 1, 0, 0)
#     btn_exec = p.addUserDebugParameter("[2] EXECUTE", 1, 0, 0)
#     return [sx, sy, sz, sr, sp, sya, btn_plan, btn_exec]

# # ---------------------------------------------------------
# # MAIN
# # ---------------------------------------------------------
# def main():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     urdf_path = os.path.join(current_dir, "ur3_mesh_.urdf") 
#     srdf_path = os.path.join(current_dir, "ur3_mplib.srdf")
    
#     if not os.path.exists(srdf_path):
#         with open(srdf_path, "w") as f: f.write('<?xml version="1.0" ?><robot name="ur3"><group name="manipulator"><chain base_link="base_link" tip_link="tool0"/></group><disable_collisions link1="wrist_3_link" link2="tool0" reason="Adjacent" /></robot>')

#     try:
#         rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
#         rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
#     except:
#         rtde_c = RTDEControlInterface(ROBOT_IP)
#         rtde_r = RTDEReceiveInterface(ROBOT_IP)

#     print(f"🔌 Connected. Moving robot to Init Pose: {INIT_JOINTS_DEG}")
#     init_q_rad = np.deg2rad(INIT_JOINTS_DEG).tolist()
#     rtde_c.moveJ(init_q_rad, 0.5, 0.3) 
#     print("✅ Robot at Init Pose.")

#     p.connect(p.GUI)
#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     p.setGravity(0, 0, -9.81)
#     p.resetDebugVisualizerCamera(1.2, 90, -30, [0, 0, 0])
#     p.loadURDF("plane.urdf")
    
#     wall_pos_r, wall_quat_r, wall_pos_w, wall_quat_w = get_object_transforms(WALL_POS_WORLD, WALL_ROT_Z_DEG)
#     table_pos_r, table_quat_r, table_pos_w, table_quat_w = get_object_transforms(TABLE_POS_WORLD, TABLE_ROT_Z_DEG)

#     p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE]), 
#                       basePosition=wall_pos_w, baseOrientation=wall_quat_w,
#                       baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE], rgbaColor=[0.8, 0.2, 0.2, 0.8]))
#     p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE]), 
#                       basePosition=table_pos_w, baseOrientation=table_quat_w,
#                       baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE], rgbaColor=[0.5, 0.3, 0.1, 1.0]))

#     sim_robot_orn = p.getQuaternionFromEuler([0, 0, SIM_ROBOT_WORLD_ROT_Z])
#     robot_id = p.loadURDF(urdf_path, [0, 0, 0], baseOrientation=sim_robot_orn, useFixedBase=1)
#     ghost_id = load_ghost_robot(urdf_path, [0, 0, 0], sim_robot_orn)
#     p.resetBasePositionAndOrientation(ghost_id, [0,0,-5], [0,0,0,1]) 

#     planner = mplib.Planner(urdf=urdf_path, srdf=srdf_path, move_group="tool0")
#     combined_pts = np.vstack([
#         get_box_point_cloud(wall_pos_r, WALL_SIZE, wall_quat_r, resolution=0.04), 
#         get_box_point_cloud(table_pos_r, TABLE_SIZE, table_quat_r, resolution=0.04)
#     ])
#     planner.update_point_cloud(combined_pts)
#     visualize_point_cloud_in_world(combined_pts)

#     joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
#     name_to_idx = {p.getJointInfo(robot_id, i)[1].decode('utf-8'): i for i in range(p.getNumJoints(robot_id))}
#     active_indices = [name_to_idx.get(n, -1) for n in joint_names]

#     # capture extra debug info here
#     gripper_id, gripper_corr, mplib_box_size, mplib_offset, mplib_quat = attach_procedural_gripper(robot_id, "tool0", planner)
    
#     actual_q = rtde_r.getActualQ()
#     sync_sim_to_real(robot_id, active_indices, actual_q, gripper_id, gripper_corr)
    
#     ee_link_idx = -1
#     for i in range(p.getNumJoints(robot_id)):
#         if p.getJointInfo(robot_id, i)[12].decode("utf-8") == "tool0":
#             ee_link_idx = i; break

#     sliders = setup_debug_sliders()
#     [sid_x, sid_y, sid_z, sid_r, sid_p, sid_yaw, sid_plan, sid_exec] = sliders
#     old_btn_plan, old_btn_exec = 0, 0
#     sim_hold_pos = [normalize_angle(q) for q in actual_q]
    
#     print("\n✅ System Ready!")

#     while True:
#         p.setJointMotorControlArray(robot_id, active_indices, p.POSITION_CONTROL, targetPositions=sim_hold_pos, forces=[100.0]*6)
#         p.stepSimulation()

#         # Update Target Frame
#         tx, ty, tz = [p.readUserDebugParameter(s) for s in [sid_x, sid_y, sid_z]]
#         rr = p.readUserDebugParameter(sid_r)
#         rp = p.readUserDebugParameter(sid_p)
#         ry = p.readUserDebugParameter(sid_yaw)
#         q_norm_world = p.getQuaternionFromEuler(np.deg2rad([rr, rp, ry])) 
        
#         draw_coordinate_frame([tx, ty, tz], q_norm_world, life_time=0.05)

#         # Update Current Frame
#         curr_pos, curr_orn = [0,0,0], [0,0,0,1]
#         if ee_link_idx != -1:
#             ls = p.getLinkState(robot_id, ee_link_idx)
#             curr_pos, curr_orn = ls[4], ls[5]
#             draw_coordinate_frame(curr_pos, curr_orn, axis_len=0.15, life_time=0.05)
            
#             # --- DRAW THE INVISIBLE MPLIB BOX ---
#             draw_mplib_debug_box(robot_id, ee_link_idx, mplib_box_size, mplib_offset, mplib_quat)

#         p.removeAllUserDebugItems() 
#         hud_msg = f"TARGET:  [{tx:.2f}, {ty:.2f}, {tz:.2f}]\nCURRENT: [{curr_pos[0]:.2f}, {curr_pos[1]:.2f}, {curr_pos[2]:.2f}]"
#         p.addUserDebugText(hud_msg, [-0.5, 0, 0.9], [0,0,0], textSize=1.5)

#         # --- PLAN ---
#         if p.readUserDebugParameter(sid_plan) > old_btn_plan:
#             old_btn_plan = p.readUserDebugParameter(sid_plan)
            
#             # Diagnostic Check (FIXED ARGUMENTS)
#             current_q_real = rtde_r.getActualQ()
#             if planner.check_for_self_collision(current_q_real):
#                 print("⚠️ START STATE IN SELF COLLISION!")
#             if planner.check_for_env_collision(current_q_real):
#                 print("⚠️ START STATE IN ENV COLLISION!")

#             target_pos_robot, target_quat_robot = world_to_robot_frame([tx, ty, tz], q_norm_world)
#             target_pose = mplib.Pose(p=target_pos_robot, q=[target_quat_robot[3], target_quat_robot[0], target_quat_robot[1], target_quat_robot[2]])
            
#             sync_sim_to_real(robot_id, active_indices, current_q_real, gripper_id, gripper_corr)
            
#             print(f"🤔 Planning...")
#             result = planner.plan_pose(target_pose, current_q_real, time_step=0.01)

#             if result["status"] == "Success":
#                 print("✅ Path Found! Displaying Ghost...")
#                 path = result["position"]
                
#                 for waypoint in path:
#                     sync_sim_to_real(robot_id, active_indices, waypoint, gripper_id, gripper_corr)
#                     p.stepSimulation()
#                     time.sleep(0.01)

#                 sync_sim_to_real(robot_id, active_indices, current_q_real, gripper_id, gripper_corr)
#                 sim_hold_pos = [normalize_angle(q) for q in current_q_real]
                
#                 p.resetBasePositionAndOrientation(ghost_id, [0,0,0], sim_robot_orn)
#                 set_ghost_config(ghost_id, active_indices, path[-1]) 
#                 pending_path = path
#             else:
#                 print(f"❌ Planning Failed: {result['status']}")
#                 pending_path = None

#         # --- EXECUTE ---
#         if p.readUserDebugParameter(sid_exec) > old_btn_exec:
#             old_btn_exec = p.readUserDebugParameter(sid_exec)
#             if pending_path is not None:
#                 print("🚀 EXECUTING on REAL ROBOT (ur_rtde 1.6.2 Compatible)...")
#                 try:
#                     robot_path = rtde_control.Path()
#                     for i, wp in enumerate(pending_path):
#                         q_list = [float(val) for val in wp]
#                         vel, acc = 0.5, 0.5
#                         blend = 0.0 if i == len(pending_path)-1 else 0.02
#                         full_params = q_list + [vel, acc, blend]
                        
#                         robot_path.addEntry(rtde_control.PathEntry(
#                             rtde_control.PathEntry.MoveJ, 
#                             rtde_control.PathEntry.PositionJoints, 
#                             full_params 
#                         ))
                    
#                     print(f"   Sending path with {len(pending_path)} waypoints...")
#                     rtde_c.movePath(robot_path)
#                     print("✅ Done.")
                    
#                     final_q = rtde_r.getActualQ()
#                     sim_hold_pos = [normalize_angle(q) for q in final_q]
#                     sync_sim_to_real(robot_id, active_indices, final_q, gripper_id, gripper_corr)
#                     set_ghost_config(ghost_id, active_indices, final_q)
#                     pending_path = None
                    
#                 except Exception as e:
#                     print(f"❌ Execution Error: {e}")

#         time.sleep(0.02) 

# if __name__ == "__main__":
#     main()


##
import mplib
import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ROBOT_IP = "192.168.1.23"
INIT_JOINTS_DEG = [49.5, -99.57, -6.18, -71.6, -266.31, 358.6]
SIM_ROBOT_WORLD_ROT_Z = np.pi 

# Obstacles
WALL_POS_WORLD = [-0.14, -0.14, 0.5]  
WALL_ROT_Z_DEG = 45  
WALL_SIZE = [0.02, 1.0, 1.0] 
TABLE_SIZE = [1.0, 1.0, 0.1]
TABLE_POS_WORLD = [0, 0, -TABLE_SIZE[2]/2]
TABLE_ROT_Z_DEG = 0

# Gripper Settings
GRIPPER_RX = -1.5708
GRIPPER_DIMS = [0.14, 0.08, 0.16] # Thick, Width, Length

# --- UR RTDE IMPORTS ---
try:
    import rtde_control
    import rtde_receive
except ImportError:
    import types
    rtde_control = types.ModuleType("rtde_control")
    rtde_control.Path = type("Path", (), {"addEntry": lambda *args: None})
    rtde_control.PathEntry = type("PathEntry", (), {
        "MoveJ": 1, "PositionJoints": 1,
        "__init__": lambda *args: None
    })
    RTDEControlInterface = type('RTDEControlInterface', (object,), {
        '__init__': lambda *args: None,
        'moveJ': lambda *args: None,
        'movePath': lambda *args: None,
        'stopScript': lambda *args: None,
        'disconnect': lambda *args: None
    })
    RTDEReceiveInterface = type('RTDEReceiveInterface', (object,), {
        '__init__': lambda *args: None,
        'getActualQ': lambda *args: np.deg2rad(INIT_JOINTS_DEG).tolist()
    })
    print("⚠️ WARNING: ur_rtde not found. Running in SIMULATION-ONLY mode.")

# ---------------------------------------------------------
# HELPER: Transform Calculations
# ---------------------------------------------------------
def get_object_transforms(pos_world_arr, rot_z_deg):
    pos_world = np.array(pos_world_arr)
    r_obj_local = R.from_euler('z', rot_z_deg, degrees=True)
    r_world_robot = R.from_euler('z', -SIM_ROBOT_WORLD_ROT_Z, degrees=False)
    pos_robot = r_world_robot.apply(pos_world)
    quat_robot = (r_world_robot * r_obj_local).as_quat()
    quat_world = r_obj_local.as_quat()
    return pos_robot, quat_robot, pos_world, quat_world

def world_to_robot_frame(pos_world, quat_world):
    r_base_world = R.from_euler('z', SIM_ROBOT_WORLD_ROT_Z, degrees=False)
    r_target_world = R.from_quat(quat_world)
    r_target_robot = r_base_world.inv() * r_target_world
    p_target_robot = r_base_world.inv().apply(pos_world)
    return p_target_robot, r_target_robot.as_quat()

def normalize_angle(angle_rad):
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

# ---------------------------------------------------------
# HELPER: Environment & Point Cloud
# ---------------------------------------------------------
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

def visualize_point_cloud_in_world(points_robot_frame):
    r_robot_world = R.from_euler('z', SIM_ROBOT_WORLD_ROT_Z, degrees=False)
    points_world = r_robot_world.apply(points_robot_frame)
    if len(points_world) > 3000: 
        indices = np.random.choice(len(points_world), 3000, replace=False)
        vis_points = points_world[indices]
    else:
        vis_points = points_world
    p.addUserDebugPoints(vis_points, [[0, 0, 1]] * len(vis_points), pointSize=2)

# ---------------------------------------------------------
# HELPER: Visualizers
# ---------------------------------------------------------
def load_ghost_robot(urdf_path, base_pos, base_orn):
    ghost_id = p.loadURDF(urdf_path, base_pos, baseOrientation=base_orn, useFixedBase=1)
    for i in range(-1, p.getNumJoints(ghost_id)):
        p.changeVisualShape(ghost_id, i, rgbaColor=[0, 1, 0, 0.4]) 
        p.setCollisionFilterGroupMask(ghost_id, i, 0, 0)
    return ghost_id

def set_ghost_config(ghost_id, active_indices, q_pos):
    for i, angle in enumerate(q_pos):
        idx = active_indices[i]
        if idx >= 0: p.resetJointState(ghost_id, idx, angle)

def draw_coordinate_frame(pos, quat, axis_len=0.15, life_time=0.04, line_width=3):
    """Draws a Red/Green/Blue coordinate axis at the given pose."""
    rot = R.from_quat(quat).as_matrix()
    p.addUserDebugLine(pos, pos + rot[:,0]*axis_len, [1,0,0], lineWidth=line_width, lifeTime=life_time) 
    p.addUserDebugLine(pos, pos + rot[:,1]*axis_len, [0,1,0], lineWidth=line_width, lifeTime=life_time) 
    p.addUserDebugLine(pos, pos + rot[:,2]*axis_len, [0,0,1], lineWidth=line_width, lifeTime=life_time) 

# ---------------------------------------------------------
# HELPER: Procedural Gripper
# ---------------------------------------------------------
def attach_procedural_gripper(robot_id, ee_link_name, planner):
    total_len, total_thick, total_width = GRIPPER_DIMS[2], GRIPPER_DIMS[0], GRIPPER_DIMS[1]
    
    # 1. VISUALS (PyBullet)
    base_len = 0.04 
    base_size = [total_thick, total_width, base_len]
    finger_size = [0.02, total_width, total_len - base_len]
    base_pos  = [0, 0, base_len/2] 
    finger1_pos = [-(total_thick - 0.02)/2, 0, base_len + finger_size[2]/2]
    finger2_pos = [(total_thick - 0.02)/2, 0, base_len + finger_size[2]/2]

    vis_shapes = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX]*3, 
                                          halfExtents=[[s/2 for s in base_size], [s/2 for s in finger_size], [s/2 for s in finger_size]], 
                                          visualFramePositions=[base_pos, finger1_pos, finger2_pos],
                                          rgbaColors=[[0.2,0.2,0.2,1]]*3)
    col_shapes = p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX]*3, 
                                             halfExtents=[[s/2 for s in base_size], [s/2 for s in finger_size], [s/2 for s in finger_size]], 
                                             collisionFramePositions=[base_pos, finger1_pos, finger2_pos])

    ee_idx = -1
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[12].decode("utf-8") == ee_link_name:
            ee_idx = i; break
            
    gripper_id, q_corr = None, [0,0,0,1]

    if ee_idx != -1:
        ls = p.getLinkState(robot_id, ee_idx)
        tool_pos, tool_orn = ls[4], ls[5]
        
        q_corr = p.getQuaternionFromEuler([GRIPPER_RX, 0, 0]) 
        # Calculate visual quaternion (Inverse of q_corr)
        q_visual_rel = R.from_quat(q_corr).inv().as_quat()
        
        spawn_orn = (R.from_quat(tool_orn) * R.from_quat(q_visual_rel)).as_quat()
        
        gripper_id = p.createMultiBody(baseMass=0.01, baseCollisionShapeIndex=col_shapes, 
                                       baseVisualShapeIndex=vis_shapes, 
                                       basePosition=tool_pos, baseOrientation=spawn_orn)
        
        p.createConstraint(robot_id, ee_idx, gripper_id, -1, p.JOINT_FIXED, 
            jointAxis=[0, 0, 0], parentFramePosition=[0,0,0], childFramePosition=[0, 0, 0],
            childFrameOrientation=q_corr)
            
        p.setCollisionFilterPair(robot_id, gripper_id, ee_idx, -1, 0)
        p.setCollisionFilterPair(robot_id, gripper_id, ee_idx-1, -1, 0)
    
        # 2. COLLISION (MPLIB) - FIXED ALIGNMENT
        safe_dims = [total_thick-0.01, total_width-0.01, total_len]
        center_offset_local = [0, 0, total_len/2 + 0.015] 
        
        r_visual = R.from_quat(q_visual_rel)
        offset_vec = r_visual.apply(center_offset_local)
        
        q_vis = r_visual.as_quat()
        quat_mplib_order = [q_vis[3], q_vis[0], q_vis[1], q_vis[2]] 
        
        tool_pose = mplib.Pose(p=offset_vec, q=quat_mplib_order) 
        planner.update_attached_box(size=safe_dims, pose=tool_pose, link_id=-1)
        
        return gripper_id, q_visual_rel

    return None, None

def sync_sim_to_real(robot_id, active_indices, real_q, gripper_id=None, gripper_offset=None):
    if len(active_indices) != len(real_q): return
    for i, angle_rad in enumerate(real_q):
        idx = active_indices[i]
        if idx >= 0: p.resetJointState(robot_id, idx, normalize_angle(angle_rad))
            
    if gripper_id is not None and gripper_offset is not None:
        ee_idx = -1 
        for i in range(p.getNumJoints(robot_id)):
            if p.getJointInfo(robot_id, i)[12].decode("utf-8") == "tool0":
                ee_idx = i; break
        if ee_idx != -1:
            ls = p.getLinkState(robot_id, ee_idx)
            tool_pos, tool_orn = ls[4], ls[5]
            
            # Gripper follows Tool * Visual Offset
            r_tool = R.from_quat(tool_orn)
            r_vis = R.from_quat(gripper_offset)
            
            gripper_new_orn = (r_tool * r_vis).as_quat()
            p.resetBasePositionAndOrientation(gripper_id, tool_pos, gripper_new_orn)

def setup_debug_sliders():
    sx = p.addUserDebugParameter("Target X", -0.6, 0.6, 0.3)
    sy = p.addUserDebugParameter("Target Y", -0.6, 0.6, 0.1)
    sz = p.addUserDebugParameter("Target Z", 0.0, 0.8, 0.3) 
    sr = p.addUserDebugParameter("Roll (Deg)", -180, 180, 180) 
    sp = p.addUserDebugParameter("Pitch (Deg)", -180, 180, 0)
    sya = p.addUserDebugParameter("Yaw (Deg)", -180, 180, 0)
    btn_plan = p.addUserDebugParameter("[1] PLAN", 1, 0, 0)
    btn_exec = p.addUserDebugParameter("[2] EXECUTE", 1, 0, 0)
    return [sx, sy, sz, sr, sp, sya, btn_plan, btn_exec]

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "ur3_mesh_.urdf") 
    srdf_path = os.path.join(current_dir, "ur3_mplib.srdf")
    
    if not os.path.exists(srdf_path):
        with open(srdf_path, "w") as f: f.write('<?xml version="1.0" ?><robot name="ur3"><group name="manipulator"><chain base_link="base_link" tip_link="tool0"/></group><disable_collisions link1="wrist_3_link" link2="tool0" reason="Adjacent" /></robot>')

    try:
        rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    except:
        rtde_c = RTDEControlInterface(ROBOT_IP)
        rtde_r = RTDEReceiveInterface(ROBOT_IP)

    print(f"🔌 Connected. Moving robot to Init Pose: {INIT_JOINTS_DEG}")
    init_q_rad = np.deg2rad(INIT_JOINTS_DEG).tolist()
    rtde_c.moveJ(init_q_rad, 0.5, 0.3) 
    print("✅ Robot at Init Pose.")

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(1.2, 90, -30, [0, 0, 0])
    p.loadURDF("plane.urdf")
    
    wall_pos_r, wall_quat_r, wall_pos_w, wall_quat_w = get_object_transforms(WALL_POS_WORLD, WALL_ROT_Z_DEG)
    table_pos_r, table_quat_r, table_pos_w, table_quat_w = get_object_transforms(TABLE_POS_WORLD, TABLE_ROT_Z_DEG)

    p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE]), 
                      basePosition=wall_pos_w, baseOrientation=wall_quat_w,
                      baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE], rgbaColor=[0.8, 0.2, 0.2, 0.8]))
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE]), 
                      basePosition=table_pos_w, baseOrientation=table_quat_w,
                      baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE], rgbaColor=[0.5, 0.3, 0.1, 1.0]))

    sim_robot_orn = p.getQuaternionFromEuler([0, 0, SIM_ROBOT_WORLD_ROT_Z])
    robot_id = p.loadURDF(urdf_path, [0, 0, 0], baseOrientation=sim_robot_orn, useFixedBase=1)
    ghost_id = load_ghost_robot(urdf_path, [0, 0, 0], sim_robot_orn)
    p.resetBasePositionAndOrientation(ghost_id, [0,0,-5], [0,0,0,1]) 

    planner = mplib.Planner(urdf=urdf_path, srdf=srdf_path, move_group="tool0")
    combined_pts = np.vstack([
        get_box_point_cloud(wall_pos_r, WALL_SIZE, wall_quat_r, resolution=0.04), 
        get_box_point_cloud(table_pos_r, TABLE_SIZE, table_quat_r, resolution=0.04)
    ])
    planner.update_point_cloud(combined_pts)
    visualize_point_cloud_in_world(combined_pts)

    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    name_to_idx = {p.getJointInfo(robot_id, i)[1].decode('utf-8'): i for i in range(p.getNumJoints(robot_id))}
    active_indices = [name_to_idx.get(n, -1) for n in joint_names]

    # Cleaned return values
    gripper_id, gripper_corr = attach_procedural_gripper(robot_id, "tool0", planner)
    
    actual_q = rtde_r.getActualQ()
    sync_sim_to_real(robot_id, active_indices, actual_q, gripper_id, gripper_corr)
    
    ee_link_idx = -1
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[12].decode("utf-8") == "tool0":
            ee_link_idx = i; break

    sliders = setup_debug_sliders()
    [sid_x, sid_y, sid_z, sid_r, sid_p, sid_yaw, sid_plan, sid_exec] = sliders
    old_btn_plan, old_btn_exec = 0, 0
    sim_hold_pos = [normalize_angle(q) for q in actual_q]
    
    print("\n✅ System Ready!")

    while True:
        p.setJointMotorControlArray(robot_id, active_indices, p.POSITION_CONTROL, targetPositions=sim_hold_pos, forces=[100.0]*6)
        p.stepSimulation()

        # Update Target Frame (Visual Only)
        tx, ty, tz = [p.readUserDebugParameter(s) for s in [sid_x, sid_y, sid_z]]
        rr, rp, ry = [p.readUserDebugParameter(s) for s in [sid_r, sid_p, sid_yaw]]
        q_norm_world = p.getQuaternionFromEuler(np.deg2rad([rr, rp, ry])) 
        
        draw_coordinate_frame([tx, ty, tz], q_norm_world, life_time=0.05)

        # Update Current Frame (Visual Only)
        curr_pos, curr_orn = [0,0,0], [0,0,0,1]
        if ee_link_idx != -1:
            ls = p.getLinkState(robot_id, ee_link_idx)
            curr_pos, curr_orn = ls[4], ls[5]
            draw_coordinate_frame(curr_pos, curr_orn, axis_len=0.15, life_time=0.05)

        p.removeAllUserDebugItems() 
        hud_msg = f"TARGET:  [{tx:.2f}, {ty:.2f}, {tz:.2f}]\nCURRENT: [{curr_pos[0]:.2f}, {curr_pos[1]:.2f}, {curr_pos[2]:.2f}]"
        p.addUserDebugText(hud_msg, [-0.5, 0, 0.9], [0,0,0], textSize=1.5)

        # --- PLAN ---
        if p.readUserDebugParameter(sid_plan) > old_btn_plan:
            old_btn_plan = p.readUserDebugParameter(sid_plan)
            
            current_q_real = rtde_r.getActualQ()
            if planner.check_for_self_collision(current_q_real):
                print("⚠️ START STATE IN SELF COLLISION!")
            if planner.check_for_env_collision(current_q_real):
                print("⚠️ START STATE IN ENV COLLISION!")

            target_pos_robot, target_quat_robot = world_to_robot_frame([tx, ty, tz], q_norm_world)
            target_pose = mplib.Pose(p=target_pos_robot, q=[target_quat_robot[3], target_quat_robot[0], target_quat_robot[1], target_quat_robot[2]])
            
            sync_sim_to_real(robot_id, active_indices, current_q_real, gripper_id, gripper_corr)
            
            print(f"🤔 Planning...")
            result = planner.plan_pose(target_pose, current_q_real, time_step=0.01)

            if result["status"] == "Success":
                print("✅ Path Found! Displaying Ghost...")
                path = result["position"]
                
                for waypoint in path:
                    sync_sim_to_real(robot_id, active_indices, waypoint, gripper_id, gripper_corr)
                    p.stepSimulation()
                    time.sleep(0.01)

                sync_sim_to_real(robot_id, active_indices, current_q_real, gripper_id, gripper_corr)
                sim_hold_pos = [normalize_angle(q) for q in current_q_real]
                
                p.resetBasePositionAndOrientation(ghost_id, [0,0,0], sim_robot_orn)
                set_ghost_config(ghost_id, active_indices, path[-1]) 
                pending_path = path
            else:
                print(f"❌ Planning Failed: {result['status']}")
                pending_path = None

        # --- EXECUTE ---
        if p.readUserDebugParameter(sid_exec) > old_btn_exec:
            old_btn_exec = p.readUserDebugParameter(sid_exec)
            if pending_path is not None:
                print("🚀 EXECUTING on REAL ROBOT (ur_rtde 1.6.2 Compatible)...")
                try:
                    robot_path = rtde_control.Path()
                    for i, wp in enumerate(pending_path):
                        q_list = [float(val) for val in wp]
                        vel, acc = 0.5, 0.5
                        blend = 0.0 if i == len(pending_path)-1 else 0.02
                        full_params = q_list + [vel, acc, blend]
                        
                        robot_path.addEntry(rtde_control.PathEntry(
                            rtde_control.PathEntry.MoveJ, 
                            rtde_control.PathEntry.PositionJoints, 
                            full_params 
                        ))
                    
                    print(f"   Sending path with {len(pending_path)} waypoints...")
                    rtde_c.movePath(robot_path)
                    print("✅ Done.")
                    
                    final_q = rtde_r.getActualQ()
                    sim_hold_pos = [normalize_angle(q) for q in final_q]
                    sync_sim_to_real(robot_id, active_indices, final_q, gripper_id, gripper_corr)
                    set_ghost_config(ghost_id, active_indices, final_q)
                    pending_path = None
                    
                except Exception as e:
                    print(f"❌ Execution Error: {e}")

        time.sleep(0.02) 

if __name__ == "__main__":
    main()