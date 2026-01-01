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
# # HELPER: GUI Sliders
# # ---------------------------------------------------------
# def setup_debug_sliders():
#     p.addUserDebugText("Left Click to Rotate!", [0, 0, 0.7], [0, 0, 0], textSize=1.2)
#     sx = p.addUserDebugParameter("Target X", -0.6, 0.6, 0.3)
#     sy = p.addUserDebugParameter("Target Y", -0.6, 0.6, 0.1)
#     sz = p.addUserDebugParameter("Target Z", 0.0, 0.8, 0.3)
#     sr = p.addUserDebugParameter("Roll (deg)", -180, 180, 180)
#     sp = p.addUserDebugParameter("Pitch (deg)", -180, 180, 0)
#     syaw = p.addUserDebugParameter("Yaw (deg)", -180, 180, 0)
#     btn = p.addUserDebugParameter(">>> PLAN & MOVE <<<", 1, 0, 0)
#     return [sx, sy, sz, sr, sp, syaw, btn]

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
    
#     # Disable default mouse picking to allow custom rotation
#     p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

#     # Set Initial Camera
#     cam_dist, cam_yaw, cam_pitch = 1.5, 45, -30
#     cam_target = [0, 0, 0]
#     p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_target)

#     p.loadURDF("plane.urdf")
    
#     # --- OBSTACLES (PyBullet Visuals) ---
#     table_size = [1.0, 1.0, 0.1]
#     table_pos  = [0, 0, -0.05]
#     wall_size = [1.0, 0.05, 1.0]
#     wall_pos  = [0, -0.225, 0.5]

#     t_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in table_size], rgbaColor=[0.6, 0.4, 0.2, 1])
#     t_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in table_size])
#     p.createMultiBody(0, t_col, t_vis, table_pos)
    
#     w_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in wall_size], rgbaColor=[0.8, 0.2, 0.2, 0.8])
#     w_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in wall_size])
#     p.createMultiBody(0, w_col, w_vis, wall_pos)

#     robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=1)
    
#     # --- MPLIB SETUP ---
#     print("🧠 Initializing Mplib Planner...")
#     planner = mplib.Planner(urdf=urdf_path, srdf=srdf_path, move_group="tool0")

#     print("🚧 Adding Obstacles to Planner (Point Cloud)...")
#     pts_table = get_box_point_cloud(table_pos, table_size)
#     pts_wall  = get_box_point_cloud(wall_pos, wall_size)
#     all_obstacle_points = np.vstack([pts_table, pts_wall])
    
#     # Add points to planner (without 'radius' arg for 0.2.1)
#     planner.update_point_cloud(all_obstacle_points)
#     print(f"✅ Added {len(all_obstacle_points)} collision points to Mplib world.")

#     mplib_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
#                          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
#     pb_joint_indices = get_pybullet_joint_indices(robot_id, mplib_joint_names)

#     home_qpos = [0, -1.57, 0, -1.57, 0, 0]
#     for i, angle in enumerate(home_qpos):
#         idx = pb_joint_indices[i]
#         if idx >= 0: p.resetJointState(robot_id, idx, angle)

#     # ---------------------------------------------------------
#     # MAIN LOOP
#     # ---------------------------------------------------------
#     sliders = setup_debug_sliders()
#     [sid_x, sid_y, sid_z, sid_r, sid_p, sid_yaw, sid_btn] = sliders
#     old_btn_val = p.readUserDebugParameter(sid_btn)
    
#     last_mouse_pos = None
#     is_rotating = False

#     print("\n✅ System Ready!")
#     print("👉 Use LEFT CLICK dragging to rotate the camera.")

#     while True:
#         p.stepSimulation()

#         # --- CUSTOM CAMERA LOGIC (Using Integers) ---
#         events = p.getMouseEvents()
#         for e in events:
#             # e[0]=eventType, e[1]=x, e[2]=y, e[3]=buttonIndex, e[4]=buttonState
#             # 2 = MOUSE_BUTTON_EVENT, 0 = Left Button
#             if e[0] == 2 and e[3] == 0:
#                 # 1 = KEY_IS_DOWN
#                 if e[4] == 1:
#                     is_rotating = True
#                     last_mouse_pos = (e[1], e[2])
#                     cam_info = p.getDebugVisualizerCamera()
#                     cam_yaw, cam_pitch, cam_dist = cam_info[8], cam_info[9], cam_info[10]
#                     cam_target = cam_info[11]
#                 else:
#                     is_rotating = False
#                     last_mouse_pos = None

#             # 1 = MOUSE_MOVE_EVENT
#             if e[0] == 1 and is_rotating and last_mouse_pos:
#                 dx = e[1] - last_mouse_pos[0]
#                 dy = e[2] - last_mouse_pos[1]
#                 cam_yaw -= dx * 0.3
#                 cam_pitch -= dy * 0.3
#                 p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_target)
#                 last_mouse_pos = (e[1], e[2])

#         # --- PLANNER LOGIC ---
#         btn_val = p.readUserDebugParameter(sid_btn)
#         if btn_val > old_btn_val:
#             old_btn_val = btn_val
            
#             x, y, z = [p.readUserDebugParameter(s) for s in [sid_x, sid_y, sid_z]]
#             r, pitch_val, yaw = [p.readUserDebugParameter(s) for s in [sid_r, sid_p, sid_yaw]]
            
#             rot = R.from_euler('xyz', [r, pitch_val, yaw], degrees=True)
#             quat = rot.as_quat() 
#             target_pose = mplib.Pose(p=[x, y, z], q=[quat[3], quat[0], quat[1], quat[2]])

#             current_qpos = []
#             for idx in pb_joint_indices:
#                 if idx >= 0:
#                     current_qpos.append(p.getJointState(robot_id, idx)[0])
#                 else:
#                     current_qpos.append(0.0)

#             print(f"🤔 Planning to: {[x,y,z]} ...")
#             result = planner.plan_pose(target_pose, current_qpos, time_step=0.01)
            
#             if result["status"] == "Success":
#                 print("🚀 Path Found!")
#                 path = result["position"]
#                 for waypoint in path:
#                     for i, angle in enumerate(waypoint):
#                         idx = pb_joint_indices[i]
#                         if idx >= 0: p.resetJointState(robot_id, idx, angle)
#                     p.stepSimulation()
#                     time.sleep(0.02)
#             else:
#                 print(f"❌ Planning Failed: {result['status']}")
                
#         time.sleep(0.01)

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
# HELPER: Generate SRDF
# ---------------------------------------------------------
def generate_srdf(srdf_path):
    srdf_content = """<?xml version="1.0" ?>
<robot name="ur3_mplib">
    <group name="manipulator">
        <chain base_link="base_link" tip_link="tool0"/>
    </group>
    <disable_collisions link1="base_link_inertia" link2="shoulder_link" reason="Adjacent" />
    <disable_collisions link1="base_link" link2="base_link_inertia" reason="Adjacent" />
    <disable_collisions link1="shoulder_link" link2="upper_arm_link" reason="Adjacent" />
    <disable_collisions link1="upper_arm_link" link2="forearm_link" reason="Adjacent" />
    <disable_collisions link1="forearm_link" link2="wrist_1_link" reason="Adjacent" />
    <disable_collisions link1="wrist_1_link" link2="wrist_2_link" reason="Adjacent" />
    <disable_collisions link1="wrist_2_link" link2="wrist_3_link" reason="Adjacent" />
    <disable_collisions link1="wrist_3_link" link2="tool0" reason="Adjacent" />
</robot>
"""
    with open(srdf_path, "w") as f:
        f.write(srdf_content)
    print(f"✅ Generated SRDF file at: {srdf_path}")

# ---------------------------------------------------------
# HELPER: Map Joint Names
# ---------------------------------------------------------
def get_pybullet_joint_indices(robot_id, joint_names):
    joint_indices = []
    num_joints = p.getNumJoints(robot_id)
    name_to_index = {}
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        name_to_index[name] = i

    for name in joint_names:
        if name in name_to_index:
            joint_indices.append(name_to_index[name])
        else:
            joint_indices.append(-1) 
    return joint_indices

def get_link_index(robot_id, link_name):
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        if info[12].decode("utf-8") == link_name:
            return i
    return -1

# ---------------------------------------------------------
# HELPER: Generate Point Cloud for Obstacles
# ---------------------------------------------------------
def get_box_point_cloud(center, size, resolution=0.02):
    x = np.arange(center[0] - size[0]/2, center[0] + size[0]/2, resolution)
    y = np.arange(center[1] - size[1]/2, center[1] + size[1]/2, resolution)
    z = np.arange(center[2] - size[2]/2, center[2] + size[2]/2, resolution)
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    return points

# ---------------------------------------------------------
# HELPER: Create & Attach "U-Shape" Gripper
# ---------------------------------------------------------
def attach_procedural_gripper(robot_id, ee_link_name, planner):
    """
    Creates a lightweight U-shaped gripper using a Compound Shape.
    Dimensions: 140mm (Thick/X) x 80mm (Width/Y) x 160mm (Length/Z)
    """
    
    # --- 1. Define Geometry Parts ---
    # We will build the gripper out of 3 boxes: 1 Base + 2 Fingers
    
    # Total Dimensions
    total_len = 0.16
    total_thick = 0.14
    total_width = 0.08
    
    # Part A: Base Plate (Connects to robot)
    # Takes up the first 4cm of length
    base_len = 0.04 
    base_size = [total_thick, total_width, base_len]
    base_pos  = [0, 0, base_len/2] # Centered in Z for this part
    
    # Part B: Finger 1 (Left)
    finger_len = total_len - base_len
    finger_thick = 0.02 # 2cm thick fingers
    finger_size = [finger_thick, total_width, finger_len]
    # Position: Shifted X to the left edge, Shifted Z to be after base
    finger1_pos = [-(total_thick/2 - finger_thick/2), 0, base_len + finger_len/2]

    # Part C: Finger 2 (Right)
    finger2_pos = [(total_thick/2 - finger_thick/2), 0, base_len + finger_len/2]

    # Arrays for Compound Shape
    halfExtents = [
        [s/2 for s in base_size], 
        [s/2 for s in finger_size], 
        [s/2 for s in finger_size]
    ]
    positions = [base_pos, finger1_pos, finger2_pos]
    
    # --- 2. Create PyBullet Visual/Collision ---
    # We make it VERY light (0.01kg) so it doesn't drag the arm down
    
    vis_shapes = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX]*3, 
                                          halfExtents=halfExtents, 
                                          visualFramePositions=positions,
                                          rgbaColors=[[0.2,0.2,0.2,1]]*3)
                                          
    col_shapes = p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX]*3, 
                                             halfExtents=halfExtents, 
                                             collisionFramePositions=positions)

    gripper_id = p.createMultiBody(baseMass=0.01,  # <--- VERY LIGHT
                                   baseCollisionShapeIndex=col_shapes, 
                                   baseVisualShapeIndex=vis_shapes, 
                                   basePosition=[0,0,0])

    # --- 3. Attach to Robot ---
    ee_idx = get_link_index(robot_id, ee_link_name)
    if ee_idx == -1: return

    # Fixed constraint at the end of the tool link
    # p.createConstraint(robot_id, ee_idx, gripper_id, -1, p.JOINT_FIXED, 
    #                    jointAxis=[0, 0, 0], parentFramePosition=[0,0,0], childFramePosition=[0, 0, 0])
    # 1. Attach the gripper with a Fixed Joint
    p.createConstraint(robot_id, ee_idx, gripper_id, -1, p.JOINT_FIXED, 
                    jointAxis=[0, 0, 0], parentFramePosition=[0,0,0], childFramePosition=[0, 0, 0])

    # 2. Disable collision ONLY between Gripper and the Link it attaches to (tool0)
    # This stops the shaking.
    p.setCollisionFilterPair(bodyUniqueIdA=robot_id, bodyUniqueIdB=gripper_id, 
                            linkIndexA=ee_idx, linkIndexB=-1, 
                            enableCollision=0)

    # 3. (Optional but recommended) Disable collision with the immediate parent (Wrist 3)
    # because the gripper base often clips slightly into the wrist mesh too.
    p.setCollisionFilterPair(bodyUniqueIdA=robot_id, bodyUniqueIdB=gripper_id, 
                            linkIndexA=ee_idx-1, linkIndexB=-1, 
                            enableCollision=0)
    print(f"🔧 Attached U-Shaped Gripper (Mass: 0.01kg)")

    # --- 4. Update Mplib Planner ---
    # For collision planning, we can simply approximate the gripper as one big box
    # OR add the 3 specific boxes. For simplicity and safety, let's add the full bounding box.
    # This ensures the planner is conservative and won't scrape the "empty space" between fingers against a wall.
    
    full_box_size = [total_thick, total_width, total_len]
    center_offset = [0, 0, total_len/2]
    
    tool_pose = mplib.Pose(p=center_offset, q=[1, 0, 0, 0])
    planner.update_attached_box(size=full_box_size, pose=tool_pose, link_id=-1)
    print("✅ Added Gripper collision box to Mplib planner.")

# ---------------------------------------------------------
# HELPER: GUI Sliders
# ---------------------------------------------------------
def setup_debug_sliders():
    p.addUserDebugText("Left Click to Rotate Camera", [0, 0, 0.7], [0, 0, 0], textSize=1.2)
    sx = p.addUserDebugParameter("Target X", -0.6, 0.6, 0.3)
    sy = p.addUserDebugParameter("Target Y", -0.6, 0.6, 0.1)
    sz = p.addUserDebugParameter("Target Z", 0.0, 0.8, 0.4) 
    sqx = p.addUserDebugParameter("Target Qx", -1.0, 1.0, 1.0)
    sqy = p.addUserDebugParameter("Target Qy", -1.0, 1.0, 0.0)
    sqz = p.addUserDebugParameter("Target Qz", -1.0, 1.0, 0.0)
    sqw = p.addUserDebugParameter("Target Qw", -1.0, 1.0, 0.0)
    btn = p.addUserDebugParameter(">>> PLAN & MOVE <<<", 1, 0, 0)
    return [sx, sy, sz, sqx, sqy, sqz, sqw, btn]

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "ur3_mesh_.urdf") 
    srdf_path = os.path.join(current_dir, "ur3_mplib.srdf")
    
    if not os.path.exists(urdf_path):
        print(f"❌ CRITICAL ERROR: Could not find {urdf_path}")
        return

    generate_srdf(srdf_path)

    print("🔮 Initializing PyBullet Visualization...")
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
    p.resetDebugVisualizerCamera(1.5, 45, -30, [0, 0, 0])

    p.loadURDF("plane.urdf")
    
    # --- OBSTACLES ---
    table_pos  = [0, 0, -0.05]
    table_size = [1.0, 1.0, 0.1]
    t_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in table_size]), 
                               baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in table_size], rgbaColor=[0.6, 0.4, 0.2, 1]), 
                               basePosition=table_pos)
    
    wall_pos  = [0, -0.225, 0.5]
    wall_size = [1.0, 0.05, 1.0]
    w_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in wall_size]), 
                               baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in wall_size], rgbaColor=[0.8, 0.2, 0.2, 0.8]), 
                               basePosition=wall_pos)

    robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=1)
    
    # --- MPLIB SETUP ---
    planner = mplib.Planner(urdf=urdf_path, srdf=srdf_path, move_group="tool0")
    pts_table = get_box_point_cloud(table_pos, table_size)
    pts_wall  = get_box_point_cloud(wall_pos, wall_size)
    planner.update_point_cloud(np.vstack([pts_table, pts_wall]))
    
    # --- ATTACH LIGHTWEIGHT GRIPPER ---
    attach_procedural_gripper(robot_id, "tool0", planner)

    mplib_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    pb_joint_indices = get_pybullet_joint_indices(robot_id, mplib_joint_names)
    
    # Filter out invalid indices
    active_joint_indices = [i for i in pb_joint_indices if i >= 0]

    # Home Pose
    home_qpos = [0, -1.57, 0, -1.57, 0, 0]
    for i, angle in enumerate(home_qpos):
        idx = pb_joint_indices[i]
        if idx >= 0: 
            p.resetJointState(robot_id, idx, angle)

    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------
    sliders = setup_debug_sliders()
    [sid_x, sid_y, sid_z, sid_qx, sid_qy, sid_qz, sid_qw, sid_btn] = sliders
    old_btn_val = p.readUserDebugParameter(sid_btn)
    
    last_mouse_pos = None
    is_rotating = False
    
    # Variable to track where we want the robot to stay
    target_joint_positions = home_qpos 

    print("\n✅ System Ready!")

    while True:
        # --- PHYSICS HOLD LOOP (Fixes "Falling Down") ---
        # PyBullet joints have 0 friction by default. 
        # We must actively command them to stay at 'target_joint_positions'.
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=active_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joint_positions,
            forces=[100.0] * len(active_joint_indices) # Sufficient force to hold
        )
        
        p.stepSimulation()

        # Camera Logic
        events = p.getMouseEvents()
        for e in events:
            if e[0] == 2 and e[3] == 0: 
                if e[4] == 1: is_rotating, last_mouse_pos = True, (e[1], e[2])
                else: is_rotating, last_mouse_pos = False, None
            if e[0] == 1 and is_rotating and last_mouse_pos:
                cam_info = p.getDebugVisualizerCamera()
                dx, dy = e[1] - last_mouse_pos[0], e[2] - last_mouse_pos[1]
                p.resetDebugVisualizerCamera(cam_info[10], cam_info[8] - dx*0.3, cam_info[9] - dy*0.3, cam_info[11])
                last_mouse_pos = (e[1], e[2])

        # Planner Logic
        btn_val = p.readUserDebugParameter(sid_btn)
        if btn_val > old_btn_val:
            old_btn_val = btn_val
            
            tx, ty, tz = [p.readUserDebugParameter(s) for s in [sid_x, sid_y, sid_z]]
            q_raw = np.array([p.readUserDebugParameter(s) for s in [sid_qx, sid_qy, sid_qz, sid_qw]])
            norm = np.linalg.norm(q_raw)
            q_norm = q_raw / norm if norm > 1e-6 else np.array([0.0, 0.0, 0.0, 1.0])
            
            target_pose = mplib.Pose(p=[tx, ty, tz], q=[q_norm[3], q_norm[0], q_norm[1], q_norm[2]])

            # Read current state directly from robot
            current_qpos = [p.getJointState(robot_id, idx)[0] for idx in active_joint_indices]

            print(f"🤔 Planning to: [{tx:.2f}, {ty:.2f}, {tz:.2f}]...")
            result = planner.plan_pose(target_pose, current_qpos, time_step=0.01)
            
            if result["status"] == "Success":
                print("🚀 Path Found!")
                for waypoint in result["position"]:
                    # 1. Move visually/physically
                    # We set the "target" for the physics engine to follow
                    target_joint_positions = waypoint
                    
                    # 2. Teleporting (resetJointState) is smoother for visualization 
                    # but since we added MotorControl, we can just update the target.
                    # However, to keep it snappy like MPLIB usually is, we can do both:
                    for i, angle in enumerate(waypoint):
                        idx = pb_joint_indices[i]
                        if idx >= 0: p.resetJointState(robot_id, idx, angle)
                        
                    p.stepSimulation()
                    time.sleep(0.02)
                
                # Update the holding target to the end of the path
                target_joint_positions = result["position"][-1]
                
            else:
                print(f"❌ Planning Failed: {result['status']}")
                
        time.sleep(0.01)

if __name__ == "__main__":
    main()
