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