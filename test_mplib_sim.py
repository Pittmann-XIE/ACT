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
INIT_JOINTS_DEG = [45.0, -52.39, -100, -90, -266.31, 0]
SIM_ROBOT_WORLD_ROT_Z = np.pi 

# Obstacles (Wall & Table)
WALL_POS_WORLD = [-0.14, -0.14, 0.5]  
WALL_ROT_Z_DEG = 45  
WALL_SIZE = [0.02, 1.0, 1.0] 
TABLE_SIZE = [1.0, 1.0, 0.1]
TABLE_POS_WORLD = [0, 0, -TABLE_SIZE[2]/2-0.01]
TABLE_ROT_Z_DEG = 0

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
    rot = R.from_quat(quat).as_matrix()
    p.addUserDebugLine(pos, pos + rot[:,0]*axis_len, [1,0,0], lineWidth=line_width, lifeTime=life_time) 
    p.addUserDebugLine(pos, pos + rot[:,1]*axis_len, [0,1,0], lineWidth=line_width, lifeTime=life_time) 
    p.addUserDebugLine(pos, pos + rot[:,2]*axis_len, [0,0,1], lineWidth=line_width, lifeTime=life_time) 

_debug_frame_ids = {}

def debug_draw_all_robot_frames(robot_id, axis_len=0.1):
    global _debug_frame_ids
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        link_name = info[12].decode("utf-8")
        if link_name != "tool0": continue

        state = p.getLinkState(robot_id, i)
        pos, orn = state[4], state[5]
        rot = R.from_quat(orn).as_matrix()
        
        items = [("x", pos + rot[:,0] * axis_len, [1, 0, 0]), 
                 ("y", pos + rot[:,1] * axis_len, [0, 1, 0]), 
                 ("z", pos + rot[:,2] * axis_len, [0, 0, 1])]

        for suffix, target, color in items:
            key = f"joint_{i}_{suffix}"
            if key in _debug_frame_ids:
                p.addUserDebugLine(pos, target, color, lineWidth=2, replaceItemUniqueId=_debug_frame_ids[key], lifeTime=0)
            else:
                _debug_frame_ids[key] = p.addUserDebugLine(pos, target, color, lineWidth=2, lifeTime=0)

        text_key = f"joint_{i}_text"
        text_pos = [pos[0], pos[1], pos[2] + 0.02]
        if text_key in _debug_frame_ids:
            p.addUserDebugText(link_name, text_pos, [0, 0, 0], textSize=1.0, replaceItemUniqueId=_debug_frame_ids[text_key], lifeTime=0)
        else:
            _debug_frame_ids[text_key] = p.addUserDebugText(link_name, text_pos, [0, 0, 0], textSize=1.0, lifeTime=0)
            
def setup_debug_sliders():
    sx = p.addUserDebugParameter("Target X", -0.6, 0.6, 0.3)
    sy = p.addUserDebugParameter("Target Y", -0.6, 0.6, 0.1)
    sz = p.addUserDebugParameter("Target Z", 0.0, 0.8, 0.4) 
    sr = p.addUserDebugParameter("Roll (Deg)", -180, 180, 180) 
    sp = p.addUserDebugParameter("Pitch (Deg)", -180, 180, 0)
    sya = p.addUserDebugParameter("Yaw (Deg)", -180, 180, 0)
    btn_plan = p.addUserDebugParameter("[1] PLAN", 1, 0, 0)
    btn_exec = p.addUserDebugParameter("[2] EXECUTE", 1, 0, 0)
    return [sx, sy, sz, sr, sp, sya, btn_plan, btn_exec]

def execute_trajectory_sim(robot_id, active_indices, path, speed=0.02):
    """Interpolates and moves the robot in simulation"""
    for waypoint in path:
        p.setJointMotorControlArray(robot_id, active_indices, p.POSITION_CONTROL, targetPositions=waypoint, forces=[200.0]*6)
        for _ in range(5):
            p.stepSimulation()
            time.sleep(0.002) 
        time.sleep(speed)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "ur3_gripper.urdf") 
    srdf_path = os.path.join(current_dir, "ur3_gripper.srdf")
    
    # PyBullet Setup
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0) 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # ✅ ZERO GRAVITY
    # p.setGravity(0, 0, 0)
    
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

    # Robot Setup
    sim_robot_orn = p.getQuaternionFromEuler([0, 0, SIM_ROBOT_WORLD_ROT_Z])
    robot_id = p.loadURDF(urdf_path, [0, 0, 0], baseOrientation=sim_robot_orn, useFixedBase=1)
    ghost_id = load_ghost_robot(urdf_path, [0, 0, 0], sim_robot_orn)
    
    # Initialize Robot Pose
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    name_to_idx = {p.getJointInfo(robot_id, i)[1].decode('utf-8'): i for i in range(p.getNumJoints(robot_id))}
    active_indices = [name_to_idx.get(n, -1) for n in joint_names]

    init_q_rad = np.deg2rad(INIT_JOINTS_DEG).tolist()
    for i, angle in enumerate(init_q_rad):
        p.resetJointState(robot_id, active_indices[i], angle)
    
    # Store initial pose as the first target
    active_target = init_q_rad 
    
    # Hide Ghost initially
    p.resetBasePositionAndOrientation(ghost_id, [0,0,-5], [0,0,0,1]) 

    # MPLib Planner Setup
    planner = mplib.Planner(urdf=urdf_path, srdf=srdf_path, move_group="tool0")
    
    # Add environment to planner
    combined_pts = np.vstack([
        get_box_point_cloud(wall_pos_r, WALL_SIZE, wall_quat_r, resolution=0.04), 
        get_box_point_cloud(table_pos_r, TABLE_SIZE, table_quat_r, resolution=0.04)
    ])
    planner.update_point_cloud(combined_pts)
    visualize_point_cloud_in_world(combined_pts)

    # Find the end-effector link index
    ee_link_idx = -1
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[12].decode("utf-8") == "tool0": 
            ee_link_idx = i; break

    # UI Controls
    sliders = setup_debug_sliders()
    [sid_x, sid_y, sid_z, sid_r, sid_p, sid_yaw, sid_plan, sid_exec] = sliders
    old_btn_plan, old_btn_exec = 0, 0
    pending_path = None
    
    last_log_time = time.time()
    
    print("\n✅ Simulation Ready!")
    print("👉 GREEN GHOST = Target is Valid")
    print("👉 RED GHOST   = Target is in Collision (DO NOT PLAN)")

    # ---------------------------------------------------------
    # LOOP
    # ---------------------------------------------------------
    while True:
        # Read current state (needed for planner start state)
        current_joint_states = [p.getJointState(robot_id, idx)[0] for idx in active_indices]
        
        # ✅ LOGGING (Every 0.5 seconds)
        if time.time() - last_log_time > 0.5:
            deg_values = np.rad2deg(current_joint_states)
            print(f"Joints (Deg): {np.array2string(deg_values, formatter={'float_kind':lambda x: '%.2f' % x})}")
            last_log_time = time.time()

        # Apply Motor Control (Using Fixed Target)
        p.setJointMotorControlArray(robot_id, active_indices, p.POSITION_CONTROL, 
                                    targetPositions=active_target, forces=[200.0]*6)
        p.stepSimulation()
        
        debug_draw_all_robot_frames(robot_id, axis_len=0.1)
        
        # UI: Update Target Frame
        tx, ty, tz = [p.readUserDebugParameter(s) for s in [sid_x, sid_y, sid_z]]
        rr, rp, ry = [p.readUserDebugParameter(s) for s in [sid_r, sid_p, sid_yaw]]
        q_norm_world = p.getQuaternionFromEuler(np.deg2rad([rr, rp, ry])) 
        draw_coordinate_frame([tx, ty, tz], q_norm_world, life_time=0.05)

        # Visual Debug: Force Ghost to Target
        if ee_link_idx != -1:
            p.resetBasePositionAndOrientation(ghost_id, [0, 0, 0], sim_robot_orn)
            ik_solution = p.calculateInverseKinematics(
                robot_id, ee_link_idx, targetPosition=[tx, ty, tz], targetOrientation=q_norm_world,
                maxNumIterations=50, residualThreshold=1e-4
            )
            ghost_config = ik_solution[:6] 
            set_ghost_config(ghost_id, active_indices, ghost_config)

            is_ghost_safe = not planner.check_for_self_collision(list(ghost_config)) and \
                            not planner.check_for_env_collision(list(ghost_config))
            
            color = [0, 1, 0] if is_ghost_safe else [1, 0, 0]
            p.addUserDebugText(f"{'SAFE' if is_ghost_safe else 'COLLISION'}", 
                            [0, 0, 1.1], color, textSize=2, lifeTime=0.1)
        else:
             is_ghost_safe = False

        # --- PLAN (Button 1) ---
        if p.readUserDebugParameter(sid_plan) > old_btn_plan:
            old_btn_plan = p.readUserDebugParameter(sid_plan)
            print("\n🔍 --- DIAGNOSTICS ---")
            
            # Check Start State
            start_self_col = planner.check_for_self_collision(current_joint_states)
            start_env_col = planner.check_for_env_collision(current_joint_states)
            print(f"📍 Start State Self-Collision: {start_self_col}")
            print(f"📍 Start State Env-Collision:  {start_env_col}")

            if start_self_col or start_env_col:
                print("❌ BLOCKER: The robot is ALREADY in collision!")

            # Check Goal State
            goal_list = list(ghost_config) 
            goal_self_col = planner.check_for_self_collision(goal_list)
            goal_env_col = planner.check_for_env_collision(goal_list)
            print(f"🏁 Goal State Self-Collision:  {goal_self_col}")
            print(f"🏁 Goal State Env-Collision:   {goal_env_col}")

            if not is_ghost_safe:
                print("❌ CANNOT PLAN: Ghost is in collision.")
            else:
                print(f"🤔 Planning to Ghost Configuration...")
                result = planner.plan_qpos([np.array(ghost_config)], current_joint_states, time_step=0.01)

                if result["status"] == "Success":
                    print("✅ Path Found!")
                    pending_path = result["position"]
                else:
                    print(f"❌ Planning Failed: {result['status']}")

        # --- EXECUTE (Button 2) ---
        if p.readUserDebugParameter(sid_exec) > old_btn_exec:
            old_btn_exec = p.readUserDebugParameter(sid_exec)
            if pending_path is not None:
                execute_trajectory_sim(robot_id, active_indices, pending_path)
                active_target = pending_path[-1] # Update holding position
                pending_path = None
            else:
                print("⚠️ No plan generated.")

        time.sleep(0.01) 

if __name__ == "__main__":
    main()