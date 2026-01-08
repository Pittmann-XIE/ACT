import mplib
import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import argparse
import sys
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="UR3 Teleoperation: Fixed Control Frame")
parser.add_argument('--real', action='store_true', help="Enable control of the real robot via RTDE")
parser.add_argument('--ip', type=str, default="10.0.0.1", help="IP address of the real robot")
parser.add_argument('--step_pos', type=float, default=0.001, help="Position step per loop in meters")
parser.add_argument('--step_rot', type=float, default=0.02, help="Rotation step per loop in radians")
parser.add_argument('--reset_time', type=float, default=3.0, help="Time in seconds to perform the reset move")

args = parser.parse_args()

# ---------------------------------------------------------
# RTDE IMPORT (Conditional)
# ---------------------------------------------------------
rtde_c = None
rtde_r = None

if args.real:
    try:
        import rtde_control
        import rtde_receive
    except ImportError:
        print("\n[ERROR] You requested --real control, but 'ur_rtde' is not installed.")
        print("Please install it using: pip install ur_rtde")
        sys.exit(1)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Default Home Configuration
INIT_JOINTS_DEG = [46, -49, -142, -9.0, -277, 2.5]
SIM_ROBOT_WORLD_ROT_Z = np.pi 

# Obstacles
WALL_POS_WORLD = [-0.14, -0.14, 0.5]  
WALL_ROT_Z_DEG = 45  
WALL_SIZE = [0.02, 1.0, 1.0] 
TABLE_SIZE = [1.0, 1.0, 0.1]
TABLE_POS_WORLD = [0, 0, -TABLE_SIZE[2]/2-0.01]
TABLE_ROT_Z_DEG = 0

# Control Loop
LOOP_FREQUENCY = 100 # Hz
DT = 1.0 / LOOP_FREQUENCY

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
    # RGB = XYZ
    p.addUserDebugLine(pos, pos + rot[:,0]*axis_len, [1,0,0], lineWidth=2, lifeTime=life_time) 
    p.addUserDebugLine(pos, pos + rot[:,1]*axis_len, [0,1,0], lineWidth=2, lifeTime=life_time) 
    p.addUserDebugLine(pos, pos + rot[:,2]*axis_len, [0,0,1], lineWidth=2, lifeTime=life_time) 

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    global rtde_c, rtde_r

    # 1. Real Robot Connection
    if args.real:
        print(f"--- Connecting to Real Robot at {args.ip} ---")
        try:
            rtde_c = rtde_control.RTDEControlInterface(args.ip)
            rtde_r = rtde_receive.RTDEReceiveInterface(args.ip)
            print("✅ Connected to Real Robot.")
        except Exception as e:
            print(f"❌ Connection Failed: {e}")
            sys.exit(1)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "ur3_gripper.urdf") 
    srdf_path = os.path.join(current_dir, "ur3_gripper.srdf")
    
    # 2. PyBullet Setup
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(1.2, 90, -30, [0, 0, 0])
    p.loadURDF("plane.urdf")    

    # 3. Obstacles
    wall_pos_r, wall_quat_r, wall_pos_w, wall_quat_w = get_object_transforms(WALL_POS_WORLD, WALL_ROT_Z_DEG)
    table_pos_r, table_quat_r, table_pos_w, table_quat_w = get_object_transforms(TABLE_POS_WORLD, TABLE_ROT_Z_DEG)

    p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE]), 
                      basePosition=wall_pos_w, baseOrientation=wall_quat_w,
                      baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE], rgbaColor=[0.8, 0.2, 0.2, 0.8]))
    
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE]), 
                      basePosition=table_pos_w, baseOrientation=table_quat_w,
                      baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE], rgbaColor=[0.5, 0.3, 0.1, 1.0]))

    # 4. Robot Setup
    sim_robot_orn = p.getQuaternionFromEuler([0, 0, SIM_ROBOT_WORLD_ROT_Z])
    robot_id = p.loadURDF(urdf_path, [0, 0, 0], baseOrientation=sim_robot_orn, useFixedBase=1)
    
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    name_to_idx = {p.getJointInfo(robot_id, i)[1].decode('utf-8'): i for i in range(p.getNumJoints(robot_id))}
    active_indices = [name_to_idx.get(n, -1) for n in joint_names]

    ee_link_idx = -1
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[12].decode("utf-8") == "tool0": 
            ee_link_idx = i; break
    if ee_link_idx == -1: raise ValueError("Could not find link 'tool0'")

    # 5. Initialization
    home_q_rad = np.deg2rad(INIT_JOINTS_DEG).tolist()
    
    # Get Home EE Transform by momentarily setting simulation
    for i, angle in enumerate(home_q_rad):
        p.resetJointState(robot_id, active_indices[i], angle)
    p.stepSimulation()
    
    home_ee_state = p.getLinkState(robot_id, ee_link_idx)
    home_pos = list(home_ee_state[4])
    home_orn = list(home_ee_state[5])

    # Real Robot Startup Sync
    if args.real:
        print("Reading actual joints from robot...")
        actual_q = rtde_r.getActualQ()
        for i, angle in enumerate(actual_q):
            p.resetJointState(robot_id, active_indices[i], angle)
        p.stepSimulation()
        start_ee = p.getLinkState(robot_id, ee_link_idx)
        target_pos = list(start_ee[4])
        target_orn = list(start_ee[5])
        
        p.setJointMotorControlArray(robot_id, active_indices, p.POSITION_CONTROL, 
                                    targetPositions=actual_q, forces=[200.0]*6)
    else:
        target_pos = home_pos
        target_orn = home_orn
        p.setJointMotorControlArray(robot_id, active_indices, p.POSITION_CONTROL, 
                                    targetPositions=home_q_rad, forces=[200.0]*6)

    # 6. MPLib Planner
    planner = mplib.Planner(urdf=urdf_path, srdf=srdf_path, move_group="tool0")
    combined_pts = np.vstack([
        get_box_point_cloud(wall_pos_r, WALL_SIZE, wall_quat_r), 
        get_box_point_cloud(table_pos_r, TABLE_SIZE, table_quat_r)
    ])
    planner.update_point_cloud(combined_pts)

    mode_str = "REAL ROBOT CONNECTED" if args.real else "SIMULATION ONLY"
    print(f"\n✅ Teleoperation Ready! [{mode_str}]")
    print(f"   Pos Step: {args.step_pos}m | Rot Step: {args.step_rot}rad")
    print("------------------------------------------------")
    print(" [UP / DOWN]    : Move X (Forward / Back)")
    print(" [LEFT / RIGHT] : Move Y (Left / Right)")
    print(" [A / Y]        : Move Z (Up / Down)")
    print(" --- TOOL FRAME ROTATIONS ---")
    print(" [1 / 2]        : Yaw (Rotate around Tool Z)")
    print(" [3 / 4]        : Roll (Rotate around Tool X)")
    print(" [5 / 6]        : Pitch (Rotate around Tool Y)")
    print(" [R]            : SMOOTH RESET to Home")
    print("------------------------------------------------")

    while True:
        start_time = time.time()
        keys = p.getKeyboardEvents()
        
        # --- SMOOTH RESET FUNCTIONALITY ---
        if ord('r') in keys and keys[ord('r')] & p.KEY_IS_DOWN:
            print(f"🔄 Smooth Resetting to Home ({args.reset_time}s)...")
            current_joint_states = p.getJointStates(robot_id, active_indices)
            start_q = np.array([state[0] for state in current_joint_states])
            end_q = np.array(home_q_rad)
            steps = int(args.reset_time * LOOP_FREQUENCY)
            for i in range(steps):
                loop_start = time.time()
                alpha = (i + 1) / steps
                interp_q = start_q + alpha * (end_q - start_q)
                interp_q_list = interp_q.tolist()
                
                p.setJointMotorControlArray(robot_id, active_indices, p.POSITION_CONTROL, 
                                            targetPositions=interp_q_list, forces=[200.0]*6)
                p.stepSimulation()
                if args.real and rtde_c: rtde_c.servoJ(interp_q_list, 0.0, 0.0, DT, 0.1, 300)
                
                temp_ee = p.getLinkState(robot_id, ee_link_idx)
                draw_coordinate_frame(temp_ee[4], temp_ee[5], life_time=DT*2)
                
                elapsed = time.time() - loop_start
                if elapsed < DT: time.sleep(DT - elapsed)
            
            target_pos = list(home_pos)
            target_orn = list(home_orn)
            print("✅ Reset Complete.")
            continue

        # --- Handle Inputs ---
        delta_pos = np.zeros(3)
        delta_rot = np.zeros(3) # [Roll (X), Pitch (Y), Yaw (Z)]

        # 1. FIXED TRANSLATION (Robot Base Frame)
        # UP = +X (Forward), DOWN = -X (Back)
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN: delta_pos[0] += args.step_pos 
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN: delta_pos[0] -= args.step_pos 
        
        # LEFT = +Y (Left), RIGHT = -Y (Right)
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN: delta_pos[1] += args.step_pos
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN: delta_pos[1] -= args.step_pos
        
        # Z Axis (Global)
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN: delta_pos[2] += args.step_pos
        if ord('y') in keys and keys[ord('y')] & p.KEY_IS_DOWN: delta_pos[2] -= args.step_pos

        # 2. FIXED ROTATION (Intrinsic / Tool Frame)
        # Yaw (Around Tool Z)
        if ord('1') in keys and keys[ord('1')] & p.KEY_IS_DOWN: delta_rot[2] += args.step_rot 
        if ord('2') in keys and keys[ord('2')] & p.KEY_IS_DOWN: delta_rot[2] -= args.step_rot
        
        # Roll (Around Tool X)
        if ord('3') in keys and keys[ord('3')] & p.KEY_IS_DOWN: delta_rot[0] += args.step_rot 
        if ord('4') in keys and keys[ord('4')] & p.KEY_IS_DOWN: delta_rot[0] -= args.step_rot
        
        # Pitch (Around Tool Y)
        if ord('5') in keys and keys[ord('5')] & p.KEY_IS_DOWN: delta_rot[1] += args.step_rot 
        if ord('6') in keys and keys[ord('6')] & p.KEY_IS_DOWN: delta_rot[1] -= args.step_rot

        # --- Calculate IK & Check Collision ---
        if np.linalg.norm(delta_pos) > 0 or np.linalg.norm(delta_rot) > 0:
            
            # Apply Translation
            temp_pos = np.array(target_pos) + delta_pos
            
            # Apply Intrinsic Rotation (Quaternion multiplication)
            # R_new = R_current * R_delta
            r_current = R.from_quat(target_orn)
            r_delta = R.from_euler('xyz', delta_rot) # Intrinsic Euler (Extrinsic would be R_delta * R_current)
            
            r_new = r_current * r_delta
            temp_quat = r_new.as_quat()

            ik_solution = p.calculateInverseKinematics(
                robot_id, ee_link_idx, targetPosition=temp_pos, targetOrientation=temp_quat,
                maxNumIterations=50, residualThreshold=1e-4
            )
            q_proposal = list(ik_solution[:6]) 
            
            is_self_collision = planner.check_for_self_collision(q_proposal)
            is_env_collision = planner.check_for_env_collision(q_proposal)
            
            if is_self_collision or is_env_collision:
                p.addUserDebugText("COLLISION!", [0,0,1], [1,0,0], lifeTime=0.1, textSize=2)
            else:
                target_pos = temp_pos
                target_orn = temp_quat
                
                # Apply to Simulation
                p.setJointMotorControlArray(robot_id, active_indices, p.POSITION_CONTROL, 
                                            targetPositions=q_proposal, forces=[200.0]*6)
                
                # Apply to Real Robot
                if args.real and rtde_c:
                    rtde_c.servoJ(q_proposal, 0.0, 0.0, DT, 0.1, 300)
        
        # --- Visualization ---
        draw_coordinate_frame(target_pos, target_orn, life_time=0.1)
        
        p.stepSimulation()
        
        elapsed = time.time() - start_time
        if elapsed < DT:
            time.sleep(DT - elapsed)

if __name__ == "__main__":
    main()