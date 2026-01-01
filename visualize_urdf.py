import pybullet as p
import pybullet_data
import time

def main():
    # 1. Start the GUI
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 2. Load the Plane (Floor)
    planeId = p.loadURDF("plane.urdf")

    # 3. Load your Custom UR3 URDF
    # useFixedBase=1 keeps the robot anchored to the world
    try:
        robotId = p.loadURDF("ur3_mesh_.urdf", [0, 0, 0], useFixedBase=1)
    except Exception as e:
        print(f"Error loading URDF: {e}")
        return

    # 4. Get Joint Info and Create Sliders
    num_joints = p.getNumJoints(robotId)
    joint_ids = []
    param_ids = []

    print(f"Loaded robot with {num_joints} joints.")

    for i in range(num_joints):
        info = p.getJointInfo(robotId, i)
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]

        # Only add sliders for movable joints (Revolute/Prismatic)
        if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
            joint_ids.append(i)
            # Add a debug slider for this joint
            # Limits: -3.14 to 3.14 (approx pi)
            param_id = p.addUserDebugParameter(joint_name, -3.14, 3.14, 0)
            param_ids.append(param_id)

    # 5. Simulation Loop
    print("Use the sliders on the right to move the robot!")
    while True:
        # Read slider values and apply to robot
        for i, joint_id in enumerate(joint_ids):
            target_pos = p.readUserDebugParameter(param_ids[i])
            p.resetJointState(robotId, joint_id, target_pos)

        p.stepSimulation()
        time.sleep(1./240.)

if __name__ == "__main__":
    main()