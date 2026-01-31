import torch
import matplotlib.pyplot as plt
import numpy as np
from training.utils_joint_1 import load_data, get_norm_stats

def display_dataset_stats(dataset_path):
    try:
        # get_norm_stats returns a dictionary with 7D arrays (6D arm + 1D gripper)
        stats = get_norm_stats(dataset_path)
        
        labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Gripper']
        
        print("\n" + "="*30)
        print("DATASET NORMALIZATION STATS")
        print("="*30)
        
        # print(f"{'Dimension':<12} | {'Mean':<10} | {'Std Dev':<10}")
        print("-" * 35)
        
        # for i, label in enumerate(labels):
        #     m = stats['action_mean'][i]
        #     s = stats['action_std'][i]
        #     print(f"{label:<12} | {m:>10.4f} | {s:>10.4f}")
        print('action stats')
        a_mean = stats['action_mean']
        a_std = stats['action_std']
        print(f'mean: {a_mean}')
        print(f'std: {a_std}')
        
        q_mean= stats['qpos_mean']
        q_std = stats['qpos_std']
        print('qpos stats')
        print(f'mean: {q_mean}')
        print(f'std: {q_std}')
    
        print("="*30)
        
    except Exception as e:
        print(f"Failed to retrieve stats: {e}")
        
def visualize_sampling(dataset_path, camera_names):
    # 1. Initialize the dataset
    # Setting augment=False for visualization helps confirm if the base colors are correct
    train_loader, _, _, _ = load_data(
        dataset_path=dataset_path,
        batch_size_train=1,
        batch_size_val=1,
        camera_names=camera_names,
        num_queries=10, 
        train_ratio=0.9
    )

    # Get a single sample
    image_data, qpos_data, action_data, is_pad = next(iter(train_loader))

    # Remove batch dimension
    image_data = image_data.squeeze(0) # (Cam, C, H, W)
    qpos_data = qpos_data.squeeze(0).numpy()
    action_data = action_data.squeeze(0).numpy()
    
    num_cams = len(camera_names)
    
    fig = plt.figure(figsize=(5 * num_cams, 8))
    grid = plt.GridSpec(2, num_cams, height_ratios=[2, 1])

    # --- Visualize Images ---
    for i, cam_name in enumerate(camera_names):
        ax = fig.add_subplot(grid[0, i])
        
        # 2. FIX COLOR CHANNELS:
        # permute(1, 2, 0) moves C to the last dimension for Matplotlib
        img = image_data[i].permute(1, 2, 0).numpy()
        
        # Swap BGR to RGB (Flip the last dimension)
        img = np.flip(img, axis=-1) 
        
        # Ensure values are clipped for clean display
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f"Camera: {cam_name}\n(BGR to RGB Corrected)")
        ax.axis('off')

    # --- Visualize Action Sequence ---
    ax_data = fig.add_subplot(grid[1, :])
    labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'Gripper']
    
    for j in range(7):
        # Plot the trajectory of each joint over the 'num_queries' window
        ax_data.plot(action_data[:, j], label=labels[j], marker='o', markersize=3)
    
    ax_data.set_title("Action Sequence (7D)")
    ax_data.set_xlabel("Time Step")
    ax_data.legend(loc='upper right', fontsize='small', ncol=7)
    ax_data.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure these names match your 'observations/images/{cam_name}' keys
    DATASET_PATH = '/mnt/Ego2Exo/pick_teleop_2/realsense/middle_corner_3rd_vertical/reduced/smooth/normalize/final_merged.h5py' 
    CAMERAS = ['cam1_rgb', 'cam2_rgb'] 
    display_dataset_stats(DATASET_PATH)
    visualize_sampling(DATASET_PATH, CAMERAS)
        
        