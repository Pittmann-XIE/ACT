# import torch
# import numpy as np
# import os
# import argparse
# import matplotlib.pyplot as plt
# import sys
# from scipy.spatial.transform import Rotation as R

# # --- Import Project Modules ---
# sys.path.append(os.getcwd())
# from config.config_1 import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
# from training.utils_1 import make_policy, get_norm_stats, load_data

# # --- Math Utilities ---

# def rot6d_to_matrix(rot_6d):
#     """
#     Convert 6D rotation to 3x3 matrix.
#     Args: rot_6d: (..., 6)
#     Returns: rot_mat: (..., 3, 3)
#     """
#     x_raw = rot_6d[..., 0:3]
#     y_raw = rot_6d[..., 3:6]
#     x = torch.nn.functional.normalize(x_raw, dim=-1)
#     z = torch.cross(x, y_raw, dim=-1)
#     z = torch.nn.functional.normalize(z, dim=-1)
#     y = torch.cross(z, x, dim=-1)
#     return torch.stack([x, y, z], dim=-1)

# def mat_to_quat(rot_mat):
#     """
#     Convert 3x3 matrix to quaternion (x, y, z, w).
#     Args: rot_mat: (Batch, Seq, 3, 3)
#     Returns: quat: (Batch, Seq, 4)
#     """
#     # Flatten batch and sequence dims for scipy
#     batch_shape = rot_mat.shape[:-2] 
#     flat_mat = rot_mat.reshape(-1, 3, 3).detach().cpu().numpy()
    
#     r = R.from_matrix(flat_mat)
#     quat = r.as_quat()
    
#     return quat.reshape(*batch_shape, 4)

# # --- Main Inference Class ---

# class ACTInference:
#     def __init__(self, checkpoint_path, policy_config, task_config):
#         self.device = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Load Normalization Stats
#         dataset_dir = task_config['dataset_dir']
#         stats = get_norm_stats(dataset_dir)
#         self.tx_mean, self.tx_std = stats['tx']['mean'], stats['tx']['std']
#         self.ty_mean, self.ty_std = stats['ty']['mean'], stats['ty']['std']
#         self.tz_mean, self.tz_std = stats['tz']['mean'], stats['tz']['std']
        
#         # Load Model
#         print(f"Loading model from: {checkpoint_path}")
#         self.policy = make_policy(policy_config['policy_class'], policy_config)
#         state_dict = torch.load(checkpoint_path, map_location=self.device)
#         self.policy.load_state_dict(state_dict)
#         self.policy.to(self.device)
#         self.policy.eval()

#     def _denormalize_pos(self, norm_pos_array):
#         """
#         Broadcasting denormalization.
#         Args: norm_pos_array (Batch, Seq, 3)
#         Returns: Real world positions (Batch, Seq, 3)
#         """
#         x = norm_pos_array[..., 0] * self.tx_std + self.tx_mean
#         y = norm_pos_array[..., 1] * self.ty_std + self.ty_mean
#         z = norm_pos_array[..., 2] * self.tz_std + self.tz_mean
#         return np.stack([x, y, z], axis=-1)

#     def predict_batch(self, image_tensor, qpos_tensor):
#         """
#         Process a full validation batch.
#         Args:
#             image_tensor: (B, Num_Cameras, C, H, W) - Must have camera dimension!
#             qpos_tensor:  (B, 10) [Already Normalized from Dataloader]
#         """
#         image_tensor = image_tensor.to(self.device)
#         qpos_tensor = qpos_tensor.to(self.device)

#         with torch.inference_mode():
#             # Model Output: (B, Num_Queries, 10)
#             action_pred_raw = self.policy(qpos_tensor, image_tensor)
        
#         # --- Post Processing (Batch Wise) ---
        
#         # 1. Position: (B, Q, 3)
#         pred_pos_norm = action_pred_raw[..., :3].cpu().numpy()
#         pred_pos_real = self._denormalize_pos(pred_pos_norm)
        
#         # 2. Rotation: (B, Q, 6) -> (B, Q, 4)
#         pred_rot6d = action_pred_raw[..., 3:9]
#         rot_mat = rot6d_to_matrix(pred_rot6d) # (B, Q, 3, 3)
#         pred_quat = mat_to_quat(rot_mat)      # (B, Q, 4)
        
#         # 3. Gripper: (B, Q, 1)
#         # Convert Probability [0,1] -> Binary {0, 1}
#         pred_gripper_prob = action_pred_raw[..., 9:]
#         pred_gripper = (pred_gripper_prob > 0.5).float().cpu().numpy()

#         return {
#             'pos': pred_pos_real,
#             'quat': pred_quat,
#             'gripper': pred_gripper
#         }

# # --- Visualizer ---

# def visualize_batch(images, gt_actions, pred_actions):
#     """
#     Plots a grid of Ground Truth vs Prediction for the whole batch.
#     Args:
#         images: (B, H, W, C) numpy array
#         gt_actions: Dictionary {'pos': ..., 'quat': ..., 'gripper': ...}
#         pred_actions: Dictionary {'pos': ..., 'quat': ..., 'gripper': ...}
#     """
#     batch_size = images.shape[0]
    
#     # Create a figure with BatchSize rows and 4 columns (Image, Pos, Quat, Gripper)
#     # Increased width to accommodate the extra column
#     fig, axs = plt.subplots(batch_size, 4, figsize=(20, 3 * batch_size))
#     if batch_size == 1: axs = axs[None, :] # Handle single batch case
    
#     for b in range(batch_size):
#         # --- Column 1: Image ---
#         axs[b, 0].imshow(images[b])
#         axs[b, 0].set_title(f"Batch {b}: Input")
#         axs[b, 0].axis('off')
        
#         # --- Column 2: Position Trajectory ---
#         labels = ['x', 'y', 'z']
#         colors = ['r', 'g', 'b']
        
#         for i in range(3):
#             # Plot GT
#             axs[b, 1].plot(gt_actions['pos'][b, :, i], label=f'GT {labels[i]}', color=colors[i], alpha=0.3)
#             # Plot Pred
#             axs[b, 1].plot(pred_actions['pos'][b, :, i], label=f'Pred {labels[i]}', color=colors[i], linestyle='--')
        
#         axs[b, 1].set_title(f"Pos Trajectory")
#         if b == 0: axs[b, 1].legend() # Only legend on top row
#         axs[b, 1].grid(True, alpha=0.3)

#         # --- Column 3: Quaternion Trajectory (New) ---
#         labels_quat = ['qx', 'qy', 'qz', 'qw']
#         colors_quat = ['r', 'g', 'b', 'c'] # Red, Green, Blue, Cyan
        
#         for i in range(4):
#             # Plot GT
#             axs[b, 2].plot(gt_actions['quat'][b, :, i], label=f'GT {labels_quat[i]}', color=colors_quat[i], alpha=0.3)
#             # Plot Pred
#             axs[b, 2].plot(pred_actions['quat'][b, :, i], label=f'Pred {labels_quat[i]}', color=colors_quat[i], linestyle='--')
            
#         axs[b, 2].set_title(f"Quat Trajectory")
#         if b == 0: axs[b, 2].legend()
#         axs[b, 2].grid(True, alpha=0.3)
        
#         # --- Column 4: Gripper ---
#         axs[b, 3].plot(gt_actions['gripper'][b, :], label='GT', color='k', alpha=0.5)
#         axs[b, 3].plot(pred_actions['gripper'][b, :], label='Pred', color='orange', linestyle='--')
#         axs[b, 3].set_title(f"Gripper State")
#         axs[b, 3].set_ylim(-0.1, 1.1)
#         if b == 0: axs[b, 3].legend()

#     plt.tight_layout()
#     plt.show()

# # --- Main Execution ---

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--task', type=str, default='pick', help='Task name')
#     parser.add_argument('--ckpt', type=str, default='policy_best.ckpt', help='Checkpoint filename')
#     parser.add_argument('--batch_size', type=int, default=8, help='Batch size for validation')
#     args = parser.parse_args()

#     # Define paths
#     ckpt_dir = os.path.join(TRAIN_CONFIG['checkpoint_dir'], args.task)
#     ckpt_path = os.path.join(ckpt_dir, args.ckpt)

#     try:
#         engine = ACTInference(ckpt_path, POLICY_CONFIG, TASK_CONFIG)
#     except Exception as e:
#         print(f"Error initializing engine: {e}")
#         return

#     print(f"\n--- Validating Batch Size: {args.batch_size} ---")
    
#     # 1. Load Data
#     data_dir = TASK_CONFIG['dataset_dir']
#     _, val_dataloader, _, _ = load_data(
#         data_dir, 
#         batch_size_train=8, 
#         batch_size_val=args.batch_size, 
#         num_queries=POLICY_CONFIG['num_queries']
#     )
    
#     # Get one batch
#     data_iter = iter(val_dataloader)
#     image_data, qpos_data, action_data, is_pad = next(data_iter)
    
#     # 2. Predict Batch
#     print("Running Inference on Batch...")
    
#     # Take the first image of the sequence
#     curr_image_tensor = image_data[:, 0, :, :, :]
#     # UNSQUEEZE to add "Camera" Dimension: (B, 1, C, H, W)
#     curr_image_tensor = curr_image_tensor.unsqueeze(1)
    
#     preds = engine.predict_batch(curr_image_tensor, qpos_data)
    
#     # 3. Process Ground Truth
#     print("Processing Ground Truth...")
    
#     # A. Denormalize Position
#     gt_pos_norm = action_data[..., :3].cpu().numpy()
#     gt_pos_real = engine._denormalize_pos(gt_pos_norm)
    
#     # B. Convert Rot6D to Quaternion for Visualization
#     gt_rot6d = action_data[..., 3:9] # Tensor
#     gt_rot_mat = rot6d_to_matrix(gt_rot6d) # Tensor (B, Seq, 3, 3)
#     gt_quat = mat_to_quat(gt_rot_mat) # Numpy (B, Seq, 4)
    
#     # C. Gripper (Keep as is, typically 0 or 1)
#     gt_gripper = action_data[..., 9:].cpu().numpy()
    
#     gt_dict = {
#         'pos': gt_pos_real,
#         'quat': gt_quat,
#         'gripper': gt_gripper
#     }

#     # 4. Prepare Images for Visualization
#     vis_images_tensor = curr_image_tensor.squeeze(1) 
#     vis_images = vis_images_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255.0
#     vis_images = vis_images.astype(np.uint8)

#     # 5. Visualize
#     print(f"Visualizing {args.batch_size} samples...")
#     visualize_batch(vis_images, gt_dict, preds)

# if __name__ == '__main__':
#     main()


##
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import sys
from scipy.spatial.transform import Rotation as R

# --- Import Project Modules ---
sys.path.append(os.getcwd())
from config.config_1 import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from training.utils_1 import make_policy, get_norm_stats, load_data

# --- Math Utilities ---

def rot6d_to_matrix(rot_6d):
    """Convert 6D rotation to 3x3 matrix."""
    x_raw = rot_6d[..., 0:3]
    y_raw = rot_6d[..., 3:6]
    x = torch.nn.functional.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = torch.nn.functional.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)

def mat_to_quat(rot_mat):
    """Convert 3x3 matrix to quaternion (x, y, z, w)."""
    batch_shape = rot_mat.shape[:-2] 
    flat_mat = rot_mat.reshape(-1, 3, 3).detach().cpu().numpy()
    r = R.from_matrix(flat_mat)
    quat = r.as_quat()
    return quat.reshape(*batch_shape, 4)

# --- Main Inference Class ---

class ACTInference:
    def __init__(self, checkpoint_path, policy_config, task_config):
        self.device = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Normalization Stats
        dataset_dir = task_config['dataset_dir']
        stats = get_norm_stats(dataset_dir)
        self.tx_mean, self.tx_std = stats['tx']['mean'], stats['tx']['std']
        self.ty_mean, self.ty_std = stats['ty']['mean'], stats['ty']['std']
        self.tz_mean, self.tz_std = stats['tz']['mean'], stats['tz']['std']
        
        # Load Model
        print(f"Loading model from: {checkpoint_path}")
        self.policy = make_policy(policy_config['policy_class'], policy_config)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.policy.to(self.device)
        self.policy.eval()

    def _denormalize_pos(self, norm_pos_array):
        """Z-Score -> Real World Meters"""
        x = norm_pos_array[..., 0] * self.tx_std + self.tx_mean
        y = norm_pos_array[..., 1] * self.ty_std + self.ty_mean
        z = norm_pos_array[..., 2] * self.tz_std + self.tz_mean
        return np.stack([x, y, z], axis=-1)

    def predict_batch(self, image_tensor, qpos_tensor):
        image_tensor = image_tensor.to(self.device)
        qpos_tensor = qpos_tensor.to(self.device)

        with torch.inference_mode():
            action_pred_raw = self.policy(qpos_tensor, image_tensor)
        
        # 1. Position
        pred_pos_norm = action_pred_raw[..., :3].cpu().numpy()
        pred_pos_real = self._denormalize_pos(pred_pos_norm)
        
        # 2. Rotation
        pred_rot6d = action_pred_raw[..., 3:9]
        rot_mat = rot6d_to_matrix(pred_rot6d) 
        pred_quat = mat_to_quat(rot_mat)      
        
        # 3. Gripper
        pred_gripper_prob = action_pred_raw[..., 9:]
        pred_gripper = (pred_gripper_prob > 0.5).float().cpu().numpy()

        return {
            'pos': pred_pos_real,
            'quat': pred_quat,
            'gripper': pred_gripper
        }

# --- Visualizer ---

def visualize_batch(images, gt_actions, pred_actions):
    """
    Returns: bool (True for Next, False for Quit)
    """
    batch_size = images.shape[0]
    
    fig, axs = plt.subplots(batch_size, 4, figsize=(20, max(4, 3 * batch_size)))
    if batch_size == 1: axs = axs[None, :] 
    
    for b in range(batch_size):
        # 1. Image
        axs[b, 0].imshow(images[b])
        axs[b, 0].set_title(f"Batch {b}: Input")
        axs[b, 0].axis('off')
        
        # 2. Position
        labels = ['x', 'y', 'z']
        colors = ['r', 'g', 'b']
        for i in range(3):
            axs[b, 1].plot(gt_actions['pos'][b, :, i], label=f'GT {labels[i]}', color=colors[i], alpha=0.3)
            axs[b, 1].plot(pred_actions['pos'][b, :, i], label=f'Pred {labels[i]}', color=colors[i], linestyle='--')
        axs[b, 1].set_title(f"Pos Trajectory")
        if b == 0: axs[b, 1].legend() 
        axs[b, 1].grid(True, alpha=0.3)

        # 3. Quaternion
        labels_quat = ['qx', 'qy', 'qz', 'qw']
        colors_quat = ['r', 'g', 'b', 'c'] 
        for i in range(4):
            axs[b, 2].plot(gt_actions['quat'][b, :, i], label=f'GT {labels_quat[i]}', color=colors_quat[i], alpha=0.3)
            axs[b, 2].plot(pred_actions['quat'][b, :, i], label=f'Pred {labels_quat[i]}', color=colors_quat[i], linestyle='--')
        axs[b, 2].set_title(f"Quat Trajectory")
        if b == 0: axs[b, 2].legend()
        axs[b, 2].grid(True, alpha=0.3)
        
        # 4. Gripper
        axs[b, 3].plot(gt_actions['gripper'][b, :], label='GT', color='k', alpha=0.5)
        axs[b, 3].plot(pred_actions['gripper'][b, :], label='Pred', color='orange', linestyle='--')
        axs[b, 3].set_title(f"Gripper State")
        axs[b, 3].set_ylim(-0.1, 1.1)
        if b == 0: axs[b, 3].legend()

    fig.suptitle(f"Inference Results (Batch Size: {batch_size}) - Press 'n' for Next, 'q' to Quit", fontsize=16)
    plt.tight_layout()

    user_action = {'next': False}
    def on_key(event):
        if event.key == 'n':
            print("Loading next batch...")
            user_action['next'] = True
            plt.close(fig)
        elif event.key == 'q':
            print("Quitting...")
            user_action['next'] = False
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    return user_action['next']

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pick', help='Task name')
    parser.add_argument('--ckpt', type=str, default='policy_best.ckpt', help='Checkpoint filename')
    # Default to 1 so you can see individual episodes clearly
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for validation')
    args = parser.parse_args()

    ckpt_dir = os.path.join(TRAIN_CONFIG['checkpoint_dir'], args.task)
    ckpt_path = os.path.join(ckpt_dir, args.ckpt)

    try:
        engine = ACTInference(ckpt_path, POLICY_CONFIG, TASK_CONFIG)
    except Exception as e:
        print(f"Error initializing engine: {e}")
        return

    print(f"\n--- Validating Batch Size: {args.batch_size} ---")
    
    # 1. Load Data
    data_dir = TASK_CONFIG['dataset_dir']
    _, val_dataloader, _, _ = load_data(
        data_dir, 
        batch_size_train=8, 
        batch_size_val=args.batch_size, 
        num_queries=POLICY_CONFIG['num_queries']
    )
    
    data_iter = iter(val_dataloader)
    
    batch_idx = 0
    while True:
        try:
            batch_data = next(data_iter)
        except StopIteration:
            print(">> End of dataset reached (covered all 8 episodes). Restarting to sample NEW random windows...")
            data_iter = iter(val_dataloader)
            batch_data = next(data_iter)
            
        batch_idx += 1
        print(f"\n--- Processing Batch {batch_idx} ---")
        image_data, qpos_data, action_data, is_pad = batch_data
        
        # 2. Predict
        curr_image_tensor = image_data[:, 0, :, :, :]
        curr_image_tensor = curr_image_tensor.unsqueeze(1) # Add Camera Dim
        
        preds = engine.predict_batch(curr_image_tensor, qpos_data)
        
        # 3. Process GT
        gt_pos_norm = action_data[..., :3].cpu().numpy()
        gt_pos_real = engine._denormalize_pos(gt_pos_norm)
        
        gt_rot6d = action_data[..., 3:9] 
        gt_rot_mat = rot6d_to_matrix(gt_rot6d)
        gt_quat = mat_to_quat(gt_rot_mat)
        
        gt_gripper = action_data[..., 9:].cpu().numpy()
        
        gt_dict = {'pos': gt_pos_real, 'quat': gt_quat, 'gripper': gt_gripper}

        # 4. Visualize
        vis_images_tensor = curr_image_tensor.squeeze(1) 
        vis_images = vis_images_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255.0
        vis_images = vis_images.astype(np.uint8)

        continue_loop = visualize_batch(vis_images, gt_dict, preds)
        
        if not continue_loop:
            break

if __name__ == '__main__':
    main()