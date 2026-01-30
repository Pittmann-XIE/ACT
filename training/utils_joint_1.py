import os
import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from einops import rearrange

from training.policy_joint_1 import ACTPolicy, CNNMLPPolicy

# Lighting Robustness (Augmentation Only)
COLOR_JITTER = transforms.ColorJitter(
    brightness=0.4, 
    contrast=0.4, 
    saturation=0.3, 
    hue=0.1
)

def print_h5_structure(dataset_path):
    """Inspects and prints the H5 file hierarchy to confirm data paths."""
    print("\n" + "="*50)
    print(f"INSPECTING H5 STRUCTURE: {os.path.basename(dataset_path)}")
    print("="*50)
    with h5py.File(dataset_path, 'r') as f:
        def print_attrs(name, obj):
            shift = name.count('/') * '  '
            if isinstance(obj, h5py.Dataset):
                print(f"{shift}D {name} (shape: {obj.shape}, dtype: {obj.dtype})")
            else:
                print(f"{shift}G {name}")
        f.visititems(print_attrs)
    print("="*50 + "\n")

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_keys, dataset_path, camera_names, num_queries, augment=False):
        super(EpisodicDataset).__init__()
        self.episode_keys = episode_keys
        self.dataset_path = dataset_path
        self.camera_names = camera_names
        self.num_queries = num_queries
        self.augment = augment

    def __len__(self):
        return len(self.episode_keys)

    def __getitem__(self, index):
        episode_key = self.episode_keys[index]
        
        with h5py.File(self.dataset_path, 'r') as root:
            demo = root[episode_key]
            
            # --- 1. Load State (Input) ---
            # Arm Qpos (6D)
            qpos_arm = demo['observations/qpos'][()]
            # Gripper State (1D)
            gripper_state = demo['observations/gripper_state'][()]
            if gripper_state.ndim == 1:
                gripper_state = gripper_state[:, np.newaxis]
            
            # Full State: (T, 7)
            full_qpos = np.concatenate([qpos_arm, gripper_state], axis=1).astype(np.float32)
            episode_len = full_qpos.shape[0]

            # --- 2. Load Action (Target) ---
            # Arm Action (6D)
            action_arm = demo['action'][()]
            # Gripper Action (1D) - In this dataset, gripper state serves as the action target
            # Full Action: (T, 7)
            full_action = np.concatenate([action_arm, gripper_state], axis=1).astype(np.float32)

            # Sample a random start time t
            # Ensure we don't pick the very last step if we need at least one action
            start_ts = np.random.choice(episode_len)
            
            # INPUT: State at current time t
            qpos_input = full_qpos[start_ts]
            
            # TARGET: Sequence of actions starting from t
            # Note: ACT usually predicts actions from t to t+k
            action_seq = full_action[start_ts:] 
            action_len = action_seq.shape[0]

            # --- 3. Load Images ---
            image_dict = dict()
            for cam_name in self.camera_names:
                # Updated path: observations/images/cam1_rgb
                h5_cam_key = f'observations/images/{cam_name}'
                image_dict[cam_name] = demo[h5_cam_key][start_ts]

        # --- Padding Action Sequence ---
        padded_action = np.zeros((self.num_queries, 7), dtype=np.float32)
        actual_len = min(action_len, self.num_queries)
        
        if actual_len > 0:
            padded_action[:actual_len] = action_seq[:actual_len]
        
        # is_pad: True for steps beyond the end of the trajectory
        is_pad = np.zeros(self.num_queries, dtype=bool)
        is_pad[actual_len:] = True

        # --- Image Processing & Augmentation ---
        all_cam_images = []
        for cam_name in self.camera_names:
            img = image_dict[cam_name]
            # Convert to [C, H, W] and scale to [0, 1]
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
            if self.augment:
                img_t = COLOR_JITTER(img_t)
            
            all_cam_images.append(img_t)
        
        image_data = torch.stack(all_cam_images, dim=0)
        qpos_data = torch.from_numpy(qpos_input).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_path):
    """Loads stats from 'normalization' group and combines arm+gripper."""
    with h5py.File(dataset_path, 'r') as root:
        try:
            norm_group = root['normalization']
            
            # Load 6D Arm stats
            # Note: Using [()] to read dataset into numpy array
            a_mean = norm_group['action_mean'][()]
            a_std = norm_group['action_std'][()]
            q_mean = norm_group['qpos_mean'][()]
            q_std = norm_group['qpos_std'][()]
            
            # Load 1D Gripper stats
            g_mean = norm_group['gripper_state_mean'][()]
            g_std = norm_group['gripper_state_std'][()]
            
            # Concatenate to 7D: [Arm_0...Arm_5, Gripper]
            action_mean = np.concatenate([a_mean, g_mean])
            action_std = np.concatenate([a_std, g_std])
            qpos_mean = np.concatenate([q_mean, g_mean])
            qpos_std = np.concatenate([q_std, g_std])
            
        except KeyError as e:
            print(f"Error: Could not find stats in {dataset_path}. Check 'normalization' group.")
            print("Available keys in root:", list(root.keys()))
            if 'normalization' in root:
                print("Keys in normalization:", list(root['normalization'].keys()))
            raise e

    return {
        "action_mean": action_mean, "action_std": action_std,
        "qpos_mean": qpos_mean, "qpos_std": qpos_std,
    }


def load_data(dataset_path, batch_size_train, batch_size_val, camera_names, 
              num_queries=8, samples_per_epoch=1, train_ratio=0.9, **kwargs):
    
    if isinstance(camera_names, str):
        camera_names = [camera_names]
        
    print_h5_structure(dataset_path)
    
    with h5py.File(dataset_path, 'r') as root:
        # Scan root for keys starting with "demo_"
        episode_keys = sorted([k for k in root.keys() if k.startswith('demo_')], 
                              key=lambda x: int(x.split('_')[1]))
        
        if not episode_keys:
            raise ValueError(f"No groups starting with 'demo_' found in {dataset_path}")

        # Multiplier to increase diversity of crops/samples per epoch
        total_episodes = episode_keys * samples_per_epoch
        num_episodes = len(total_episodes)

    print(f'Original Demos: {len(episode_keys)} | Virtual Count: {num_episodes}')

    indices = np.random.permutation(num_episodes)
    
    train_count = int(train_ratio * num_episodes) 
    train_idx = indices[:train_count]
    val_idx = indices[train_count:]

    train_keys = [total_episodes[i] for i in train_idx]
    val_keys = [total_episodes[i] for i in val_idx]

    norm_stats = get_norm_stats(dataset_path)

    train_dataset = EpisodicDataset(train_keys, dataset_path, camera_names, num_queries, augment=True)
    val_dataset = EpisodicDataset(val_keys, dataset_path, camera_names, num_queries, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=2)

    return train_loader, val_loader, norm_stats, False

# --- Policy & Optimizer Boilerplate ---
def make_policy(policy_class, policy_config):
    if policy_class == "ACT": return ACTPolicy(policy_config)
    elif policy_class == "CNNMLP": return CNNMLPPolicy(policy_config)
    raise ValueError(f"Unknown policy class: {policy_class}")

def make_optimizer(policy_class, policy):
    return policy.configure_optimizers()

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    for k in result:
        result[k] = np.mean([d[k].item() if hasattr(d[k], 'item') else d[k] for d in epoch_dicts])
    return result

def detach_dict(d):
    return {k: v.detach() for k, v in d.items()}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_image(images, camera_names, device='cpu'):
    """Inference helper: Scales input image for policy.py."""
    curr_images = []
    for cam_name in camera_names:
        img = images[cam_name]
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        curr_images.append(img_t)
    return torch.stack(curr_images, dim=0).to(device).unsqueeze(0)