import os
import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from einops import rearrange
import torchvision.transforms.functional as F

from training.policy_joint import ACTPolicy, CNNMLPPolicy

# Lighting Robustness (Augmentation Only)
# Applied to float [0, 1] images before the policy handles normalization
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
            
            # 1. Load full episode qpos (6D) and gripper (1D)
            qpos_all = demo['observations/qpos'][()]      
            gripper_all = demo['observations/gripper'][()] 
            
            if gripper_all.ndim == 1:
                gripper_all = gripper_all[:, np.newaxis]
                
            # Combine into 7D sequence [q1, q2, q3, q4, q5, q6, gripper]
            full_states = np.concatenate([qpos_all, gripper_all], axis=1).astype(np.float32)
            episode_len = full_states.shape[0]
            
            # Sample a random start time t
            start_ts = np.random.choice(episode_len)
            
            # 2. INPUT: State at current time t
            qpos_input = full_states[start_ts] 
            
            # 3. TARGET: Sequence of states starting from t+1
            # (t+1, t+2, ..., t+num_queries)
            action_label = full_states[start_ts + 1:] 
            action_len = action_label.shape[0]

            # 4. IMAGE: Taken at current time t
            image_dict = dict()
            for cam_name in self.camera_names:
                h5_cam_key = f'observations/images_{cam_name}'
                image_dict[cam_name] = demo[h5_cam_key][start_ts]

        # Padding Action Sequence
        padded_action = np.zeros((self.num_queries, 7), dtype=np.float32)
        actual_len = min(action_len, self.num_queries)
        if actual_len > 0:
            # We take the next 'num_queries' steps as the label
            padded_action[:actual_len] = action_label[:actual_len]
        
        # is_pad: True for steps beyond the end of the trajectory
        is_pad = np.zeros(self.num_queries, dtype=bool)
        is_pad[actual_len:] = True

        # --- Image Processing & Augmentation ---
        # all_cam_images = []
        # for cam_name in self.camera_names:
        #     img = image_dict[cam_name]
        #     # Convert to [C, H, W] and scale to [0, 1]
        #     img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
        #     if self.augment:
        #         # Lighting Robustness: Shift brightness/contrast
        #         img_t = COLOR_JITTER(img_t)
        #         # Spatial Robustness: Small random shifts
        #         h, w = img_t.shape[1:]
        #         img_t = transforms.RandomCrop((h, w), padding=4)(img_t)
            
        #     all_cam_images.append(img_t)
        all_cam_images = []
        for cam_name in self.camera_names:
            img = image_dict[cam_name]
            # Convert to [C, H, W] and scale to [0, 1]
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
            # --- SPECIFIC TRANSFORMATION FOR REALSENSE ---
            if cam_name == 'realsense':
                c, h, w = img_t.shape
                # 1. Keep the left half [0 to W/2], discard the right half
                img_t = img_t[:, :h//3*2, :w//2]
                
                # 2. Resize from (H, W/2) back to (H, W) -> (480, 640)
                # Using antialias=True for better quality
                img_t = F.resize(img_t, (h, w), antialias=True)
            
            if cam_name == 'aria':
                c, h, w = img_t.shape
                # 1. Keep the left half [0 to W/2], discard the right half
                img_t = img_t[:, h//7*3:h//5*4, w//5:w//5*4]
                
                # 2. Resize from (H, W/2) back to (H, W) -> (480, 640)
                # Using antialias=True for better quality
                img_t = F.resize(img_t, (h, w), antialias=True)
            
            if self.augment:
                # Lighting Robustness
                img_t = COLOR_JITTER(img_t)
                # Spatial Robustness: Small random shifts
                h_now, w_now = img_t.shape[1:]
                img_t = transforms.RandomCrop((h_now, w_now), padding=4)(img_t)
            
            all_cam_images.append(img_t)
        
        image_data = torch.stack(all_cam_images, dim=0)
        qpos_data = torch.from_numpy(qpos_input).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_path):
    """Loads 6D stats and appends 1.0/0.0 for the raw binary gripper."""
    with h5py.File(dataset_path, 'r') as root:
        try:
            # Since target actions are future qpos+gripper, 
            # they share the same stats as qpos.
            qpos_mean_6d = root['norm_stats/qpos/mean'][()]
            qpos_std_6d = root['norm_stats/qpos/std'][()]
            
            # Gripper (7th dim) is binary 0/1; we append Mean 0 and Std 1
            mean_7d = np.append(qpos_mean_6d, 0.0)
            std_7d = np.append(qpos_std_6d, 1.0)
            
        except KeyError as e:
            print(f"Error: Could not find stats in {dataset_path}. Check 'norm_stats' group.")
            raise e

    return {
        "action_mean": mean_7d, "action_std": std_7d,
        "qpos_mean": mean_7d, "qpos_std": std_7d,
    }


def load_data(dataset_path, batch_size_train, batch_size_val, camera_names, 
              num_queries=8, samples_per_epoch=1, **kwargs):
    
    # Safety: ensure camera_names is a list
    if isinstance(camera_names, str):
        camera_names = [camera_names]
        
    print_h5_structure(dataset_path)
    
    with h5py.File(dataset_path, 'r') as root:
        # Find all demos inside the 'data/' group
        episode_keys = []
        if 'data' in root:
            data_group = root['data']
            for key in data_group.keys():
                full_key = f"data/{key}"
                # Verify it's a valid demo group
                if 'observations' in data_group[key]:
                    episode_keys.append(full_key)
        
        if not episode_keys:
            raise ValueError(f"No demo groups found in 'data/' inside {dataset_path}")

        # Multiplier to increase diversity of crops/samples per epoch
        total_episodes = episode_keys * samples_per_epoch
        num_episodes = len(total_episodes)

    print(f'Original Demos: {len(episode_keys)} | Virtual Count: {num_episodes}')

    # Train/Val Split (80/20)
    indices = np.random.permutation(num_episodes)
    train_ratio = 23.0/31.0
    train_idx = indices[:int(train_ratio * num_episodes)]
    val_idx = indices[int(train_ratio * num_episodes):]

    train_keys = [total_episodes[i] for i in train_idx]
    val_keys = [total_episodes[i] for i in val_idx]

    norm_stats = get_norm_stats(dataset_path)

    train_dataset = EpisodicDataset(train_keys, dataset_path, camera_names, num_queries, augment=True)
    val_dataset = EpisodicDataset(val_keys, dataset_path, camera_names, num_queries, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, pin_memory=True, num_workers=1)

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

# def get_image(images, camera_names, device='cpu'):
#     """Inference helper: Scales input image for policy.py."""
#     curr_images = []
#     for cam_name in camera_names:
#         img = images[cam_name]
#         img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
#         curr_images.append(img_t)
#     return torch.stack(curr_images, dim=0).to(device).unsqueeze(0)

def get_image(images, camera_names, device='cpu'):
    """Inference helper: Scales input image for policy.py."""
    curr_images = []
    for cam_name in camera_names:
        img = images[cam_name]
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # --- MATCH TRAINING PREPROCESSING ---
        if cam_name == 'realsense':
            c, h, w = img_t.shape
            img_t = img_t[:, :h//3*2, :w//2]  # Crop right half
            img_t = F.resize(img_t, (h, w), antialias=True) # Stretch back to 480x640
            
        if cam_name == 'aria':
            c, h, w = img_t.shape
            img_t = img_t[:, h//7*3:h//5*4, w//5:w//5*4]
            img_t = F.resize(img_t, (h, w), antialias=True) # Stretch back to 480x640
            
        curr_images.append(img_t)
    return torch.stack(curr_images, dim=0).to(device).unsqueeze(0)