# import os
# import h5py
# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF

# from training.policy_joint_1 import ACTPolicy, CNNMLPPolicy

# # Lighting Robustness (Augmentation Only)
# COLOR_JITTER = transforms.ColorJitter(
#     brightness=0.4, 
#     contrast=0.4, 
#     saturation=0.3, 
#     hue=0.1
# )

# def cycle(dataloader):
#     """
#     Infinite iterator for the dataloader.
#     When the dataloader is exhausted, it restarts automatically.
#     """
#     while True:
#         for batch in dataloader:
#             yield batch

# class EpisodicDataset(torch.utils.data.Dataset):
#     def __init__(self, samples, dataset_path, camera_names, num_queries, augment=False, state_dim=7):
#         """
#         Args:
#             samples: List of tuples (episode_key, start_ts) representing every frame.
#         """
#         super().__init__()
#         self.samples = samples 
#         self.dataset_path = dataset_path
#         self.camera_names = camera_names
#         self.num_queries = num_queries
#         self.augment = augment
#         self.h5_file = None 
#         self.state_dim = state_dim
#         self.shift_padding = 4  # Pixels to shift

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         # 1. Get the specific episode and timestamp for this index
#         episode_key, start_ts = self.samples[index]
        
#         if self.h5_file is None:
#             # SWMR (Single Writer Multiple Reader) allows reading while writing
#             self.h5_file = h5py.File(self.dataset_path, 'r', libver='latest', swmr=True)
            
#         demo = self.h5_file[episode_key]
#         episode_len = demo['observations/qpos'].shape[0]
        
#         # --- 2. Load State (Sliced Read) ---
#         # We use the specific start_ts, no random sampling here
#         qpos_arm = demo['observations/qpos'][start_ts]
#         gripper_state = demo['observations/gripper_state'][start_ts]
        
#         if hasattr(gripper_state, '__len__'):
#             gripper_state = gripper_state 
#         else:
#             gripper_state = np.array([gripper_state])

#         full_qpos = np.concatenate([qpos_arm, gripper_state], axis=0).astype(np.float32)

#         # --- 3. Load Action (Sliced Read) ---
#         end_ts = min(start_ts + self.num_queries, episode_len)
        
#         action_arm = demo['action'][start_ts:end_ts]
#         action_gripper = demo['observations/gripper_state'][start_ts:end_ts]
        
#         if action_gripper.ndim == 1:
#             action_gripper = action_gripper[:, np.newaxis]
            
#         action_seq = np.concatenate([action_arm, action_gripper], axis=1).astype(np.float32)
#         actual_len = action_seq.shape[0]

#         # --- 4. Load Images (Sliced Read) ---
#         image_dict = dict()
#         for cam_name in self.camera_names:
#             h5_cam_key = f'observations/images/{cam_name}'
#             image_dict[cam_name] = demo[h5_cam_key][start_ts]

#         # --- Padding ---
#         padded_action = np.zeros((self.num_queries, self.state_dim), dtype=np.float32)
#         if actual_len > 0:
#             padded_action[:actual_len] = action_seq
        
#         is_pad = np.zeros(self.num_queries, dtype=bool)
#         is_pad[actual_len:] = True

#         all_cam_images = []
#         for cam_name in self.camera_names:
#             img = image_dict[cam_name]
#             img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
#             if self.augment:
#                 # A. Color Jitter (Existing)
#                 img_t = COLOR_JITTER(img_t) 
                
#                 # B. Random Shift (Spatial Augmentation)
#                 # 1. Pad image borders to create room for shifting
#                 # padding_mode='edge' repeats the last pixel to avoid black borders
#                 img_t = TF.pad(img_t, self.shift_padding, padding_mode='edge')
                
#                 # 2. Randomly Crop back to original size
#                 # This effectively "shifts" the view left/right/up/down
#                 orig_h, orig_w = img.shape[0], img.shape[1]
#                 i, j, h, w = transforms.RandomCrop.get_params(
#                     img_t, output_size=(orig_h, orig_w)
#                 )
#                 img_t = TF.crop(img_t, i, j, h, w)
#             all_cam_images.append(img_t)
        
#         image_data = torch.stack(all_cam_images, dim=0)
#         qpos_data = torch.from_numpy(full_qpos).float()
        
#         # Augmentation: Randomly mask joint positions (10% chance)
#         if self.augment and np.random.rand() < 0.10:
#             qpos_data = torch.zeros_like(qpos_data)
            
#         action_data = torch.from_numpy(padded_action).float()
#         is_pad = torch.from_numpy(is_pad).bool()

#         return image_data, qpos_data, action_data, is_pad
    
#     def __del__(self):
#         if hasattr(self, 'h5_file') and self.h5_file is not None:
#             self.h5_file.close()

# def get_norm_stats(dataset_path):
#     with h5py.File(dataset_path, 'r') as root:
#         norm_group = root['normalization']
#         a_mean = norm_group['action_mean'][()]
#         a_std = norm_group['action_std'][()]
#         q_mean = norm_group['qpos_mean'][()]
#         q_std = norm_group['qpos_std'][()]
#         g_mean = norm_group['gripper_state_mean'][()]
#         g_std = norm_group['gripper_state_std'][()]
        
#         action_mean = np.concatenate([a_mean, g_mean])
#         action_std = np.concatenate([a_std, g_std])
#         qpos_mean = np.concatenate([q_mean, g_mean])
#         qpos_std = np.concatenate([q_std, g_std])
            
#     return {
#         "action_mean": action_mean, "action_std": action_std,
#         "qpos_mean": qpos_mean, "qpos_std": qpos_std,
#     }

# def load_data(dataset_path, batch_size_train, batch_size_val, camera_names, 
#               num_queries=8, train_ratio=0.9, state_dim=7, **kwargs):
    
#     if isinstance(camera_names, str):
#         camera_names = [camera_names]
        
#     print(f"\nScanning dataset structure: {os.path.basename(dataset_path)}")
    
#     # 1. Get Episode Keys
#     with h5py.File(dataset_path, 'r') as root:
#         episode_keys = sorted([k for k in root.keys() if k.startswith('demo_')], 
#                               key=lambda x: int(x.split('_')[1]))

#     # 2. Split Episodes (Train/Val)
#     # Important: Split by episode first to prevent data leakage between frames
#     num_episodes = len(episode_keys)
#     indices = np.random.permutation(num_episodes)
#     train_count = int(train_ratio * num_episodes)
    
#     train_keys = [episode_keys[i] for i in indices[:train_count]]
#     val_keys = [episode_keys[i] for i in indices[train_count:]]

#     print(f"Total Episodes: {num_episodes} (Train: {len(train_keys)}, Val: {len(val_keys)})")

#     # 3. Flatten Dataset (Create 'Real Epoch' indices)
#     def flatten_dataset(keys):
#         flattened_samples = []
#         with h5py.File(dataset_path, 'r') as root:
#             for key in keys:
#                 # Get the exact length of this video
#                 episode_len = root[key]['observations/qpos'].shape[0]
#                 # Create a sample index for EVERY frame
#                 for t in range(episode_len):
#                     flattened_samples.append((key, t))
#         return flattened_samples

#     print("Indexing training frames... (This ensures 100% coverage)")
#     train_samples = flatten_dataset(train_keys)
    
#     print("Indexing validation frames...")
#     val_samples = flatten_dataset(val_keys)

#     print(f"Total Training Steps per Epoch: {len(train_samples) // batch_size_train}")

#     norm_stats = get_norm_stats(dataset_path)

#     # 4. Create Datasets
#     train_dataset = EpisodicDataset(train_samples, dataset_path, camera_names, num_queries, augment=True, state_dim=state_dim)
#     val_dataset = EpisodicDataset(val_samples, dataset_path, camera_names, num_queries, augment=False, state_dim=state_dim)

#     # 5. Create Loaders
#     # shuffle=True ensures we see every frame, but in random order (standard SGD)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=4)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, pin_memory=True, num_workers=2)

#     return train_loader, val_loader, norm_stats, False

# # Boilerplate functions
# def make_policy(policy_class, policy_config):
#     if policy_class == "ACT": return ACTPolicy(policy_config)
#     elif policy_class == "CNNMLP": return CNNMLPPolicy(policy_config)
#     raise ValueError(f"Unknown policy class: {policy_class}")

# def make_optimizer(policy_class, policy):
#     return policy.configure_optimizers()

# def compute_dict_mean(epoch_dicts):
#     result = {k: None for k in epoch_dicts[0]}
#     for k in result:
#         result[k] = np.mean([d[k].item() if hasattr(d[k], 'item') else d[k] for d in epoch_dicts])
#     return result

# def detach_dict(d):
#     return {k: v.detach() for k, v in d.items()}

# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)


## convert BGR to RGB

import os
import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from training.policy_joint_1 import ACTPolicy, CNNMLPPolicy

# Lighting Robustness (Augmentation Only)
COLOR_JITTER = transforms.ColorJitter(
    brightness=0.4, 
    contrast=0.4, 
    saturation=0.3, 
    hue=0.1
)

def cycle(dataloader):
    """
    Infinite iterator for the dataloader.
    When the dataloader is exhausted, it restarts automatically.
    """
    while True:
        for batch in dataloader:
            yield batch

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, samples, dataset_path, camera_names, num_queries, augment=False, state_dim=7):
        """
        Args:
            samples: List of tuples (episode_key, start_ts) representing every frame.
        """
        super().__init__()
        self.samples = samples 
        self.dataset_path = dataset_path
        self.camera_names = camera_names
        self.num_queries = num_queries
        self.augment = augment
        self.h5_file = None 
        self.state_dim = state_dim
        self.shift_padding = 4  # Pixels to shift

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 1. Get the specific episode and timestamp for this index
        episode_key, start_ts = self.samples[index]
        
        if self.h5_file is None:
            # SWMR (Single Writer Multiple Reader) allows reading while writing
            self.h5_file = h5py.File(self.dataset_path, 'r', libver='latest', swmr=True)
            
        demo = self.h5_file[episode_key]
        episode_len = demo['observations/qpos'].shape[0]
        
        # --- 2. Load State (Sliced Read) ---
        # We use the specific start_ts, no random sampling here
        qpos_arm = demo['observations/qpos'][start_ts]
        gripper_state = demo['observations/gripper_state'][start_ts]
        
        if hasattr(gripper_state, '__len__'):
            gripper_state = gripper_state 
        else:
            gripper_state = np.array([gripper_state])

        full_qpos = np.concatenate([qpos_arm, gripper_state], axis=0).astype(np.float32)

        # --- 3. Load Action (Sliced Read) ---
        end_ts = min(start_ts + self.num_queries, episode_len)
        
        action_arm = demo['action'][start_ts:end_ts]
        action_gripper = demo['observations/gripper_state'][start_ts:end_ts]
        
        if action_gripper.ndim == 1:
            action_gripper = action_gripper[:, np.newaxis]
            
        action_seq = np.concatenate([action_arm, action_gripper], axis=1).astype(np.float32)
        actual_len = action_seq.shape[0]

        # --- 4. Load Images (Sliced Read) ---
        image_dict = dict()
        for cam_name in self.camera_names:
            h5_cam_key = f'observations/images/{cam_name}'
            image_dict[cam_name] = demo[h5_cam_key][start_ts]

        # --- Padding ---
        padded_action = np.zeros((self.num_queries, self.state_dim), dtype=np.float32)
        if actual_len > 0:
            padded_action[:actual_len] = action_seq
        
        is_pad = np.zeros(self.num_queries, dtype=bool)
        is_pad[actual_len:] = True

        all_cam_images = []
        for cam_name in self.camera_names:
            img = image_dict[cam_name]
            
            # --- FIX: Convert BGR to RGB ---
            # HDF5 often stores as BGR (OpenCV default), but PyTorch models expect RGB.
            # We reverse the last dimension (channels). 
            # .copy() ensures memory is contiguous, avoiding negative stride issues in torch.
            # img = img[..., ::-1].copy()

            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            if self.augment:
                # A. Color Jitter (Existing)
                img_t = COLOR_JITTER(img_t) 
                
                # # B. Random Shift (Spatial Augmentation)
                # # 1. Pad image borders to create room for shifting
                # # padding_mode='edge' repeats the last pixel to avoid black borders
                # img_t = TF.pad(img_t, self.shift_padding, padding_mode='edge')
                
                # # 2. Randomly Crop back to original size
                # # This effectively "shifts" the view left/right/up/down
                # orig_h, orig_w = img.shape[0], img.shape[1]
                # i, j, h, w = transforms.RandomCrop.get_params(
                #     img_t, output_size=(orig_h, orig_w)
                # )
                # img_t = TF.crop(img_t, i, j, h, w)
            all_cam_images.append(img_t)
        
        image_data = torch.stack(all_cam_images, dim=0)
        qpos_data = torch.from_numpy(full_qpos).float()
        
        # Augmentation: Randomly mask joint positions (10% chance)
        if self.augment and np.random.rand() < 0.10:
            qpos_data = torch.zeros_like(qpos_data)
            
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        return image_data, qpos_data, action_data, is_pad
    
    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()

def get_norm_stats(dataset_path):
    with h5py.File(dataset_path, 'r') as root:
        norm_group = root['normalization']
        a_mean = norm_group['action_mean'][()]
        a_std = norm_group['action_std'][()]
        q_mean = norm_group['qpos_mean'][()]
        q_std = norm_group['qpos_std'][()]
        g_mean = norm_group['gripper_state_mean'][()]
        g_std = norm_group['gripper_state_std'][()]
        
        action_mean = np.concatenate([a_mean, g_mean])
        action_std = np.concatenate([a_std, g_std])
        qpos_mean = np.concatenate([q_mean, g_mean])
        qpos_std = np.concatenate([q_std, g_std])
            
    return {
        "action_mean": action_mean, "action_std": action_std,
        "qpos_mean": qpos_mean, "qpos_std": qpos_std,
    }

def load_data(dataset_path, batch_size_train, batch_size_val, camera_names, 
              num_queries=8, train_ratio=0.9, state_dim=7, **kwargs):
    
    if isinstance(camera_names, str):
        camera_names = [camera_names]
        
    print(f"\nScanning dataset structure: {os.path.basename(dataset_path)}")
    
    # 1. Get Episode Keys
    with h5py.File(dataset_path, 'r') as root:
        episode_keys = sorted([k for k in root.keys() if k.startswith('demo_')], 
                              key=lambda x: int(x.split('_')[1]))

    # 2. Split Episodes (Train/Val)
    # Important: Split by episode first to prevent data leakage between frames
    num_episodes = len(episode_keys)
    indices = np.random.permutation(num_episodes)
    train_count = int(train_ratio * num_episodes)
    
    train_keys = [episode_keys[i] for i in indices[:train_count]]
    val_keys = [episode_keys[i] for i in indices[train_count:]]

    print(f"Total Episodes: {num_episodes} (Train: {len(train_keys)}, Val: {len(val_keys)})")

    # 3. Flatten Dataset (Create 'Real Epoch' indices)
    def flatten_dataset(keys):
        flattened_samples = []
        with h5py.File(dataset_path, 'r') as root:
            for key in keys:
                # Get the exact length of this video
                episode_len = root[key]['observations/qpos'].shape[0]
                # Create a sample index for EVERY frame
                for t in range(episode_len):
                    flattened_samples.append((key, t))
        return flattened_samples

    print("Indexing training frames... (This ensures 100% coverage)")
    train_samples = flatten_dataset(train_keys)
    
    print("Indexing validation frames...")
    val_samples = flatten_dataset(val_keys)

    print(f"Total Training Steps per Epoch: {len(train_samples) // batch_size_train}")

    norm_stats = get_norm_stats(dataset_path)

    # 4. Create Datasets
    train_dataset = EpisodicDataset(train_samples, dataset_path, camera_names, num_queries, augment=True, state_dim=state_dim)
    val_dataset = EpisodicDataset(val_samples, dataset_path, camera_names, num_queries, augment=False, state_dim=state_dim)

    # 5. Create Loaders
    # shuffle=True ensures we see every frame, but in random order (standard SGD)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=0)

    return train_loader, val_loader, norm_stats, False

# Boilerplate functions
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