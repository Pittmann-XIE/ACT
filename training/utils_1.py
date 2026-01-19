## enhanced sampling

import os
import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from einops import rearrange # Added based on usage in get_image

from training.policy_1 import ACTPolicy, CNNMLPPolicy

import IPython
e = IPython.embed


def quaternion_to_rot6d(quaternion):
    """Convert quaternion (qx, qy, qz, qw) to 6D rotation representation."""
    if isinstance(quaternion, np.ndarray):
        quaternion = torch.from_numpy(quaternion).float()
    
    q = F.normalize(quaternion, dim=-1)
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    rot_matrix = torch.stack([
        1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy),
        2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx),
        2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)
    ], dim=-1)
    
    rot_matrix = rot_matrix.reshape(*quaternion.shape[:-1], 3, 3)
    rot_6d = rot_matrix[..., :2, :].reshape(*quaternion.shape[:-1], 6)
    
    return rot_6d.numpy() if isinstance(rot_6d, torch.Tensor) else rot_6d


def construct_10d_state_batch(tx, ty, tz, qx, qy, qz, qw, distance_class):
    """
    Construct 10D state for entire sequence: [tx, ty, tz, 6D_rotation, distance_class]
    Pre-computed version for batch processing.
    """
    quaternion = np.stack([qx, qy, qz, qw], axis=-1)  # (seq_len, 4)
    rot_6d = quaternion_to_rot6d(quaternion)  # (seq_len, 6)
    
    state_10d = np.concatenate([
        tx[:, None],
        ty[:, None],
        tz[:, None],
        rot_6d,
        distance_class[:, None]
    ], axis=-1)  # (seq_len, 10)
    
    return state_10d.astype(np.float32)


class InstanceDataset(torch.utils.data.Dataset):
    """Dataset that loads instances from task HDF5 files with pre-loading and sampling strategy."""
    
    def __init__(self, instance_list, num_queries, preload, augment=False, samples_per_epoch=1):
        """
        Args:
            instance_list: list of (h5_path, instance_key) tuples
            num_queries: sequence length for actions
            preload: whether to load all data into RAM
            augment: (bool) whether to apply data augmentation (True for train, False for val)
            samples_per_epoch: (int) how many times to sample each episode per epoch
        """
        self.instance_list = instance_list
        self.num_queries = num_queries
        self.data_cache = {}
        self.preload = preload
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch # ### UPDATED: Store sampling rate
        
        # --- Define Augmentations ---
        # Note: We rely on dynamic transform in __getitem__ for image size safety
        if self.augment:
            print(f"Augmentation Enabled: RandomCrop + ColorJitter + State Noise")
        
        if self.preload:
            print(f"Pre-loading {len(instance_list)} instances into memory...")
            for idx, (h5_path, instance_key) in enumerate(tqdm(instance_list, desc="Loading data")):
                cache_key = (h5_path, instance_key)
                
                with h5py.File(h5_path, 'r') as h5f:
                    instance_group = h5f[instance_key]
                    data_group = instance_group['data']
                    
                    tx = data_group['tx'][()].astype(np.float32)
                    ty = data_group['ty'][()].astype(np.float32)
                    tz = data_group['tz'][()].astype(np.float32)
                    qx = data_group['qx'][()].astype(np.float32)
                    qy = data_group['qy'][()].astype(np.float32)
                    qz = data_group['qz'][()].astype(np.float32)
                    qw = data_group['qw'][()].astype(np.float32)
                    distance_class = data_group['distance_class'][()].astype(np.float32)
                    
                    image = data_group['image'][()]
                    
                    state_10d = construct_10d_state_batch(
                        tx, ty, tz, qx, qy, qz, qw, distance_class
                    )
                    
                    image_torch = torch.from_numpy(image).float()
                    image_torch = image_torch.permute(0, 3, 1, 2)
                    image_torch = image_torch / 255.0  # Keep in [0, 1]
                    
                    self.data_cache[cache_key] = {
                        'state_10d': state_10d,
                        'images': image_torch,
                        'seq_len': len(tx)
                    }
            
            print(f"✓ Pre-loaded {len(self.data_cache)} instances")

    def __len__(self):
        # ### UPDATED: Return virtual length based on samples_per_epoch
        return len(self.instance_list) * self.samples_per_epoch

    def __getitem__(self, index):
        # ### UPDATED: Map virtual index back to actual instance index
        index = index % len(self.instance_list)
        
        h5_path, instance_key = self.instance_list[index]
        cache_key = (h5_path, instance_key)
        
        # Retrieve pre-loaded data
        cached_data = self.data_cache[cache_key]
        state_10d = cached_data['state_10d']
        images = cached_data['images']
        seq_len = cached_data['seq_len']
        
        # Sample start timestep
        max_start = max(0, seq_len - self.num_queries)
        start_ts = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        
        # --- State Handling ---
        qpos_10d = state_10d[start_ts].copy() # Copy to avoid modifying cache
        
        # 3. Add Noise to Robot State (Training Only)
        if self.augment:
            # Add Gaussian noise to everything: position, rotation, gripper
            # Sigma = 0.005 is safe for normalized/small scale data
            noise = np.random.normal(loc=0, scale=0.005, size=qpos_10d.shape).astype(np.float32)
            qpos_10d = qpos_10d + noise
        
        action_seq = state_10d[start_ts:]
        action_len = len(action_seq)
        
        padded_action = np.zeros((self.num_queries, 10), dtype=np.float32)
        if action_len > 0:
            padded_action[:min(action_len, self.num_queries)] = action_seq[:self.num_queries]
        
        is_pad = np.zeros(self.num_queries, dtype=bool)
        if action_len < self.num_queries:
            is_pad[action_len:] = True
        
        # --- Image Handling ---
        # Get single image frame
        image_data = images[start_ts]  # (C, H, W)
        
        # 1 & 2. Apply Image Augmentations (Training Only)
        if self.augment:
            # Helper to handle dynamic sizes for RandomCrop
            _, h, w = image_data.shape
            # Re-initialize transform if size is needed (or just use padding trick)
            augmenter = transforms.Compose([
                transforms.RandomCrop(size=(h, w), padding=4), # Pad 4 then crop to original
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            ])
            image_data = augmenter(image_data)
            
        # Add batch dim for repeating
        image_data = image_data.unsqueeze(0) # (1, C, H, W)
        image_data = image_data.repeat(self.num_queries, 1, 1, 1)
        
        # Convert to tensors
        qpos_data = torch.from_numpy(qpos_10d).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        
        return image_data, qpos_data, action_data, is_pad


def get_instance_list(dataset_dir):
    instance_list = []
    h5_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.h5')])
    for h5_file in h5_files:
        h5_path = os.path.join(dataset_dir, h5_file)
        with h5py.File(h5_path, 'r') as h5f:
            instance_keys = sorted([k for k in h5f.keys() if k.startswith('Instance_')])
            for instance_key in instance_keys:
                instance_list.append((h5_path, instance_key))
    return instance_list


def get_norm_stats(dataset_dir):
    h5_files = [f for f in os.listdir(dataset_dir) if f.endswith('.h5')]
    if h5_files:
        h5_path = os.path.join(dataset_dir, h5_files[0])
        with h5py.File(h5_path, 'r') as h5f:
            if 'global_normalization_params' in h5f:
                norm_stats = {}
                for field in ['tx', 'ty', 'tz', 'distance']:
                    if field in h5f['global_normalization_params']:
                        field_group = h5f['global_normalization_params'][field]
                        norm_stats[field] = {
                            'mean': float(field_group['mean'][()]),
                            'std': float(field_group['std'][()])
                        }
                return norm_stats
    return {}


def load_data(dataset_dir, batch_size_train, batch_size_val, num_queries=8, data_loader_config=None, samples_per_epoch=1):
    """
    Load data from task HDF5 files.
    
    Args:
        samples_per_epoch: (int) Multiplier for dataset length. 
                           Increases sampling density per epoch. 
                           Recommended: 10-20 (and reduce total epochs accordingly).
    """
    if data_loader_config is None:
        from config.config_1 import DATA_LOADER_CONFIG
        data_loader_config = DATA_LOADER_CONFIG
    
    print(f'\nData from: {dataset_dir}\n')
    print(f'Samples per Epoch: {samples_per_epoch}')
    
    instance_list = get_instance_list(dataset_dir)
    norm_stats = get_norm_stats(dataset_dir)
    
    num_instances = len(instance_list)
    train_ratio = 64/88.0
    shuffled_indices = np.random.permutation(num_instances)
    train_indices = shuffled_indices[:int(train_ratio * num_instances)]
    val_indices = shuffled_indices[int(train_ratio * num_instances):]
    
    train_instances = [instance_list[i] for i in train_indices]
    val_instances = [instance_list[i] for i in val_indices]
    
    print(f"Train instances: {len(train_instances)} (Augmentation: ENABLED)")
    print(f"Val instances: {len(val_instances)} (Augmentation: DISABLED)\n")
    
    # --- Create Datasets ---
    # We apply samples_per_epoch to both train and val to stabilize validation curves
    
    train_dataset = InstanceDataset(
        train_instances, 
        num_queries,
        preload=data_loader_config['preload_data'],
        augment=True,
        samples_per_epoch=samples_per_epoch # ### UPDATED
    )
    val_dataset = InstanceDataset(
        val_instances, 
        num_queries,
        preload=data_loader_config['preload_data'],
        augment=False,
        samples_per_epoch=samples_per_epoch # ### UPDATED
    )
    
    num_workers_train = data_loader_config['num_workers_train']
    num_workers_val = data_loader_config['num_workers_val']
    
    train_kwargs = {
        'batch_size': batch_size_train,
        'shuffle': True,
        'pin_memory': data_loader_config['pin_memory'],
        'num_workers': num_workers_train
    }
    
    if num_workers_train > 0:
        train_kwargs['prefetch_factor'] = data_loader_config['prefetch_factor']
        train_kwargs['persistent_workers'] = data_loader_config['persistent_workers']
    
    val_kwargs = {
        'batch_size': batch_size_val,
        'shuffle': False,
        'pin_memory': data_loader_config['pin_memory'],
        'num_workers': num_workers_val
    }
    
    if num_workers_val > 0:
        val_kwargs['prefetch_factor'] = data_loader_config['prefetch_factor']
        val_kwargs['persistent_workers'] = data_loader_config['persistent_workers']
    
    train_dataloader = DataLoader(train_dataset, **train_kwargs)
    val_dataloader = DataLoader(val_dataset, **val_kwargs)
    
    is_sim = True
    
    return train_dataloader, val_dataloader, norm_stats, is_sim

# ... policy and optimizer functions ...
def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return optimizer

### helper functions

def get_image(images, camera_names, device='cpu'):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(images[cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

### env utils (kept for compatibility)
def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])
    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])
    return peg_pose, socket_pose