import os
import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

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


def construct_10d_state(tx, ty, tz, qx, qy, qz, qw, distance_class):
    """
    Construct 10D state: [tx, ty, tz, 6D_rotation, distance_class]
    
    Args:
        tx, ty, tz: scalars or arrays
        qx, qy, qz, qw: scalars or arrays
        distance_class: scalar or array
    
    Returns:
        state_10d: (seq_len, 10) array
    """
    quaternion = np.stack([qx, qy, qz, qw], axis=-1)
    rot_6d = quaternion_to_rot6d(quaternion)
    
    if np.isscalar(tx):
        state_10d = np.concatenate([[tx], [ty], [tz], rot_6d, [distance_class]])
    else:
        state_10d = np.concatenate([
            tx[:, None],
            ty[:, None],
            tz[:, None],
            rot_6d,
            distance_class[:, None]
        ], axis=-1)
    
    return state_10d


class InstanceDataset(torch.utils.data.Dataset):
    """Dataset that loads instances from task HDF5 files."""
    
    def __init__(self, instance_list, num_queries):
        """
        Args:
            instance_list: list of (h5_path, instance_key) tuples
            num_queries: sequence length for actions (number of future timesteps)
        """
        self.instance_list = instance_list
        self.num_queries = num_queries

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, index):
        h5_path, instance_key = self.instance_list[index]
        
        with h5py.File(h5_path, 'r') as h5f:
            instance_group = h5f[instance_key]
            data_group = instance_group['data']
            
            # Load entire sequences
            tx = data_group['tx'][()]
            ty = data_group['ty'][()]
            tz = data_group['tz'][()]
            qx = data_group['qx'][()]
            qy = data_group['qy'][()]
            qz = data_group['qz'][()]
            qw = data_group['qw'][()]
            distance_class = data_group['distance_class'][()]
            
            # Load image sequence
            image = data_group['image'][()]  # (seq_len, H, W, C)
            
            seq_len = len(tx)
        
        # Randomly sample a start timestep
        max_start = max(0, seq_len - self.num_queries)
        if max_start > 0:
            start_ts = np.random.randint(0, max_start + 1)
        else:
            start_ts = 0
        
        # qpos at start_ts (current state)
        qpos_10d = construct_10d_state(
            tx[start_ts], ty[start_ts], tz[start_ts],
            qx[start_ts], qy[start_ts], qz[start_ts], qw[start_ts],
            distance_class[start_ts]
        )
        
        # action sequence from start_ts onwards
        action_seq = construct_10d_state(
            tx[start_ts:], ty[start_ts:], tz[start_ts:],
            qx[start_ts:], qy[start_ts:], qz[start_ts:], qw[start_ts:],
            distance_class[start_ts:]
        )  # (remaining_seq_len, 10)
        
        action_len = len(action_seq)
        
        # Pad action sequence to num_queries length
        padded_action = np.zeros((self.num_queries, 10), dtype=np.float32)
        if action_len > 0:
            padded_action[:min(action_len, self.num_queries)] = action_seq[:self.num_queries]
        
        is_pad = np.zeros(self.num_queries, dtype=bool)
        if action_len < self.num_queries:
            is_pad[action_len:] = True
        
        # Process image at start_ts (current observation)
        image_data = torch.from_numpy(image[start_ts:start_ts+1]).float()  # (1, H, W, C)
        image_data = torch.einsum('t h w c -> t c h w', image_data)
        image_data = image_data / 255.0  # Normalize to [0, 1]
        
        # Repeat image for each query if needed (or keep single)
        # Assuming single-camera setup, replicate for all queries
        image_data = image_data.repeat(self.num_queries, 1, 1, 1)  # (num_queries, C, H, W)
        
        # Convert to tensors
        qpos_data = torch.from_numpy(qpos_10d).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        
        return image_data, qpos_data, action_data, is_pad


def get_instance_list(dataset_dir):
    """Collect all instances from all H5 files in dataset_dir."""
    instance_list = []
    
    h5_files = [f for f in os.listdir(dataset_dir) if f.endswith('.h5')]
    
    if not h5_files:
        raise ValueError(f"No .h5 files found in {dataset_dir}")
    
    for h5_file in h5_files:
        h5_path = os.path.join(dataset_dir, h5_file)
        
        with h5py.File(h5_path, 'r') as h5f:
            instance_keys = [k for k in h5f.keys() if k.startswith('Instance_')]
            
            for instance_key in instance_keys:
                instance_list.append((h5_path, instance_key))
    
    print(f"Found {len(instance_list)} instances across {len(h5_files)} H5 files")
    return instance_list


def get_norm_stats(dataset_dir):
    """
    Load normalization stats from HDF5 files.
    Stats are stored in global_normalization_params.
    """
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
                
                print("Loaded normalization stats from HDF5:")
                for field, stats in norm_stats.items():
                    print(f"  {field}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
                
                return norm_stats
    
    print("Warning: Could not load normalization stats from HDF5")
    return {}


def load_data(dataset_dir, batch_size_train, batch_size_val, num_queries=8):
    """
    Load data from task HDF5 files.
    
    Args:
        dataset_dir: directory containing .h5 files
        num_episodes: (unused, kept for compatibility)
        camera_names: (unused, kept for compatibility)
        batch_size_train: training batch size
        batch_size_val: validation batch size
        num_queries: sequence length for action predictions
    """
    print(f'\nData from: {dataset_dir}\n')
    print('Data structure: Instance-based, single camera')
    print('qpos format: (tx, ty, tz, 6D_rotation, distance_class) at time t')
    print('action format: sequence of (tx, ty, tz, 6D_rotation, distance_class) at times [t, t+1, ..., t+T]\n')
    print(f'Sequence length (num_queries): {num_queries}\n')
    
    # Get all instances
    instance_list = get_instance_list(dataset_dir)
    
    # Load normalization stats
    norm_stats = get_norm_stats(dataset_dir)
    
    # Split into train/val
    num_instances = len(instance_list)
    train_ratio = 0.75
    shuffled_indices = np.random.permutation(num_instances)
    train_indices = shuffled_indices[:int(train_ratio * num_instances)]
    val_indices = shuffled_indices[int(train_ratio * num_instances):]
    print(f'training: {train_indices}')
    print(f'validation: {val_indices}')
    
    train_instances = [instance_list[i] for i in train_indices]
    val_instances = [instance_list[i] for i in val_indices]
    
    # Create datasets
    train_dataset = InstanceDataset(train_instances, num_queries)
    val_dataset = InstanceDataset(val_instances, num_queries)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
                                  pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True,
                               pin_memory=True, num_workers=1, prefetch_factor=1)
    
    print(f"Train instances: {len(train_instances)}")
    print(f"Val instances: {len(val_instances)}\n")
    
    is_sim = True
    
    return train_dataloader, val_dataloader, norm_stats, is_sim


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

### env utils
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


def pos2pwm(pos:np.ndarray) -> np.ndarray:
    """
    :param pos: numpy array of joint positions in range [-pi, pi]
    :return: numpy array of pwm values in range [0, 4096]
    """ 
    return (pos / 3.14 + 1.) * 2048
    
def pwm2pos(pwm:np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm values in range [0, 4096]
    :return: numpy array of joint positions in range [-pi, pi]
    """
    return (pwm / 2048 - 1) * 3.14

def pwm2vel(pwm:np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm/s joint velocities
    :return: numpy array of rad/s joint velocities 
    """
    return pwm * 3.14 / 2048

def vel2pwm(vel:np.ndarray) -> np.ndarray:
    """
    :param vel: numpy array of rad/s joint velocities
    :return: numpy array of pwm/s joint velocities
    """
    return vel * 2048 / 3.14
    
def pwm2norm(x:np.ndarray) -> np.ndarray:
    """
    :param x: numpy array of pwm values in range [0, 4096]
    :return: numpy array of values in range [0, 1]
    """
    return x / 4096
    
def norm2pwm(x:np.ndarray) -> np.ndarray:
    """
    :param x: numpy array of values in range [0, 1]
    :return: numpy array of pwm values in range [0, 4096]
    """
    return x * 4096