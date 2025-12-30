# import os
# import h5py
# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# import torch.nn.functional as F

# from training.policy_1 import ACTPolicy, CNNMLPPolicy

# import IPython
# e = IPython.embed


# def quaternion_to_rot6d(quaternion):
#     """Convert quaternion (qx, qy, qz, qw) to 6D rotation representation."""
#     if isinstance(quaternion, np.ndarray):
#         quaternion = torch.from_numpy(quaternion).float()
    
#     q = F.normalize(quaternion, dim=-1)
#     qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
#     rot_matrix = torch.stack([
#         1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy),
#         2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx),
#         2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)
#     ], dim=-1)
    
#     rot_matrix = rot_matrix.reshape(*quaternion.shape[:-1], 3, 3)
#     rot_6d = rot_matrix[..., :2, :].reshape(*quaternion.shape[:-1], 6)
    
#     return rot_6d.numpy() if isinstance(rot_6d, torch.Tensor) else rot_6d


# def construct_10d_state(tx, ty, tz, qx, qy, qz, qw, distance_class):
#     """
#     Construct 10D state: [tx, ty, tz, 6D_rotation, distance_class]
    
#     Args:
#         tx, ty, tz: scalars or arrays
#         qx, qy, qz, qw: scalars or arrays
#         distance_class: scalar or array
    
#     Returns:
#         state_10d: (seq_len, 10) array
#     """
#     quaternion = np.stack([qx, qy, qz, qw], axis=-1)
#     rot_6d = quaternion_to_rot6d(quaternion)
    
#     if np.isscalar(tx):
#         state_10d = np.concatenate([[tx], [ty], [tz], rot_6d, [distance_class]])
#     else:
#         state_10d = np.concatenate([
#             tx[:, None],
#             ty[:, None],
#             tz[:, None],
#             rot_6d,
#             distance_class[:, None]
#         ], axis=-1)
    
#     return state_10d


# class InstanceDataset(torch.utils.data.Dataset):
#     """Dataset that loads instances from task HDF5 files."""
    
#     def __init__(self, instance_list, num_queries):
#         """
#         Args:
#             instance_list: list of (h5_path, instance_key) tuples
#             num_queries: sequence length for actions (number of future timesteps)
#         """
#         self.instance_list = instance_list
#         self.num_queries = num_queries

#     def __len__(self):
#         return len(self.instance_list)

#     def __getitem__(self, index):
#         h5_path, instance_key = self.instance_list[index]
        
#         with h5py.File(h5_path, 'r') as h5f:
#             instance_group = h5f[instance_key]
#             data_group = instance_group['data']
            
#             # Load entire sequences
#             tx = data_group['tx'][()]
#             ty = data_group['ty'][()]
#             tz = data_group['tz'][()]
#             qx = data_group['qx'][()]
#             qy = data_group['qy'][()]
#             qz = data_group['qz'][()]
#             qw = data_group['qw'][()]
#             distance_class = data_group['distance_class'][()]
            
#             # Load image sequence
#             image = data_group['image'][()]  # (seq_len, H, W, C)
            
#             seq_len = len(tx)
        
#         # Randomly sample a start timestep
#         max_start = max(0, seq_len - self.num_queries)
#         if max_start > 0:
#             start_ts = np.random.randint(0, max_start + 1)
#         else:
#             start_ts = 0
        
#         # qpos at start_ts (current state)
#         qpos_10d = construct_10d_state(
#             tx[start_ts], ty[start_ts], tz[start_ts],
#             qx[start_ts], qy[start_ts], qz[start_ts], qw[start_ts],
#             distance_class[start_ts]
#         )
        
#         # action sequence from start_ts onwards
#         action_seq = construct_10d_state(
#             tx[start_ts:], ty[start_ts:], tz[start_ts:],
#             qx[start_ts:], qy[start_ts:], qz[start_ts:], qw[start_ts:],
#             distance_class[start_ts:]
#         )  # (remaining_seq_len, 10)
        
#         action_len = len(action_seq)
        
#         # Pad action sequence to num_queries length
#         padded_action = np.zeros((self.num_queries, 10), dtype=np.float32)
#         if action_len > 0:
#             padded_action[:min(action_len, self.num_queries)] = action_seq[:self.num_queries]
        
#         is_pad = np.zeros(self.num_queries, dtype=bool)
#         if action_len < self.num_queries:
#             is_pad[action_len:] = True
        
#         # Process image at start_ts (current observation)
#         image_data = torch.from_numpy(image[start_ts:start_ts+1]).float()  # (1, H, W, C)
#         image_data = torch.einsum('t h w c -> t c h w', image_data)
#         image_data = image_data / 255.0  # Normalize to [0, 1]
        
#         # Repeat image for each query if needed (or keep single)
#         # Assuming single-camera setup, replicate for all queries
#         image_data = image_data.repeat(self.num_queries, 1, 1, 1)  # (num_queries, C, H, W)
        
#         # Convert to tensors
#         qpos_data = torch.from_numpy(qpos_10d).float()
#         action_data = torch.from_numpy(padded_action).float()
#         is_pad = torch.from_numpy(is_pad).bool()
        
#         return image_data, qpos_data, action_data, is_pad


# def get_instance_list(dataset_dir):
#     """Collect all instances from all H5 files in dataset_dir."""
#     instance_list = []
    
#     h5_files = [f for f in os.listdir(dataset_dir) if f.endswith('.h5')]
    
#     if not h5_files:
#         raise ValueError(f"No .h5 files found in {dataset_dir}")
    
#     for h5_file in h5_files:
#         h5_path = os.path.join(dataset_dir, h5_file)
        
#         with h5py.File(h5_path, 'r') as h5f:
#             instance_keys = [k for k in h5f.keys() if k.startswith('Instance_')]
            
#             for instance_key in instance_keys:
#                 instance_list.append((h5_path, instance_key))
    
#     print(f"Found {len(instance_list)} instances across {len(h5_files)} H5 files")
#     return instance_list


# def get_norm_stats(dataset_dir):
#     """
#     Load normalization stats from HDF5 files.
#     Stats are stored in global_normalization_params.
#     """
#     h5_files = [f for f in os.listdir(dataset_dir) if f.endswith('.h5')]
    
#     if h5_files:
#         h5_path = os.path.join(dataset_dir, h5_files[0])
#         with h5py.File(h5_path, 'r') as h5f:
#             if 'global_normalization_params' in h5f:
#                 norm_stats = {}
#                 for field in ['tx', 'ty', 'tz', 'distance']:
#                     if field in h5f['global_normalization_params']:
#                         field_group = h5f['global_normalization_params'][field]
#                         norm_stats[field] = {
#                             'mean': float(field_group['mean'][()]),
#                             'std': float(field_group['std'][()])
#                         }
                
#                 print("Loaded normalization stats from HDF5:")
#                 for field, stats in norm_stats.items():
#                     print(f"  {field}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
                
#                 return norm_stats
    
#     print("Warning: Could not load normalization stats from HDF5")
#     return {}


# def load_data(dataset_dir, batch_size_train, batch_size_val, num_queries=8):
#     """
#     Load data from task HDF5 files.
    
#     Args:
#         dataset_dir: directory containing .h5 files
#         num_episodes: (unused, kept for compatibility)
#         camera_names: (unused, kept for compatibility)
#         batch_size_train: training batch size
#         batch_size_val: validation batch size
#         num_queries: sequence length for action predictions
#     """
#     print(f'\nData from: {dataset_dir}\n')
#     print('Data structure: Instance-based, single camera')
#     print('qpos format: (tx, ty, tz, 6D_rotation, distance_class) at time t')
#     print('action format: sequence of (tx, ty, tz, 6D_rotation, distance_class) at times [t, t+1, ..., t+T]\n')
#     print(f'Sequence length (num_queries): {num_queries}\n')
    
#     # Get all instances
#     instance_list = get_instance_list(dataset_dir)
    
#     # Load normalization stats
#     norm_stats = get_norm_stats(dataset_dir)
    
#     # Split into train/val
#     num_instances = len(instance_list)
#     train_ratio = 0.75
#     shuffled_indices = np.random.permutation(num_instances)
#     train_indices = shuffled_indices[:int(train_ratio * num_instances)]
#     val_indices = shuffled_indices[int(train_ratio * num_instances):]
#     print(f'training: {train_indices}')
#     print(f'validation: {val_indices}')
    
#     train_instances = [instance_list[i] for i in train_indices]
#     val_instances = [instance_list[i] for i in val_indices]
    
#     # Create datasets
#     train_dataset = InstanceDataset(train_instances, num_queries)
#     val_dataset = InstanceDataset(val_instances, num_queries)
    
#     # Create dataloaders
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
#                                   pin_memory=True, num_workers=1, prefetch_factor=1)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True,
#                                pin_memory=True, num_workers=1, prefetch_factor=1)
    
#     print(f"Train instances: {len(train_instances)}")
#     print(f"Val instances: {len(val_instances)}\n")
    
#     is_sim = True
    
#     return train_dataloader, val_dataloader, norm_stats, is_sim


# def make_policy(policy_class, policy_config):
#     if policy_class == "ACT":
#         policy = ACTPolicy(policy_config)
#     elif policy_class == "CNNMLP":
#         policy = CNNMLPPolicy(policy_config)
#     else:
#         raise ValueError(f"Unknown policy class: {policy_class}")
#     return policy


# def make_optimizer(policy_class, policy):
#     if policy_class == 'ACT':
#         optimizer = policy.configure_optimizers()
#     elif policy_class == 'CNNMLP':
#         optimizer = policy.configure_optimizers()
#     else:
#         raise ValueError(f"Unknown policy class: {policy_class}")
#     return optimizer

# ### env utils
# def sample_box_pose():
#     x_range = [0.0, 0.2]
#     y_range = [0.4, 0.6]
#     z_range = [0.05, 0.05]

#     ranges = np.vstack([x_range, y_range, z_range])
#     cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

#     cube_quat = np.array([1, 0, 0, 0])
#     return np.concatenate([cube_position, cube_quat])

# def sample_insertion_pose():
#     # Peg
#     x_range = [0.1, 0.2]
#     y_range = [0.4, 0.6]
#     z_range = [0.05, 0.05]

#     ranges = np.vstack([x_range, y_range, z_range])
#     peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

#     peg_quat = np.array([1, 0, 0, 0])
#     peg_pose = np.concatenate([peg_position, peg_quat])

#     # Socket
#     x_range = [-0.2, -0.1]
#     y_range = [0.4, 0.6]
#     z_range = [0.05, 0.05]

#     ranges = np.vstack([x_range, y_range, z_range])
#     socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

#     socket_quat = np.array([1, 0, 0, 0])
#     socket_pose = np.concatenate([socket_position, socket_quat])

#     return peg_pose, socket_pose

# ### helper functions

# def get_image(images, camera_names, device='cpu'):
#     curr_images = []
#     for cam_name in camera_names:
#         curr_image = rearrange(images[cam_name], 'h w c -> c h w')
#         curr_images.append(curr_image)
#     curr_image = np.stack(curr_images, axis=0)
#     curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
#     return curr_image

# def compute_dict_mean(epoch_dicts):
#     result = {k: None for k in epoch_dicts[0]}
#     num_items = len(epoch_dicts)
#     for k in result:
#         value_sum = 0
#         for epoch_dict in epoch_dicts:
#             value_sum += epoch_dict[k]
#         result[k] = value_sum / num_items
#     return result

# def detach_dict(d):
#     new_d = dict()
#     for k, v in d.items():
#         new_d[k] = v.detach()
#     return new_d

# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)


# def pos2pwm(pos:np.ndarray) -> np.ndarray:
#     """
#     :param pos: numpy array of joint positions in range [-pi, pi]
#     :return: numpy array of pwm values in range [0, 4096]
#     """ 
#     return (pos / 3.14 + 1.) * 2048
    
# def pwm2pos(pwm:np.ndarray) -> np.ndarray:
#     """
#     :param pwm: numpy array of pwm values in range [0, 4096]
#     :return: numpy array of joint positions in range [-pi, pi]
#     """
#     return (pwm / 2048 - 1) * 3.14

# def pwm2vel(pwm:np.ndarray) -> np.ndarray:
#     """
#     :param pwm: numpy array of pwm/s joint velocities
#     :return: numpy array of rad/s joint velocities 
#     """
#     return pwm * 3.14 / 2048

# def vel2pwm(vel:np.ndarray) -> np.ndarray:
#     """
#     :param vel: numpy array of rad/s joint velocities
#     :return: numpy array of pwm/s joint velocities
#     """
#     return vel * 2048 / 3.14
    
# def pwm2norm(x:np.ndarray) -> np.ndarray:
#     """
#     :param x: numpy array of pwm values in range [0, 4096]
#     :return: numpy array of values in range [0, 1]
#     """
#     return x / 4096
    
# def norm2pwm(x:np.ndarray) -> np.ndarray:
#     """
#     :param x: numpy array of values in range [0, 1]
#     :return: numpy array of pwm values in range [0, 4096]
#     """
#     return x * 4096

# ###
# import os
# import h5py
# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from tqdm import tqdm

# from training.policy_1 import ACTPolicy, CNNMLPPolicy

# import IPython
# e = IPython.embed


# def quaternion_to_rot6d(quaternion):
#     """Convert quaternion (qx, qy, qz, qw) to 6D rotation representation."""
#     if isinstance(quaternion, np.ndarray):
#         quaternion = torch.from_numpy(quaternion).float()
    
#     q = F.normalize(quaternion, dim=-1)
#     qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
#     rot_matrix = torch.stack([
#         1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy),
#         2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx),
#         2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)
#     ], dim=-1)
    
#     rot_matrix = rot_matrix.reshape(*quaternion.shape[:-1], 3, 3)
#     rot_6d = rot_matrix[..., :2, :].reshape(*quaternion.shape[:-1], 6)
    
#     return rot_6d.numpy() if isinstance(rot_6d, torch.Tensor) else rot_6d


# def construct_10d_state_batch(tx, ty, tz, qx, qy, qz, qw, distance_class):
#     """
#     Construct 10D state for entire sequence: [tx, ty, tz, 6D_rotation, distance_class]
#     Pre-computed version for batch processing.
    
#     Args:
#         tx, ty, tz: arrays (seq_len,)
#         qx, qy, qz, qw: arrays (seq_len,)
#         distance_class: array (seq_len,)
    
#     Returns:
#         state_10d: (seq_len, 10) array
#     """
#     quaternion = np.stack([qx, qy, qz, qw], axis=-1)  # (seq_len, 4)
#     rot_6d = quaternion_to_rot6d(quaternion)  # (seq_len, 6)
    
#     state_10d = np.concatenate([
#         tx[:, None],
#         ty[:, None],
#         tz[:, None],
#         rot_6d,
#         distance_class[:, None]
#     ], axis=-1)  # (seq_len, 10)
    
#     return state_10d.astype(np.float32)


# class InstanceDataset(torch.utils.data.Dataset):
#     """Dataset that loads instances from task HDF5 files with pre-loading."""
    
#     def __init__(self, instance_list, num_queries, preload):
#         """
#         Args:
#             instance_list: list of (h5_path, instance_key) tuples
#             num_queries: sequence length for actions (number of future timesteps)
#         """
#         self.instance_list = instance_list
#         self.num_queries = num_queries
#         self.data_cache = {}
#         self.preload = preload
        
#         if self.preload:
#             print(f"Pre-loading {len(instance_list)} instances into memory...")
#             # Pre-load all data into memory
#             for idx, (h5_path, instance_key) in enumerate(tqdm(instance_list, desc="Loading data")):
#                 cache_key = (h5_path, instance_key)
                
#                 with h5py.File(h5_path, 'r') as h5f:
#                     instance_group = h5f[instance_key]
#                     data_group = instance_group['data']
                    
#                     # Load raw data
#                     tx = data_group['tx'][()].astype(np.float32)
#                     ty = data_group['ty'][()].astype(np.float32)
#                     tz = data_group['tz'][()].astype(np.float32)
#                     qx = data_group['qx'][()].astype(np.float32)
#                     qy = data_group['qy'][()].astype(np.float32)
#                     qz = data_group['qz'][()].astype(np.float32)
#                     qw = data_group['qw'][()].astype(np.float32)
#                     distance_class = data_group['distance_class'][()].astype(np.float32)
                    
#                     # Load and pre-process images
#                     image = data_group['image'][()]  # (seq_len, H, W, C)
                    
#                     # Pre-compute 10D state representation for entire sequence
#                     state_10d = construct_10d_state_batch(
#                         tx, ty, tz, qx, qy, qz, qw, distance_class
#                     )  # (seq_len, 10)
                    
#                     # Pre-process images: convert to torch, normalize, transpose
#                     # (seq_len, H, W, C) -> (seq_len, C, H, W)
#                     image_torch = torch.from_numpy(image).float()  # (seq_len, H, W, C)
#                     image_torch = image_torch.permute(0, 3, 1, 2)  # (seq_len, C, H, W)
#                     image_torch = image_torch / 255.0  # Normalize to [0, 1]
                    
#                     # Store in cache
#                     self.data_cache[cache_key] = {
#                         'state_10d': state_10d,  # (seq_len, 10)
#                         'images': image_torch,   # (seq_len, C, H, W)
#                         'seq_len': len(tx)
#                     }
            
#             print(f"✓ Pre-loaded {len(self.data_cache)} instances into memory")
            
#             # Calculate approximate memory usage
#             total_size_mb = sum(
#                 data['state_10d'].nbytes + data['images'].element_size() * data['images'].nelement()
#                 for data in self.data_cache.values()
#             ) / (1024 * 1024)
#             print(f"  Approximate memory usage: {total_size_mb:.2f} MB")
#         else:
#             print(f"On-demand loading enabled for {len(instance_list)} instances")

#     def __len__(self):
#         return len(self.instance_list)

#     def __getitem__(self, index):
#         h5_path, instance_key = self.instance_list[index]
#         cache_key = (h5_path, instance_key)
        
#         # Retrieve pre-loaded data
#         cached_data = self.data_cache[cache_key]
#         state_10d = cached_data['state_10d']  # (seq_len, 10)
#         images = cached_data['images']  # (seq_len, C, H, W)
#         seq_len = cached_data['seq_len']
        
#         # Randomly sample a start timestep
#         max_start = max(0, seq_len - self.num_queries)
#         if max_start > 0:
#             start_ts = np.random.randint(0, max_start + 1)
#         else:
#             start_ts = 0
        
#         # qpos at start_ts (current state)
#         qpos_10d = state_10d[start_ts]  # (10,)
        
#         # action sequence from start_ts onwards
#         action_seq = state_10d[start_ts:]  # (remaining_seq_len, 10)
#         action_len = len(action_seq)
        
#         # Pad action sequence to num_queries length
#         padded_action = np.zeros((self.num_queries, 10), dtype=np.float32)
#         if action_len > 0:
#             padded_action[:min(action_len, self.num_queries)] = action_seq[:self.num_queries]
        
#         is_pad = np.zeros(self.num_queries, dtype=bool)
#         if action_len < self.num_queries:
#             is_pad[action_len:] = True
        
#         # Get image at start_ts (current observation)
#         image_data = images[start_ts:start_ts+1]  # (1, C, H, W)
        
#         # Repeat image for each query
#         image_data = image_data.repeat(self.num_queries, 1, 1, 1)  # (num_queries, C, H, W)
        
#         # Convert to tensors
#         qpos_data = torch.from_numpy(qpos_10d).float()
#         action_data = torch.from_numpy(padded_action).float()
#         is_pad = torch.from_numpy(is_pad).bool()
        
#         return image_data, qpos_data, action_data, is_pad


# def get_instance_list(dataset_dir):
#     """Collect all instances from all H5 files in dataset_dir."""
#     instance_list = []
    
#     h5_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.h5')])
    
#     if not h5_files:
#         raise ValueError(f"No .h5 files found in {dataset_dir}")
    
#     for h5_file in h5_files:
#         h5_path = os.path.join(dataset_dir, h5_file)
        
#         with h5py.File(h5_path, 'r') as h5f:
#             instance_keys = sorted([k for k in h5f.keys() if k.startswith('Instance_')])
            
#             for instance_key in instance_keys:
#                 instance_list.append((h5_path, instance_key))
    
#     print(f"Found {len(instance_list)} instances across {len(h5_files)} H5 files")
#     return instance_list


# def get_norm_stats(dataset_dir):
#     """
#     Load normalization stats from HDF5 files.
#     Stats are stored in global_normalization_params.
#     """
#     h5_files = [f for f in os.listdir(dataset_dir) if f.endswith('.h5')]
    
#     if h5_files:
#         h5_path = os.path.join(dataset_dir, h5_files[0])
#         with h5py.File(h5_path, 'r') as h5f:
#             if 'global_normalization_params' in h5f:
#                 norm_stats = {}
#                 for field in ['tx', 'ty', 'tz', 'distance']:
#                     if field in h5f['global_normalization_params']:
#                         field_group = h5f['global_normalization_params'][field]
#                         norm_stats[field] = {
#                             'mean': float(field_group['mean'][()]),
#                             'std': float(field_group['std'][()])
#                         }
                
#                 print("Loaded normalization stats from HDF5:")
#                 for field, stats in norm_stats.items():
#                     print(f"  {field}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
                
#                 return norm_stats
    
#     print("Warning: Could not load normalization stats from HDF5")
#     return {}


# def load_data(dataset_dir, batch_size_train, batch_size_val, num_queries=8, data_loader_config=None):
#     """
#     Load data from task HDF5 files with configurable pre-loading optimization.
#     """
#     # Import config if not provided
#     if data_loader_config is None:
#         from config.config_1 import DATA_LOADER_CONFIG
#         data_loader_config = DATA_LOADER_CONFIG
    
#     print(f'\nData from: {dataset_dir}\n')
#     print('Data structure: Instance-based, single camera')
#     print('qpos format: (tx, ty, tz, 6D_rotation, distance_class) at time t')
#     print('action format: sequence of (tx, ty, tz, 6D_rotation, distance_class) at times [t, t+1, ..., t+T]\n')
#     print(f'Sequence length (num_queries): {num_queries}\n')
#     print(f'Pre-loading enabled: {data_loader_config["preload_data"]}\n')
    
#     # Get all instances
#     instance_list = get_instance_list(dataset_dir)
    
#     # Load normalization stats
#     norm_stats = get_norm_stats(dataset_dir)
    
#     # Split into train/val
#     num_instances = len(instance_list)
#     train_ratio = 64.0/88.0
#     shuffled_indices = np.random.permutation(num_instances)
#     train_indices = shuffled_indices[:int(train_ratio * num_instances)]
#     val_indices = shuffled_indices[int(train_ratio * num_instances):]
    
#     train_instances = [instance_list[i] for i in train_indices]
#     val_instances = [instance_list[i] for i in val_indices]
    
#     print(f"Train instances: {len(train_instances)}")
#     print(f"Val instances: {len(val_instances)}\n")
    
#     # Create datasets (with optional pre-loading)
#     train_dataset = InstanceDataset(
#         train_instances, 
#         num_queries,
#         preload=data_loader_config['preload_data']
#     )
#     val_dataset = InstanceDataset(
#         val_instances, 
#         num_queries,
#         preload=data_loader_config['preload_data']
#     )
    
#     # ✅ Prepare DataLoader kwargs based on num_workers
#     num_workers_train = data_loader_config['num_workers_train']
#     num_workers_val = data_loader_config['num_workers_val']
    
#     # Build kwargs for training dataloader
#     train_kwargs = {
#         'batch_size': batch_size_train,
#         'shuffle': True,
#         'pin_memory': data_loader_config['pin_memory'],
#         'num_workers': num_workers_train
#     }
    
#     # Only add prefetch_factor and persistent_workers if num_workers > 0
#     if num_workers_train > 0:
#         train_kwargs['prefetch_factor'] = data_loader_config['prefetch_factor']
#         train_kwargs['persistent_workers'] = data_loader_config['persistent_workers']
    
#     # Build kwargs for validation dataloader
#     val_kwargs = {
#         'batch_size': batch_size_val,
#         'shuffle': False,
#         'pin_memory': data_loader_config['pin_memory'],
#         'num_workers': num_workers_val
#     }
    
#     # Only add prefetch_factor and persistent_workers if num_workers > 0
#     if num_workers_val > 0:
#         val_kwargs['prefetch_factor'] = data_loader_config['prefetch_factor']
#         val_kwargs['persistent_workers'] = data_loader_config['persistent_workers']
    
#     # Create dataloaders
#     train_dataloader = DataLoader(train_dataset, **train_kwargs)
#     val_dataloader = DataLoader(val_dataset, **val_kwargs)
    
#     is_sim = True
    
#     return train_dataloader, val_dataloader, norm_stats, is_sim


# def make_policy(policy_class, policy_config):
#     if policy_class == "ACT":
#         policy = ACTPolicy(policy_config)
#     elif policy_class == "CNNMLP":
#         policy = CNNMLPPolicy(policy_config)
#     else:
#         raise ValueError(f"Unknown policy class: {policy_class}")
#     return policy


# def make_optimizer(policy_class, policy):
#     if policy_class == 'ACT':
#         optimizer = policy.configure_optimizers()
#     elif policy_class == 'CNNMLP':
#         optimizer = policy.configure_optimizers()
#     else:
#         raise ValueError(f"Unknown policy class: {policy_class}")
#     return optimizer

# ### env utils

# def sample_box_pose():
#     x_range = [0.0, 0.2]
#     y_range = [0.4, 0.6]
#     z_range = [0.05, 0.05]

#     ranges = np.vstack([x_range, y_range, z_range])
#     cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

#     cube_quat = np.array([1, 0, 0, 0])
#     return np.concatenate([cube_position, cube_quat])

# def sample_insertion_pose():
#     # Peg
#     x_range = [0.1, 0.2]
#     y_range = [0.4, 0.6]
#     z_range = [0.05, 0.05]

#     ranges = np.vstack([x_range, y_range, z_range])
#     peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

#     peg_quat = np.array([1, 0, 0, 0])
#     peg_pose = np.concatenate([peg_position, peg_quat])

#     # Socket
#     x_range = [-0.2, -0.1]
#     y_range = [0.4, 0.6]
#     z_range = [0.05, 0.05]

#     ranges = np.vstack([x_range, y_range, z_range])
#     socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

#     socket_quat = np.array([1, 0, 0, 0])
#     socket_pose = np.concatenate([socket_position, socket_quat])

#     return peg_pose, socket_pose

# ### helper functions

# def get_image(images, camera_names, device='cpu'):
#     curr_images = []
#     for cam_name in camera_names:
#         curr_image = rearrange(images[cam_name], 'h w c -> c h w')
#         curr_images.append(curr_image)
#     curr_image = np.stack(curr_images, axis=0)
#     curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
#     return curr_image

# def compute_dict_mean(epoch_dicts):
#     result = {k: None for k in epoch_dicts[0]}
#     num_items = len(epoch_dicts)
#     for k in result:
#         value_sum = 0
#         for epoch_dict in epoch_dicts:
#             value_sum += epoch_dict[k]
#         result[k] = value_sum / num_items
#     return result

# def detach_dict(d):
#     new_d = dict()
#     for k, v in d.items():
#         new_d[k] = v.detach()
#     return new_d

# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)


# def pos2pwm(pos:np.ndarray) -> np.ndarray:
#     """
#     :param pos: numpy array of joint positions in range [-pi, pi]
#     :return: numpy array of pwm values in range [0, 4096]
#     """ 
#     return (pos / 3.14 + 1.) * 2048
    
# def pwm2pos(pwm:np.ndarray) -> np.ndarray:
#     """
#     :param pwm: numpy array of pwm values in range [0, 4096]
#     :return: numpy array of joint positions in range [-pi, pi]
#     """
#     return (pwm / 2048 - 1) * 3.14

# def pwm2vel(pwm:np.ndarray) -> np.ndarray:
#     """
#     :param pwm: numpy array of pwm/s joint velocities
#     :return: numpy array of rad/s joint velocities 
#     """
#     return pwm * 3.14 / 2048

# def vel2pwm(vel:np.ndarray) -> np.ndarray:
#     """
#     :param vel: numpy array of rad/s joint velocities
#     :return: numpy array of pwm/s joint velocities
#     """
#     return vel * 2048 / 3.14
    
# def pwm2norm(x:np.ndarray) -> np.ndarray:
#     """
#     :param x: numpy array of pwm values in range [0, 4096]
#     :return: numpy array of values in range [0, 1]
#     """
#     return x / 4096
    
# def norm2pwm(x:np.ndarray) -> np.ndarray:
#     """
#     :param x: numpy array of values in range [0, 1]
#     :return: numpy array of pwm values in range [0, 4096]
#     """
#     return x * 4096


# ## augmentation
# import os
# import h5py
# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import torchvision.transforms as transforms # Added imports
# from tqdm import tqdm

# from training.policy_1 import ACTPolicy, CNNMLPPolicy

# import IPython
# e = IPython.embed


# def quaternion_to_rot6d(quaternion):
#     """Convert quaternion (qx, qy, qz, qw) to 6D rotation representation."""
#     if isinstance(quaternion, np.ndarray):
#         quaternion = torch.from_numpy(quaternion).float()
    
#     q = F.normalize(quaternion, dim=-1)
#     qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
#     rot_matrix = torch.stack([
#         1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy),
#         2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx),
#         2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)
#     ], dim=-1)
    
#     rot_matrix = rot_matrix.reshape(*quaternion.shape[:-1], 3, 3)
#     rot_6d = rot_matrix[..., :2, :].reshape(*quaternion.shape[:-1], 6)
    
#     return rot_6d.numpy() if isinstance(rot_6d, torch.Tensor) else rot_6d


# def construct_10d_state_batch(tx, ty, tz, qx, qy, qz, qw, distance_class):
#     """
#     Construct 10D state for entire sequence: [tx, ty, tz, 6D_rotation, distance_class]
#     Pre-computed version for batch processing.
#     """
#     quaternion = np.stack([qx, qy, qz, qw], axis=-1)  # (seq_len, 4)
#     rot_6d = quaternion_to_rot6d(quaternion)  # (seq_len, 6)
    
#     state_10d = np.concatenate([
#         tx[:, None],
#         ty[:, None],
#         tz[:, None],
#         rot_6d,
#         distance_class[:, None]
#     ], axis=-1)  # (seq_len, 10)
    
#     return state_10d.astype(np.float32)


# class InstanceDataset(torch.utils.data.Dataset):
#     """Dataset that loads instances from task HDF5 files with pre-loading."""
    
#     def __init__(self, instance_list, num_queries, preload, augment=False):
#         """
#         Args:
#             instance_list: list of (h5_path, instance_key) tuples
#             num_queries: sequence length for actions
#             preload: whether to load all data into RAM
#             augment: (bool) whether to apply data augmentation (True for train, False for val)
#         """
#         self.instance_list = instance_list
#         self.num_queries = num_queries
#         self.data_cache = {}
#         self.preload = preload
#         self.augment = augment
        
#         # --- Define Augmentations ---
#         self.transform = None
#         if self.augment:
#             print("Augmentation Enabled: RandomCrop + ColorJitter + State Noise")
#             self.transform = transforms.Compose([
#                 # Randomly shift the image by padding 4 pixels and cropping back
#                 transforms.RandomCrop(size=(480, 640), padding=4), 
#                 # Randomly change brightness, contrast, etc.
#                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
#             ])
#             # Note: We hardcode 480x640 here. If your image size is different, 
#             # the RandomCrop will error. Let's make it dynamic in __getitem__ 
#             # or rely on the fact that ACT usually uses 480x640.
#         else:
#             self.transform = torch.nn.Identity()

#         if self.preload:
#             print(f"Pre-loading {len(instance_list)} instances into memory...")
#             for idx, (h5_path, instance_key) in enumerate(tqdm(instance_list, desc="Loading data")):
#                 cache_key = (h5_path, instance_key)
                
#                 with h5py.File(h5_path, 'r') as h5f:
#                     instance_group = h5f[instance_key]
#                     data_group = instance_group['data']
                    
#                     tx = data_group['tx'][()].astype(np.float32)
#                     ty = data_group['ty'][()].astype(np.float32)
#                     tz = data_group['tz'][()].astype(np.float32)
#                     qx = data_group['qx'][()].astype(np.float32)
#                     qy = data_group['qy'][()].astype(np.float32)
#                     qz = data_group['qz'][()].astype(np.float32)
#                     qw = data_group['qw'][()].astype(np.float32)
#                     distance_class = data_group['distance_class'][()].astype(np.float32)
                    
#                     image = data_group['image'][()]
                    
#                     state_10d = construct_10d_state_batch(
#                         tx, ty, tz, qx, qy, qz, qw, distance_class
#                     )
                    
#                     image_torch = torch.from_numpy(image).float()
#                     image_torch = image_torch.permute(0, 3, 1, 2)
#                     image_torch = image_torch / 255.0  # Keep in [0, 1]
                    
#                     self.data_cache[cache_key] = {
#                         'state_10d': state_10d,
#                         'images': image_torch,
#                         'seq_len': len(tx)
#                     }
            
#             print(f"✓ Pre-loaded {len(self.data_cache)} instances")

#     def __len__(self):
#         return len(self.instance_list)

#     def __getitem__(self, index):
#         h5_path, instance_key = self.instance_list[index]
#         cache_key = (h5_path, instance_key)
        
#         # Retrieve pre-loaded data
#         cached_data = self.data_cache[cache_key]
#         state_10d = cached_data['state_10d']
#         images = cached_data['images']
#         seq_len = cached_data['seq_len']
        
#         # Sample start timestep
#         max_start = max(0, seq_len - self.num_queries)
#         start_ts = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        
#         # --- State Handling ---
#         qpos_10d = state_10d[start_ts].copy() # Copy to avoid modifying cache
        
#         # 3. Add Noise to Robot State (Training Only)
#         if self.augment:
#             # Add Gaussian noise to everything: position, rotation, gripper
#             # Sigma = 0.005 is safe for normalized/small scale data
#             noise = np.random.normal(loc=0, scale=0.005, size=qpos_10d.shape).astype(np.float32)
#             qpos_10d = qpos_10d + noise
        
#         action_seq = state_10d[start_ts:]
#         action_len = len(action_seq)
        
#         padded_action = np.zeros((self.num_queries, 10), dtype=np.float32)
#         if action_len > 0:
#             padded_action[:min(action_len, self.num_queries)] = action_seq[:self.num_queries]
        
#         is_pad = np.zeros(self.num_queries, dtype=bool)
#         if action_len < self.num_queries:
#             is_pad[action_len:] = True
        
#         # --- Image Handling ---
#         # Get single image frame
#         image_data = images[start_ts]  # (C, H, W)
        
#         # 1 & 2. Apply Image Augmentations (Training Only)
#         if self.augment:
#             # Helper to handle dynamic sizes for RandomCrop
#             _, h, w = image_data.shape
#             # Re-initialize transform if size is needed (or just use padding trick)
#             augmenter = transforms.Compose([
#                 transforms.RandomCrop(size=(h, w), padding=4), # Pad 4 then crop to original
#                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
#             ])
#             image_data = augmenter(image_data)
            
#         # Add batch dim for repeating
#         image_data = image_data.unsqueeze(0) # (1, C, H, W)
#         image_data = image_data.repeat(self.num_queries, 1, 1, 1)
        
#         # Convert to tensors
#         qpos_data = torch.from_numpy(qpos_10d).float()
#         action_data = torch.from_numpy(padded_action).float()
#         is_pad = torch.from_numpy(is_pad).bool()
        
#         return image_data, qpos_data, action_data, is_pad


# def get_instance_list(dataset_dir):
#     instance_list = []
#     h5_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.h5')])
#     for h5_file in h5_files:
#         h5_path = os.path.join(dataset_dir, h5_file)
#         with h5py.File(h5_path, 'r') as h5f:
#             instance_keys = sorted([k for k in h5f.keys() if k.startswith('Instance_')])
#             for instance_key in instance_keys:
#                 instance_list.append((h5_path, instance_key))
#     return instance_list


# def get_norm_stats(dataset_dir):
#     # (Same as your original code, omitted for brevity but should be kept)
#     h5_files = [f for f in os.listdir(dataset_dir) if f.endswith('.h5')]
#     if h5_files:
#         h5_path = os.path.join(dataset_dir, h5_files[0])
#         with h5py.File(h5_path, 'r') as h5f:
#             if 'global_normalization_params' in h5f:
#                 norm_stats = {}
#                 for field in ['tx', 'ty', 'tz', 'distance']:
#                     if field in h5f['global_normalization_params']:
#                         field_group = h5f['global_normalization_params'][field]
#                         norm_stats[field] = {
#                             'mean': float(field_group['mean'][()]),
#                             'std': float(field_group['std'][()])
#                         }
#                 return norm_stats
#     return {}


# def load_data(dataset_dir, batch_size_train, batch_size_val, num_queries=8, data_loader_config=None):
#     if data_loader_config is None:
#         from config.config_1 import DATA_LOADER_CONFIG
#         data_loader_config = DATA_LOADER_CONFIG
    
#     print(f'\nData from: {dataset_dir}\n')
    
#     instance_list = get_instance_list(dataset_dir)
#     norm_stats = get_norm_stats(dataset_dir)
    
#     num_instances = len(instance_list)
#     train_ratio = 48/64.0
#     shuffled_indices = np.random.permutation(num_instances)
#     train_indices = shuffled_indices[:int(train_ratio * num_instances)]
#     val_indices = shuffled_indices[int(train_ratio * num_instances):]
    
#     train_instances = [instance_list[i] for i in train_indices]
#     val_instances = [instance_list[i] for i in val_indices]
    
#     print(f"Train instances: {len(train_instances)} (Augmentation: ENABLED)")
#     print(f"Val instances: {len(val_instances)} (Augmentation: DISABLED)\n")
    
#     # --- Pass augment flag here ---
#     train_dataset = InstanceDataset(
#         train_instances, 
#         num_queries,
#         preload=data_loader_config['preload_data'],
#         augment=True  # <--- Augmentation ON for Training
#     )
#     val_dataset = InstanceDataset(
#         val_instances, 
#         num_queries,
#         preload=data_loader_config['preload_data'],
#         augment=False # <--- Augmentation OFF for Validation
#     )
    
#     num_workers_train = data_loader_config['num_workers_train']
#     num_workers_val = data_loader_config['num_workers_val']
    
#     train_kwargs = {
#         'batch_size': batch_size_train,
#         'shuffle': True,
#         'pin_memory': data_loader_config['pin_memory'],
#         'num_workers': num_workers_train
#     }
    
#     if num_workers_train > 0:
#         train_kwargs['prefetch_factor'] = data_loader_config['prefetch_factor']
#         train_kwargs['persistent_workers'] = data_loader_config['persistent_workers']
    
#     val_kwargs = {
#         'batch_size': batch_size_val,
#         'shuffle': False,
#         'pin_memory': data_loader_config['pin_memory'],
#         'num_workers': num_workers_val
#     }
    
#     if num_workers_val > 0:
#         val_kwargs['prefetch_factor'] = data_loader_config['prefetch_factor']
#         val_kwargs['persistent_workers'] = data_loader_config['persistent_workers']
    
#     train_dataloader = DataLoader(train_dataset, **train_kwargs)
#     val_dataloader = DataLoader(val_dataset, **val_kwargs)
    
#     is_sim = True
    
#     return train_dataloader, val_dataloader, norm_stats, is_sim

# # ... keep make_policy, make_optimizer etc. unchanged ...
# def make_policy(policy_class, policy_config):
#     if policy_class == "ACT":
#         policy = ACTPolicy(policy_config)
#     elif policy_class == "CNNMLP":
#         policy = CNNMLPPolicy(policy_config)
#     else:
#         raise ValueError(f"Unknown policy class: {policy_class}")
#     return policy

# def make_optimizer(policy_class, policy):
#     if policy_class == 'ACT':
#         optimizer = policy.configure_optimizers()
#     elif policy_class == 'CNNMLP':
#         optimizer = policy.configure_optimizers()
#     else:
#         raise ValueError(f"Unknown policy class: {policy_class}")
#     return optimizer
    
# def get_image(images, camera_names, device='cpu'):
#     curr_images = []
#     for cam_name in camera_names:
#         curr_image = rearrange(images[cam_name], 'h w c -> c h w')
#         curr_images.append(curr_image)
#     curr_image = np.stack(curr_images, axis=0)
#     curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
#     return curr_image

# def compute_dict_mean(epoch_dicts):
#     result = {k: None for k in epoch_dicts[0]}
#     num_items = len(epoch_dicts)
#     for k in result:
#         value_sum = 0
#         for epoch_dict in epoch_dicts:
#             value_sum += epoch_dict[k]
#         result[k] = value_sum / num_items
#     return result

# def detach_dict(d):
#     new_d = dict()
#     for k, v in d.items():
#         new_d[k] = v.detach()
#     return new_d

# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)

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