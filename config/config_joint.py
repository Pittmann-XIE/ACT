import os
# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory
DATA_DIR = '../dataset/joint/all_data_merged.h5'

# checkpoint directory
CHECKPOINT_DIR = './checkpoints/joint/pick_aria_realsense'

# wandb
WANDB_NAME = '20260126_joint_aria_realsense'
USE_WANDB = True

# device
device = 'cuda:0'
#if torch.backends.mps.is_available(): device = 'mps'
os.environ['DEVICE'] = device

# robot port names
ROBOT_PORTS = {
    'leader': '/dev/tty.usbmodem57380045221',
    'follower': '/dev/tty.usbmodem57380046991'
}


# task config (you can add new tasks)
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 1000, ### FIXME
    'state_dim': 7,
    'action_dim': 7,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['aria', 'realsense'],
    'camera_port': 0
}


# policy config
POLICY_CONFIG = {
    'lr': 1e-5,
    'device': device,
    'num_queries': 100,
    'kl_weight': 0.00155,
    'dist_weight': 1.0,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 2e-5,
    'backbone': 'resnet34',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': TASK_CONFIG['camera_names'],
    'policy_class': 'ACT',
    'temporal_agg': True,
    'action_dim': TASK_CONFIG['action_dim'],
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 2000,#2000/8
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_last.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR, 
    'wandb_name': WANDB_NAME
}

# Data loading
DATA_LOADER_CONFIG = {
    'preload_data': True,           # Pre-load all data into memory (fixes Issue 1)
    'num_workers_train': 0,         # DataLoader workers for training
    'num_workers_val': 0,           # DataLoader workers for validation
    'prefetch_factor': 2,           # Batches to prefetch per worker
    'persistent_workers': False,     # Keep workers alive between epochs
    'pin_memory': True,              # Use pinned memory for faster GPU transfer
}

