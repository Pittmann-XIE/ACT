## keep the original model

import os
# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory
DATA_DIR = '/mnt/Ego2Exo/pick_teleop_2/realsense/middle_corner_3rd_vertical/reduced/smooth/normalize/final_merged.h5py'

# wandb
USE_WANDB = False
WANDB_NAME = '20260130_pick_and_place_single_place_kl'

# checkpoint directory
CHECKPOINT_DIR = f'./checkpoints/joint/{WANDB_NAME}'


# device
device = 'cuda'

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
    'episode_len': 900, ### FIXME
    'state_dim': 7,
    'action_dim': 7,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['cam1_rgb', 'cam2_rgb'], 
    'camera_port': 0
}


# policy config
POLICY_CONFIG = {
    'lr': 1e-5,
    'device': device,
    'num_queries': 100,
    'kl_weight': 10.0, ## 0.0003
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': TASK_CONFIG['camera_names'],
    'policy_class': 'ACT',
    'temporal_agg': True
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 2000,#2000/8
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_last.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR,
    'wandb_run_name' : WANDB_NAME,
    'train_ratio': 0.692307692 # 80/97
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


