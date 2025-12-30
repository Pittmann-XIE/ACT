import os
# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory
DATA_DIR = '/home/pengtao/thesis/ws_ros2humble-main_lab/datatset/5_6_7'

# checkpoint directory
CHECKPOINT_DIR = 'checkpoints/'

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
    'episode_len': 300,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['front'],
    'camera_port': 0
}


# policy config
POLICY_CONFIG = {
    'lr': 6.07e-6,
    'device': device,
    'num_queries': 30,
    'kl_weight': 1.754568,
    'trans_weight': 0.035264,
    'rot_weight': 0.027659,
    'dist_weight': 0.021186,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['front'],
    'policy_class': 'ACT',
    'temporal_agg': True,
    'position_encoding': "sine"
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 2000/8,  # since we use 8 samples per episode
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_last.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR,
    'ckpt_interval': 15
}

# Data loading
DATA_LOADER_CONFIG = {
    'preload_data': True,           # Pre-load all data into memory (fixes Issue 1)
    'num_workers_train': 0,         # DataLoader workers for training
    'num_workers_val': 0,           # DataLoader workers for validation
    'prefetch_factor': 2,           # Batches to prefetch per worker
    'persistent_workers': False,     # Keep workers alive between epochs
    'pin_memory': True              # Use pinned memory for faster GPU transfer
}

# wandb
USE_WANDB = True