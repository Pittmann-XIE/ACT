# ## Step-Based Training Script (Fixed Logging & Resume)
# from config.config_joint_1_step import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, USE_WANDB, DATA_LOADER_CONFIG

# import os
# import pickle
# import argparse
# from copy import deepcopy
# import matplotlib.pyplot as plt
# import wandb
# import torch
# import numpy as np
# import sys
# import time

# from training.utils_joint_1_step import *

# # parse args
# parser = argparse.ArgumentParser()
# parser.add_argument('--task', type=str, default='pick')
# parser.add_argument('--checkpoint', type=str, default='/home/chavvive/ws_ros2humble-main_lab/ACT/checkpoints/joint/20260214_pick_and_place_66_realsense/pick/policy_step_50004.ckpt', help='Path to .ckpt file to resume training from')
# # ADDED: Support for manual start step (crucial for resuming/continual training)
# parser.add_argument('--start_step', type=int, default=None, help='Force execution to start from this step (overrides checkpoint)')
# args = parser.parse_args()

# task = args.task
# ckpt_path_args = args.checkpoint
# start_step_args = args.start_step

# # configs
# task_cfg = TASK_CONFIG
# train_cfg = TRAIN_CONFIG
# policy_config = POLICY_CONFIG
# data_loader_config = DATA_LOADER_CONFIG
# checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task)
# device = os.environ['DEVICE']

# def forward_pass(data, policy):
#     image_data, qpos_data, action_data, is_pad = data
#     image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
#     return policy(qpos_data, image_data, action_data, is_pad)

# def train_bc(train_dataloader, val_dataloader, policy_config, resume_ckpt=None, manual_start_step=None):
    
#     # --- CONFIG SETUP ---
#     MAX_STEPS = train_cfg.get('max_steps', 100000)
#     EVAL_INTERVAL = train_cfg.get('eval_interval', 2000)
#     SAVE_INTERVAL = train_cfg.get('save_interval', 2000)
#     PRINT_INTERVAL = 100
    
#     print("\n" + "="*80)
#     print(f"STARTING STEP-BASED TRAINING")
#     print(f"Max Steps: {MAX_STEPS} | Eval Interval: {EVAL_INTERVAL} | Save Interval: {SAVE_INTERVAL}")
#     print("="*80 + "\n")
    
#     # 1. Initialize
#     policy = make_policy(policy_config['policy_class'], policy_config)
#     policy.to(device)
#     optimizer = make_optimizer(policy_config['policy_class'], policy)
    
#     current_step = 0

#     # 2. Resume Logic
#     if resume_ckpt is not None:
#         if os.path.isfile(resume_ckpt):
#             print(f"Loading checkpoint from: {resume_ckpt}")
#             checkpoint = torch.load(resume_ckpt, map_location=device)
            
#             # Load Weights
#             if 'model_state_dict' in checkpoint:
#                 policy.load_state_dict(checkpoint['model_state_dict'])
#                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
#                 # Checkpoint Step
#                 ckpt_step = checkpoint.get('step', 0)
#                 print(f"   Checkpoint reports step: {ckpt_step}")
#                 current_step = ckpt_step
#             else:
#                 # Weights only (Legacy)
#                 policy.load_state_dict(checkpoint)
#                 print("   Loaded weights-only checkpoint.")
                
#         else:
#             print(f"Warning: Checkpoint {resume_ckpt} not found. Starting from scratch.")

#     # Override step if provided manually (Continual Training support)
#     if manual_start_step is not None:
#         print(f"   Overriding start step to: {manual_start_step}")
#         current_step = manual_start_step

#     os.makedirs(checkpoint_dir, exist_ok=True)

#     # 3. Create Infinite Iterator
#     train_iterator = cycle(train_dataloader)
    
#     policy.train()
    
#     # 4. Main Step Loop
#     # Loop range handles the resumption automatically
#     for step in range(current_step, MAX_STEPS + 1):
        
#         # --- VALIDATION ---
#         if step % EVAL_INTERVAL == 0 and step != 0:
#             print(f"\nStep {step}: Validating...")
#             policy.eval()
#             with torch.inference_mode():
#                 # We collect ALL metrics during validation now
#                 val_losses = [] # Just for printing average total loss
#                 val_metrics_sum = {} # To average all components (kl, l1, etc)
#                 num_val_batches = 50 
                
#                 for i, data in enumerate(val_dataloader):
#                     if i >= num_val_batches: break
#                     forward_dict = forward_pass(data, policy)
                    
#                     # Accumulate all keys (loss, l1, kl, etc.)
#                     for k, v in forward_dict.items():
#                         v_item = v.item()
#                         if k not in val_metrics_sum: val_metrics_sum[k] = 0.0
#                         val_metrics_sum[k] += v_item
                    
#                     val_losses.append(forward_dict['loss'].item())
                
#                 # Compute Averages
#                 avg_val_loss = np.mean(val_losses)
#                 val_metrics_avg = {k: v / len(val_losses) for k, v in val_metrics_sum.items()}
                
#                 # Print formatted validation metrics
#                 val_str = f"   Val Loss: {avg_val_loss:.5f} | "
#                 val_str += " ".join([f"{k}: {v:.5f}" for k, v in val_metrics_avg.items() if k != 'loss'])
#                 print(val_str)
                
#                 if USE_WANDB:
#                     wandb.log({f'val/{k}': v for k, v in val_metrics_avg.items()})
#                     wandb.log({'step': step})

#             policy.train()

#         # --- TRAINING ---
#         t0 = time.time()
        
#         data = next(train_iterator)
        
#         forward_dict = forward_pass(data, policy)
#         loss = forward_dict['loss']
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # --- LOGGING (Restored Detailed Breakdown) ---
#         if step % PRINT_INTERVAL == 0:
#             t1 = time.time()
            
#             # Create a string for all loss components (kl, l1)
#             # This loops over everything ACT returns in forward_dict
#             metrics_str = ""
#             wandb_dict = {'step': step}
            
#             for k, v in forward_dict.items():
#                 v_item = v.item()
#                 metrics_str += f"{k}: {v_item:.4f} "
#                 wandb_dict[f'train/{k}'] = v_item

#             print(f"Step {step} | {metrics_str}| Time: {t1-t0:.3f}s")
            
#             if USE_WANDB:
#                 wandb.log(wandb_dict)

#         # --- CHECKPOINTING ---
#         if step > 0 and step % SAVE_INTERVAL == 0:
#             ckpt_path = os.path.join(checkpoint_dir, f'policy_step_{step}.ckpt')
#             save_payload = {
#                 'model_state_dict': policy.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'step': step,
#                 'train_loss': loss.item(),
#             }
#             torch.save(save_payload, ckpt_path)
#             print(f"   Saved checkpoint: {os.path.basename(ckpt_path)}")
            
#     # Final Save
#     ckpt_path = os.path.join(checkpoint_dir, f'policy_last.ckpt')
#     torch.save({
#         'model_state_dict': policy.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'step': MAX_STEPS,
#     }, ckpt_path)
#     print("Training Finished.")


# if __name__ == '__main__':
    
#     print("\n" + "="*80)
#     print("INITIALIZATION")
#     print("="*80 + "\n")
    
#     if USE_WANDB:
#         wandb_name = f"{task}_{train_cfg['seed']}_{train_cfg['wandb_run_name']}"
#         if ckpt_path_args:
#             wandb_name += "_resumed"

#         wandb.init(
#             project="ACT_origin",
#             name=wandb_name,
#             config={
#                 "task": task,
#                 **{f"task_cfg/{k}": v for k, v in TASK_CONFIG.items()},
#                 **{f"train_cfg/{k}": v for k, v in TRAIN_CONFIG.items()},
#                 **{f"policy_cfg/{k}": v for k, v in POLICY_CONFIG.items()},
#             }
#         )
    
#     set_seed(train_cfg['seed'])
#     os.makedirs(checkpoint_dir, exist_ok=True)
    
#     data_dir = task_cfg['dataset_dir']
#     num_queries = policy_config.get('num_queries', 8)
    
#     print("\n" + "="*80)
#     print("LOADING DATA (Calculating 100% Coverage indices...)")
#     print("="*80)
    
#     train_dataloader, val_dataloader, stats, _ = load_data(
#         data_dir, 
#         batch_size_train=train_cfg['batch_size_train'], 
#         batch_size_val=train_cfg['batch_size_val'],
#         num_queries=num_queries,
#         data_loader_config=data_loader_config,
#         camera_names=policy_config['camera_names'],
#         train_ratio = train_cfg['train_ratio'],
#         state_dim=TASK_CONFIG['state_dim']
#     )
    
#     stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
#     with open(stats_path, 'wb') as f:
#         pickle.dump(stats, f)

#     # Pass the manual start step to the training function
#     train_bc(train_dataloader, val_dataloader, policy_config, 
#              resume_ckpt=ckpt_path_args, 
#              manual_start_step=start_step_args)

# ## Step-Based Training Script (Fixed Logging & Resume)
# from config.config_joint_1_step import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, USE_WANDB, DATA_LOADER_CONFIG

# import os
# import pickle
# import argparse
# from copy import deepcopy
# import matplotlib.pyplot as plt
# import wandb
# import torch
# import numpy as np
# import sys
# import time

# from training.utils_joint_1_step import *

# # parse args
# parser = argparse.ArgumentParser()
# parser.add_argument('--task', type=str, default='pick')
# parser.add_argument('--checkpoint', type=str, default=None, help='Path to .ckpt file to resume training from')
# # ADDED: Support for manual start step (crucial for resuming/continual training)
# parser.add_argument('--start_step', type=int, default=None, help='Force execution to start from this step (overrides checkpoint)')
# args = parser.parse_args()

# task = args.task
# ckpt_path_args = args.checkpoint
# start_step_args = args.start_step

# # configs
# task_cfg = TASK_CONFIG
# train_cfg = TRAIN_CONFIG
# policy_config = POLICY_CONFIG
# data_loader_config = DATA_LOADER_CONFIG
# checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task)
# device = os.environ['DEVICE']

# def forward_pass(data, policy):
#     image_data, qpos_data, action_data, is_pad = data
#     image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
#     return policy(qpos_data, image_data, action_data, is_pad)

# def train_bc(train_dataloader, val_dataloader, policy_config, resume_ckpt=None, manual_start_step=None):
    
#     # --- CONFIG SETUP ---
#     MAX_STEPS = train_cfg.get('max_steps', 100000)
#     EVAL_INTERVAL = train_cfg.get('eval_interval', 2000)
#     SAVE_INTERVAL = train_cfg.get('save_interval', 2000)
#     PRINT_INTERVAL = 1
    
#     print("\n" + "="*80)
#     print(f"STARTING STEP-BASED TRAINING")
#     print(f"Max Steps: {MAX_STEPS} | Eval Interval: {EVAL_INTERVAL} | Save Interval: {SAVE_INTERVAL}")
#     print("="*80 + "\n")
    
#     # 1. Initialize
#     policy = make_policy(policy_config['policy_class'], policy_config)
#     policy.to(device)
#     optimizer = make_optimizer(policy_config['policy_class'], policy)
    
#     current_step = 0

#     # 2. Resume Logic
#     if resume_ckpt is not None:
#         if os.path.isfile(resume_ckpt):
#             print(f"Loading checkpoint from: {resume_ckpt}")
#             checkpoint = torch.load(resume_ckpt, map_location=device)
            
#             # Load Weights
#             if 'model_state_dict' in checkpoint:
#                 policy.load_state_dict(checkpoint['model_state_dict'])
#                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
#                 # Checkpoint Step
#                 ckpt_step = checkpoint.get('step', 0)
#                 print(f"   Checkpoint reports step: {ckpt_step}")
#                 current_step = ckpt_step
#             else:
#                 # Weights only (Legacy)
#                 policy.load_state_dict(checkpoint)
#                 print("   Loaded weights-only checkpoint.")
                
#         else:
#             print(f"Warning: Checkpoint {resume_ckpt} not found. Starting from scratch.")

#     # Override step if provided manually (Continual Training support)
#     if manual_start_step is not None:
#         print(f"   Overriding start step to: {manual_start_step}")
#         current_step = manual_start_step

#     os.makedirs(checkpoint_dir, exist_ok=True)

#     # 3. Create Infinite Iterator
#     train_iterator = cycle(train_dataloader)
    
#     policy.train()
    
#     # Trackers for averaged logging
#     running_metrics = {}
#     interval_start_time = time.time()
#     steps_this_interval = 0
    
#     # 4. Main Step Loop
#     # Loop range handles the resumption automatically
#     for step in range(current_step, MAX_STEPS + 1):
        
#         # --- VALIDATION ---
#         if step % EVAL_INTERVAL == 0 and step != 0 and step != current_step:
#             print(f"\nStep {step}: Validating...")
#             policy.eval()
#             with torch.inference_mode():
#                 # We collect ALL metrics during validation now
#                 val_losses = [] # Just for printing average total loss
#                 val_metrics_sum = {} # To average all components (kl, l1, etc)
#                 num_val_batches = 50 
                
#                 for i, data in enumerate(val_dataloader):
#                     if i >= num_val_batches: break
#                     forward_dict = forward_pass(data, policy)
                    
#                     # Accumulate all keys (loss, l1, kl, etc.)
#                     for k, v in forward_dict.items():
#                         v_item = v.item()
#                         if k not in val_metrics_sum: val_metrics_sum[k] = 0.0
#                         val_metrics_sum[k] += v_item
                    
#                     val_losses.append(forward_dict['loss'].item())
                
#                 # Compute Averages
#                 avg_val_loss = np.mean(val_losses)
#                 val_metrics_avg = {k: v / len(val_losses) for k, v in val_metrics_sum.items()}
                
#                 # Print formatted validation metrics
#                 val_str = f"   Val Loss: {avg_val_loss:.5f} | "
#                 val_str += " ".join([f"{k}: {v:.5f}" for k, v in val_metrics_avg.items() if k != 'loss'])
#                 print(val_str)
                
#                 if USE_WANDB:
#                     wandb.log({f'val/{k}': v for k, v in val_metrics_avg.items()})
#                     wandb.log({'step': step})

#             policy.train()

#         # --- TRAINING ---
#         data = next(train_iterator)
        
#         forward_dict = forward_pass(data, policy)
#         loss = forward_dict['loss']
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # Accumulate metrics for averaging
#         steps_this_interval += 1
#         for k, v in forward_dict.items():
#             if k not in running_metrics:
#                 running_metrics[k] = 0.0
#             running_metrics[k] += v.item()

#         # --- LOGGING (Averaged Detailed Breakdown) ---
#         if step % PRINT_INTERVAL == 0 and steps_this_interval > 0:
#             t1 = time.time()
#             time_taken = t1 - interval_start_time
            
#             # Create a string for all loss components (kl, l1)
#             metrics_str = ""
#             wandb_dict = {'step': step}
            
#             for k, sum_val in running_metrics.items():
#                 # Safely compute the average using the actual number of steps executed
#                 avg_val = sum_val / steps_this_interval
#                 metrics_str += f"{k}: {avg_val:.4f} "
#                 wandb_dict[f'train/{k}'] = avg_val

#             print(f"Step {step} | Train Loss: {metrics_str}| Time for {steps_this_interval} steps: {time_taken:.3f}s")
            
#             if USE_WANDB:
#                 wandb.log(wandb_dict)

#             # Reset trackers for the next interval
#             running_metrics = {}
#             interval_start_time = time.time()
#             steps_this_interval = 0

#         # --- CHECKPOINTING ---
#         if step > 0 and step % SAVE_INTERVAL == 0 and step != current_step:
#             ckpt_path = os.path.join(checkpoint_dir, f'policy_step_{step}.ckpt')
#             save_payload = {
#                 'model_state_dict': policy.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'step': step,
#                 'train_loss': loss.item(),
#             }
#             torch.save(save_payload, ckpt_path)
#             print(f"   Saved checkpoint: {os.path.basename(ckpt_path)}")
            
#     # Final Save
#     ckpt_path = os.path.join(checkpoint_dir, f'policy_last.ckpt')
#     torch.save({
#         'model_state_dict': policy.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'step': MAX_STEPS,
#     }, ckpt_path)
#     print("Training Finished.")


# if __name__ == '__main__':
    
#     print("\n" + "="*80)
#     print("INITIALIZATION")
#     print("="*80 + "\n")
    
#     if USE_WANDB:
#         wandb_name = f"{task}_{train_cfg['seed']}_{train_cfg['wandb_run_name']}"
#         if ckpt_path_args:
#             wandb_name += "_resumed"

#         wandb.init(
#             project="ACT_origin",
#             name=wandb_name,
#             config={
#                 "task": task,
#                 **{f"task_cfg/{k}": v for k, v in TASK_CONFIG.items()},
#                 **{f"train_cfg/{k}": v for k, v in TRAIN_CONFIG.items()},
#                 **{f"policy_cfg/{k}": v for k, v in POLICY_CONFIG.items()},
#             }
#         )
    
#     set_seed(train_cfg['seed'])
#     os.makedirs(checkpoint_dir, exist_ok=True)
    
#     data_dir = task_cfg['dataset_dir']
#     num_queries = policy_config.get('num_queries', 8)
    
#     print("\n" + "="*80)
#     print("LOADING DATA (Calculating 100% Coverage indices...)")
#     print("="*80)
    
#     train_dataloader, val_dataloader, stats, _ = load_data(
#         data_dir, 
#         batch_size_train=train_cfg['batch_size_train'], 
#         batch_size_val=train_cfg['batch_size_val'],
#         num_queries=num_queries,
#         data_loader_config=data_loader_config,
#         camera_names=policy_config['camera_names'],
#         train_ratio = train_cfg['train_ratio'],
#         state_dim=TASK_CONFIG['state_dim']
#     )
    
#     stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
#     with open(stats_path, 'wb') as f:
#         pickle.dump(stats, f)

#     # Pass the manual start step to the training function
#     train_bc(train_dataloader, val_dataloader, policy_config, 
#              resume_ckpt=ckpt_path_args, 
#              manual_start_step=start_step_args)


## step based, save best checkpoint
from config.config_joint_1_step import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, USE_WANDB, DATA_LOADER_CONFIG

import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import wandb
import torch
import numpy as np
import sys
import time

from training.utils_joint_1_step import *

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='pick')
parser.add_argument('--checkpoint', type=str, default='/home/chavvive/ws_ros2humble-main_lab/ACT/checkpoints/joint/20260220_pick_and_place_realsense_aria_100/pick/policy_step_1115.ckpt', help='Path to .ckpt file to resume training from')
# ADDED: Support for manual start step (crucial for resuming/continual training)
parser.add_argument('--start_step', type=int, default=None, help='Force execution to start from this step (overrides checkpoint)')
args = parser.parse_args()

task = args.task
ckpt_path_args = args.checkpoint
start_step_args = args.start_step

# configs
task_cfg = TASK_CONFIG
train_cfg = TRAIN_CONFIG
policy_config = POLICY_CONFIG
data_loader_config = DATA_LOADER_CONFIG
checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task)
device = os.environ['DEVICE']

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)

def train_bc(train_dataloader, val_dataloader, policy_config, resume_ckpt=None, manual_start_step=None):
    
    # --- CONFIG SETUP ---
    MAX_STEPS = train_cfg.get('max_steps', 100000)
    EVAL_INTERVAL = train_cfg.get('eval_interval', 2000)
    SAVE_INTERVAL = train_cfg.get('save_interval', 2000)
    VALID_NUMS = train_cfg.get('valid_nums', 7)
    PRINT_INTERVAL = 1
    
    print("\n" + "="*80)
    print(f"STARTING STEP-BASED TRAINING")
    print(f"Max Steps: {MAX_STEPS} | Eval Interval: {EVAL_INTERVAL} | Save Interval: {SAVE_INTERVAL}")
    print("="*80 + "\n")
    
    # 1. Initialize
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to(device)
    optimizer = make_optimizer(policy_config['policy_class'], policy)
    
    current_step = 0
    best_val_loss = float('inf') # Initialize best loss tracking

    # 2. Resume Logic
    if resume_ckpt is not None:
        if os.path.isfile(resume_ckpt):
            print(f"Loading checkpoint from: {resume_ckpt}")
            checkpoint = torch.load(resume_ckpt, map_location=device)
            
            # Load Weights
            if 'model_state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Checkpoint Step
                ckpt_step = checkpoint.get('step', 0)
                print(f"   Checkpoint reports step: {ckpt_step}")
                current_step = ckpt_step
            else:
                # Weights only (Legacy)
                policy.load_state_dict(checkpoint)
                print("   Loaded weights-only checkpoint.")
                
        else:
            print(f"Warning: Checkpoint {resume_ckpt} not found. Starting from scratch.")

    # Override step if provided manually (Continual Training support)
    if manual_start_step is not None:
        print(f"   Overriding start step to: {manual_start_step}")
        current_step = manual_start_step

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 3. Create Infinite Iterator
    train_iterator = cycle(train_dataloader)
    
    policy.train()
    
    # Trackers for averaged logging
    running_metrics = {}
    interval_start_time = time.time()
    steps_this_interval = 0
    
    # 4. Main Step Loop
    # Loop range handles the resumption automatically
    for step in range(current_step, MAX_STEPS + 1):
        
        # --- VALIDATION ---
        if step % EVAL_INTERVAL == 0 and step != 0 and step != current_step:
            print(f"\nStep {step}: Validating...")
            policy.eval()
            with torch.inference_mode():
                # We collect ALL metrics during validation now
                val_losses = [] # Just for printing average total loss
                val_metrics_sum = {} # To average all components (kl, l1, etc)
                num_val_batches = VALID_NUMS 
                
                for i, data in enumerate(val_dataloader):
                    if i >= num_val_batches: break
                    forward_dict = forward_pass(data, policy)
                    
                    # Accumulate all keys (loss, l1, kl, etc.)
                    for k, v in forward_dict.items():
                        v_item = v.item()
                        if k not in val_metrics_sum: val_metrics_sum[k] = 0.0
                        val_metrics_sum[k] += v_item
                    
                    val_losses.append(forward_dict['loss'].item())
                
                # Compute Averages
                avg_val_loss = np.mean(val_losses)
                val_metrics_avg = {k: v / len(val_losses) for k, v in val_metrics_sum.items()}
                
                # --- SAVE BEST CHECKPOINT LOGIC ---
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_ckpt_path = os.path.join(checkpoint_dir, 'policy_best.ckpt')
                    torch.save({
                        'model_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': step,
                        'val_loss': avg_val_loss,
                    }, best_ckpt_path)
                    print(f"   New Best Val Loss: {best_val_loss:.5f} | Saved: policy_best.ckpt")
                # ----------------------------------

                # Print formatted validation metrics
                val_str = f"   Val Loss: {avg_val_loss:.5f} | "
                val_str += " ".join([f"{k}: {v:.5f}" for k, v in val_metrics_avg.items() if k != 'loss'])
                print(val_str)
                
                if USE_WANDB:
                    wandb.log({f'val/{k}': v for k, v in val_metrics_avg.items()})
                    wandb.log({'step': step})

            policy.train()

        # --- TRAINING ---
        data = next(train_iterator)
        
        forward_dict = forward_pass(data, policy)
        loss = forward_dict['loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics for averaging
        steps_this_interval += 1
        for k, v in forward_dict.items():
            if k not in running_metrics:
                running_metrics[k] = 0.0
            running_metrics[k] += v.item()

        # --- LOGGING (Averaged Detailed Breakdown) ---
        if step % PRINT_INTERVAL == 0 and steps_this_interval > 0:
            t1 = time.time()
            time_taken = t1 - interval_start_time
            
            # Create a string for all loss components (kl, l1)
            metrics_str = ""
            wandb_dict = {'step': step}
            
            for k, sum_val in running_metrics.items():
                # Safely compute the average using the actual number of steps executed
                avg_val = sum_val / steps_this_interval
                metrics_str += f"{k}: {avg_val:.4f} "
                wandb_dict[f'train/{k}'] = avg_val

            print(f"Step {step} | Train Loss: {metrics_str}| Time for {steps_this_interval} steps: {time_taken:.3f}s")
            
            if USE_WANDB:
                wandb.log(wandb_dict)

            # Reset trackers for the next interval
            running_metrics = {}
            interval_start_time = time.time()
            steps_this_interval = 0

        # --- CHECKPOINTING ---
        if step > 0 and step % SAVE_INTERVAL == 0 and step != current_step:
            ckpt_path = os.path.join(checkpoint_dir, f'policy_step_{step}.ckpt')
            save_payload = {
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'train_loss': loss.item(),
            }
            torch.save(save_payload, ckpt_path)
            print(f"   Saved checkpoint: {os.path.basename(ckpt_path)}")
            
    # Final Save
    ckpt_path = os.path.join(checkpoint_dir, f'policy_last.ckpt')
    torch.save({
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': MAX_STEPS,
    }, ckpt_path)
    print("Training Finished.")


if __name__ == '__main__':
    
    print("\n" + "="*80)
    print("INITIALIZATION")
    print("="*80 + "\n")
    
    if USE_WANDB:
        wandb_name = f"{task}_{train_cfg['seed']}_{train_cfg['wandb_run_name']}"
        if ckpt_path_args:
            wandb_name += "_resumed"

        wandb.init(
            project="ACT_origin",
            name=wandb_name,
            config={
                "task": task,
                **{f"task_cfg/{k}": v for k, v in TASK_CONFIG.items()},
                **{f"train_cfg/{k}": v for k, v in TRAIN_CONFIG.items()},
                **{f"policy_cfg/{k}": v for k, v in POLICY_CONFIG.items()},
            }
        )
    
    set_seed(train_cfg['seed'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    data_dir = task_cfg['dataset_dir']
    num_queries = policy_config.get('num_queries', 8)
    
    print("\n" + "="*80)
    print("LOADING DATA (Calculating 100% Coverage indices...)")
    print("="*80)
    
    train_dataloader, val_dataloader, stats, _ = load_data(
        data_dir, 
        batch_size_train=train_cfg['batch_size_train'], 
        batch_size_val=train_cfg['batch_size_val'],
        num_queries=num_queries,
        data_loader_config=data_loader_config,
        camera_names=policy_config['camera_names'],
        train_ratio = train_cfg['train_ratio'],
        state_dim=TASK_CONFIG['state_dim']
    )
    
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # Pass the manual start step to the training function
    train_bc(train_dataloader, val_dataloader, policy_config, 
             resume_ckpt=ckpt_path_args, 
             manual_start_step=start_step_args)