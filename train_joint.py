## continue training


## Caps number of samples per epoch
from config.config_joint import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, USE_WANDB, DATA_LOADER_CONFIG

import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import wandb
import torch
import numpy as np
import sys

from training.utils_joint import *

# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='pick')
# --- ADDED ARGS FOR RESUMING ---
parser.add_argument('--checkpoint', type=str, default=None, help='Path to .ckpt file to resume training from')
parser.add_argument('--start_epoch', type=int, default=None, help='Epoch to start/resume from. If checkpoint has epoch info, this is ignored.')
# -------------------------------
args = parser.parse_args()
task = args.task
ckpt_path_args = args.checkpoint
start_epoch_args = args.start_epoch

# configs
task_cfg = TASK_CONFIG
train_cfg = TRAIN_CONFIG
policy_config = POLICY_CONFIG
data_loader_config = DATA_LOADER_CONFIG
checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task)

# device
device = os.environ['DEVICE']


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        # Adjust x-axis to match actual epochs if history length differs from num_epochs
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def train_bc(train_dataloader, val_dataloader, policy_config, resume_ckpt=None, start_epoch=0):
    print("\n" + "="*80)
    print("STARTING TRAINING WITH TIMING DIAGNOSTICS")
    print("="*80 + "\n")
    
    # load policy
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to(device)

    # load optimizer
    optimizer = make_optimizer(policy_config['policy_class'], policy)

    # ============ RESUME LOGIC ============
    if resume_ckpt is not None:
        if os.path.isfile(resume_ckpt):
            print(f"Loading checkpoint from: {resume_ckpt}")
            
            # --- FIX APPLIED HERE ---
            checkpoint = torch.load(resume_ckpt, map_location=device, weights_only=False)
            # ------------------------
            
            # Check if checkpoint is new format (dict with 'model_state_dict') or old format (just state_dict)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("   Detected Full State Checkpoint (Model + Optimizer)")
                policy.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Auto-update start epoch if stored in checkpoint
                if 'epoch' in checkpoint:
                    saved_epoch = checkpoint['epoch']
                    start_epoch = saved_epoch + 1
                    print(f"   Resuming from epoch {start_epoch} (Saved epoch was {saved_epoch})")
            else:
                print("   Detected Weights-Only Checkpoint")
                policy.load_state_dict(checkpoint)
                print(f"   Using command line start_epoch: {start_epoch}")
                
            print("Checkpoint loaded successfully.")
        else:
            print(f"Warning: Checkpoint file {resume_ckpt} not found. Starting from scratch.")
    # ======================================

    # create checkpoint dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    
    # Loop from start_epoch to num_epochs
    if resume_ckpt is not None: 
        for epoch in range(start_epoch, train_cfg['num_epochs']):
            print("\n" + "="*80)
            print(f'EPOCH {epoch}')
            print("="*80)
            
            # ============ VALIDATION ============
            print(f"\n🔍 Starting validation...")
            
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    epoch_dicts.append(forward_dict)
                
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    # We only save model weights for "Best" to save memory/space, 
                    # as we usually don't resume training from "Best", we just eval it.
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
            
            print(f'\n📊 Val loss: {epoch_val_loss:.5f}')
            summary_string = ''
            val_log_dict = {'epoch': epoch}
            for k, v in epoch_summary.items():
                v_scalar = v.item() if hasattr(v, 'item') else v
                summary_string += f'{k}: {v_scalar:.3f} '
                val_log_dict[f'val/{k}'] = v_scalar
            print(f'   {summary_string}')
            
            # Log validation metrics to wandb
            if USE_WANDB:
                wandb.log(val_log_dict)

            # ============ TRAINING ============
            print(f"\n🚀 Starting training...")
            
            policy.train()
            epoch_train_dicts = []
            
            for batch_idx, data in enumerate(train_dataloader):
                forward_dict = forward_pass(data, policy)
                loss = forward_dict['loss']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_train_dicts.append(detach_dict(forward_dict))

            
            # Compute training summary
            epoch_summary = compute_dict_mean(epoch_train_dicts)
            train_history.append(epoch_summary)
            
            epoch_train_loss = epoch_summary['loss']
            
            print(f'\n📊 Train loss: {epoch_train_loss:.5f}')
            summary_string = ''
            train_log_dict = {'epoch': epoch}
            for k, v in epoch_summary.items():
                v_scalar = v.item() if hasattr(v, 'item') else v
                summary_string += f'{k}: {v_scalar:.3f} '
                train_log_dict[f'train/{k}'] = v_scalar
            print(f'   {summary_string}')
            
            # Log training metrics to wandb
            if USE_WANDB:
                wandb.log(train_log_dict)

            # ============ CHECKPOINTING ============
            if epoch % train_cfg.get('ckpt_interval', 100) == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"policy_epoch_{epoch}_seed_{train_cfg['seed']}.ckpt")
                
                # Save FULL state (Model + Optimizer) for interval checkpoints
                save_payload = {
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss
                }
                torch.save(save_payload, ckpt_path)
            
            # Flush stdout to see output immediately
            sys.stdout.flush()
    else:
        for epoch in range(train_cfg['num_epochs']):
            print("\n" + "="*80)
            print(f'EPOCH {epoch}')
            print("="*80)
            
            # ============ VALIDATION ============
            print(f"\n🔍 Starting validation...")
            
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                
                for batch_idx, data in enumerate(val_dataloader):
                    
                    forward_dict = forward_pass(data, policy)
                    epoch_dicts.append(forward_dict)
                
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
            
            print(f'\n📊 Val loss: {epoch_val_loss:.5f}')
            summary_string = ''
            val_log_dict = {'epoch': epoch}
            for k, v in epoch_summary.items():
                v_scalar = v.item() if hasattr(v, 'item') else v
                summary_string += f'{k}: {v_scalar:.3f} '
                val_log_dict[f'val/{k}'] = v_scalar
            print(f'   {summary_string}')
            
            # Log validation metrics to wandb
            if USE_WANDB:
                wandb.log(val_log_dict)

            # ============ TRAINING ============
            print(f"\n🚀 Starting training...")
            
            policy.train()
            epoch_train_dicts = []
            
            for batch_idx, data in enumerate(train_dataloader):

                forward_dict = forward_pass(data, policy)
                loss = forward_dict['loss']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_train_dicts.append(detach_dict(forward_dict))

            
            # Compute training summary
            epoch_summary = compute_dict_mean(epoch_train_dicts)
            train_history.append(epoch_summary)
            
            epoch_train_loss = epoch_summary['loss']
            
            print(f'\n📊 Train loss: {epoch_train_loss:.5f}')
            summary_string = ''
            train_log_dict = {'epoch': epoch}
            for k, v in epoch_summary.items():
                v_scalar = v.item() if hasattr(v, 'item') else v
                summary_string += f'{k}: {v_scalar:.3f} '
                train_log_dict[f'train/{k}'] = v_scalar
            print(f'   {summary_string}')
            
            # Log training metrics to wandb
            if USE_WANDB:
                wandb.log(train_log_dict)

            # ============ CHECKPOINTING ============
            if epoch % train_cfg.get('ckpt_interval', 100) == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"policy_epoch_{epoch}_seed_{train_cfg['seed']}.ckpt")
                save_payload = {
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss
                }
                torch.save(save_payload, ckpt_path)
            
            # Flush stdout to see output immediately
            sys.stdout.flush()

    # ============ FINAL SAVE ============
    print("\n" + "="*80)
    print("SAVING FINAL CHECKPOINTS")
    print("="*80)
    
    # Save final checkpoint (Full State)
    ckpt_path = os.path.join(checkpoint_dir, f'policy_last.ckpt')
    save_payload = {
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': train_cfg['num_epochs'] - 1,
    }
    torch.save(save_payload, ckpt_path)
    
    # Save best checkpoint (Weights Only, kept simple for inference compatibility)
    if best_ckpt_info is not None:
        best_epoch, best_loss, best_state = best_ckpt_info
        ckpt_path = os.path.join(checkpoint_dir, f'policy_best.ckpt')
        
        # We save this in the old format (just weights) or wrap it. 
        # Standard practice: just weights for 'best' inference model.
        torch.save(best_state, ckpt_path)
        
        print(f'   Best epoch: {best_epoch}, Loss: {best_loss:.5f}')
        if USE_WANDB:
            wandb.log({"best_epoch": best_epoch, "best_val_loss": best_loss.item() if hasattr(best_loss, 'item') else best_loss})
    
    if USE_WANDB:
        wandb.finish()
    

if __name__ == '__main__':
    
    print("\n" + "="*80)
    print("INITIALIZATION")
    print("="*80 + "\n")
    
    # Initialize wandb
    if USE_WANDB:
        # If resuming, you might want to adjust the name to indicate resumption
        wandb_name = f"{task}_optimized_{train_cfg['seed']}_{train_cfg['wandb_name']}"
        if ckpt_path_args:
            wandb_name += "_resumed"

        wandb.init(
            project="ACT_origin",
            entity=None,
            name=wandb_name,
            config={
                "task": task,
                "resume_checkpoint": ckpt_path_args,
                "start_epoch": start_epoch_args,
                **{f"task_cfg/{k}": v for k, v in TASK_CONFIG.items()},
                **{f"train_cfg/{k}": v for k, v in TRAIN_CONFIG.items()},
                **{f"policy_cfg/{k}": v for k, v in POLICY_CONFIG.items()},
                **{f"data_loader_cfg/{k}": v for k, v in DATA_LOADER_CONFIG.items()},
                'remarks': 'Optimized with timing diagnostics + Resume Support',
            }
        )
    
    # set seed
    set_seed(train_cfg['seed'])
    
    # create ckpt dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # dataset directory
    data_dir = task_cfg['dataset_dir']
    
    # Get num_queries
    num_queries = policy_config.get('num_queries', 8)
    print(f"\nUsing num_queries (action sequence length): {num_queries}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80 + "\n")
    
    train_dataloader, val_dataloader, stats, _ = load_data(
        data_dir, 
        batch_size_train=train_cfg['batch_size_train'], 
        batch_size_val=train_cfg['batch_size_val'],
        num_queries=num_queries,
        data_loader_config=data_loader_config,
        samples_per_epoch=8,
        camera_names=policy_config['camera_names']
    )
    
    # save stats
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # Train
    train_bc(train_dataloader, val_dataloader, policy_config, 
             resume_ckpt=ckpt_path_args, 
             start_epoch=start_epoch_args)
    
    print("\n" + "="*80)
    print("TIMING SUMMARY")
    print("="*80)
    print("="*80 + "\n")