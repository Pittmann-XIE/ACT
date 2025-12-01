from config.config_1 import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, USE_WANDB # must import first

import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import wandb

from training.utils_1 import *

# parse the task name via command line

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='pick')
args = parser.parse_args()
task = args.task

# configs

task_cfg = TASK_CONFIG
train_cfg = TRAIN_CONFIG
policy_config = POLICY_CONFIG
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
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def train_bc(train_dataloader, val_dataloader, policy_config):
    # load policy
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to(device)

    # load optimizer
    optimizer = make_optimizer(policy_config['policy_class'], policy)

    # create checkpoint dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    
    for epoch in range(train_cfg['num_epochs']):
        print(f'\nEpoch {epoch}')
        
        # validation
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
        
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        val_log_dict = {'epoch': epoch}
        for k, v in epoch_summary.items():
            v_scalar = v.item() if hasattr(v, 'item') else v
            summary_string += f'{k}: {v_scalar:.3f} '
            val_log_dict[f'val/{k}'] = v_scalar
        print(summary_string)
        
        # Log validation metrics to wandb
        if USE_WANDB:
            wandb.log(val_log_dict)

        # training
        policy.train()
        optimizer.zero_grad()
        num_batches = 0
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            num_batches = batch_idx + 1
        
        # Compute training summary for this epoch
        epoch_start_idx = len(train_history) - num_batches
        epoch_summary = compute_dict_mean(train_history[epoch_start_idx:])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        train_log_dict = {'epoch': epoch}
        for k, v in epoch_summary.items():
            v_scalar = v.item() if hasattr(v, 'item') else v
            summary_string += f'{k}: {v_scalar:.3f} '
            train_log_dict[f'train/{k}'] = v_scalar
        print(summary_string)
        
        # Log training metrics to wandb
        if USE_WANDB:
            wandb.log(train_log_dict)

        # save checkpoint periodically
        if epoch % train_cfg.get('ckpt_interval', 200) == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"policy_epoch_{epoch}_seed_{train_cfg['seed']}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            # plot_history(train_history, validation_history, epoch, checkpoint_dir, train_cfg['seed'])

    # save final checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)
    
    # save best checkpoint
    if best_ckpt_info is not None:
        best_epoch, best_loss, best_state = best_ckpt_info
        ckpt_path = os.path.join(checkpoint_dir, f'policy_best.ckpt')
        torch.save(best_state, ckpt_path)
        print(f'\nBest checkpoint: Epoch {best_epoch}, Loss {best_loss:.5f}')
        if USE_WANDB:
            wandb.log({"best_epoch": best_epoch, "best_val_loss": best_loss.item() if hasattr(best_loss, 'item') else best_loss})
    if USE_WANDB:
        wandb.finish()
    

if __name__ == '__main__':
    # Initialize wandb with all configs
    if USE_WANDB:
        wandb.init(
            project="ACT_origin",
            entity=None,  # Change to your wandb team name if needed
            name=f"{task}_20251201_01",
            config={
                "task": task,
                # Task config
                **{f"task_cfg/{k}": v for k, v in TASK_CONFIG.items()},
                # Training config
                **{f"train_cfg/{k}": v for k, v in TRAIN_CONFIG.items()},
                # Policy config
                **{f"policy_cfg/{k}": v for k, v in POLICY_CONFIG.items()},
                'remarks': 'testing'
            }
        )
    
    # set seed
    set_seed(train_cfg['seed'])
    
    # create ckpt dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # dataset directory (contains .h5 files)
    data_dir = task_cfg['dataset_dir']
    
    # Get num_queries from policy_config (for sequence length)
    num_queries = policy_config.get('num_queries', 8)
    print(f"Using num_queries (action sequence length): {num_queries}")
    
    # load data
    train_dataloader, val_dataloader, stats, _ = load_data(
        data_dir, 
        batch_size_train=train_cfg['batch_size_train'], 
        batch_size_val=train_cfg['batch_size_val'],
        num_queries=num_queries
    )
    
    # save stats
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # train
    train_bc(train_dataloader, val_dataloader, policy_config)