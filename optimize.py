
# hyperparameter_optuna.py

import os
import pickle
import argparse
from copy import deepcopy
import numpy as np
import torch
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import logging

from config.config_1 import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, USE_WANDB
from training.utils_1 import (
    load_data, make_policy, make_optimizer, 
    compute_dict_mean, set_seed, detach_dict
)

# Set up logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command line arguments

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='pick', help='Task name')
parser.add_argument('--n_trials', type=int, default=100, help='Number of trials')
parser.add_argument('--n_epochs_per_trial', type=int, default=50, help='Epochs per trial')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--db_path', type=str, default='./optuna_study.db', help='Path to save Optuna database')
args = parser.parse_args()

# Configuration

DEVICE = args.device
TASK = args.task
N_TRIALS = args.n_trials
N_EPOCHS_PER_TRIAL = args.n_epochs_per_trial
DB_PATH = args.db_path

os.environ['DEVICE'] = DEVICE


def forward_pass(data, policy):
    """Forward pass through the model."""
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.to(DEVICE)
    qpos_data = qpos_data.to(DEVICE)
    action_data = action_data.to(DEVICE)
    is_pad = is_pad.to(DEVICE)
    return policy(qpos_data, image_data, action_data, is_pad)


def validate_policy(policy, val_dataloader):
    """Run validation and return average loss."""
    policy.eval()
    validation_losses = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_dataloader):
            forward_dict = forward_pass(data, policy)
            validation_losses.append(forward_dict['loss'].item())
    
    avg_val_loss = np.mean(validation_losses)
    return avg_val_loss


def train_one_epoch(policy, train_dataloader, optimizer):
    """Train for one epoch and return average loss."""
    policy.train()
    epoch_losses = []
    
    for batch_idx, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        forward_dict = forward_pass(data, policy)
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    
    avg_train_loss = np.mean(epoch_losses)
    return avg_train_loss


def objective(trial: Trial):
    """Objective function for Optuna optimization."""
    
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    kl_weight = trial.suggest_float('kl_weight', 0.1, 100, log=True)
    trans_weight = trial.suggest_float('trans_weight', 0.01, 10, log=True)
    rot_weight = trial.suggest_float('rot_weight', 0.01, 10, log=True)
    dist_weight = trial.suggest_float('dist_weight', 0.01, 10, log=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Trial {trial.number}: Testing hyperparameters")
    logger.info(f"  lr={lr:.2e}, kl_weight={kl_weight:.4f}")
    logger.info(f"  trans_weight={trans_weight:.4f}, rot_weight={rot_weight:.4f}")
    logger.info(f"  dist_weight={dist_weight:.4f}")
    logger.info(f"{'='*60}\n")
    
    # Set seed for reproducibility
    set_seed(TRAIN_CONFIG['seed'])
    
    # Create policy config with suggested hyperparameters
    policy_config = deepcopy(POLICY_CONFIG)
    policy_config['lr'] = lr
    policy_config['kl_weight'] = kl_weight
    policy_config['trans_weight'] = trans_weight
    policy_config['rot_weight'] = rot_weight
    policy_config['dist_weight'] = dist_weight
    policy_config['device'] = DEVICE
    
    # Create policy
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to(DEVICE)
    
    # Create optimizer
    optimizer = make_optimizer(policy_config['policy_class'], policy)
    
    # Load data
    data_dir = TASK_CONFIG['dataset_dir']
    num_queries = policy_config.get('num_queries', 8)
    
    try:
        train_dataloader, val_dataloader, stats, _ = load_data(
            data_dir,
            batch_size_train=TRAIN_CONFIG['batch_size_train'],
            batch_size_val=TRAIN_CONFIG['batch_size_val'],
            num_queries=num_queries
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return float('inf')
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(N_EPOCHS_PER_TRIAL):
        # Train
        train_loss = train_one_epoch(policy, train_dataloader, optimizer)
        
        # Validate
        val_loss = validate_policy(policy, val_dataloader)
        
        logger.info(f"Epoch {epoch+1}/{N_EPOCHS_PER_TRIAL} | "
                   f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        # Track best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Report intermediate value for pruning
        trial.report(val_loss, epoch)
        
        # Check for pruning
        if trial.should_prune():
            logger.info(f"Trial pruned at epoch {epoch+1}")
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (patience exceeded)")
            break
    
    logger.info(f"Trial {trial.number} - Best Val Loss: {best_val_loss:.5f}\n")
    
    return best_val_loss


def run_optimization():
    """Run the Optuna hyperparameter optimization."""
    
    # Create study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        storage=f'sqlite:///{DB_PATH}',
        study_name='act_hyperparameter_optimization'
    )
    
    # Run optimization
    logger.info(f"Starting hyperparameter optimization with {N_TRIALS} trials")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Epochs per trial: {N_EPOCHS_PER_TRIAL}\n")
    
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Print results
    logger.info(f"\n{'='*60}")
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"{'='*60}\n")
    
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best validation loss: {study.best_value:.5f}\n")
    
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Save best params
    best_params = study.best_params
    best_params_path = f'best_hyperparams_{TASK}.pkl'
    with open(best_params_path, 'wb') as f:
        pickle.dump(best_params, f)
    logger.info(f"\nBest hyperparameters saved to: {best_params_path}")
    
    # Save study visualization
    try:
        import plotly
        fig = optuna.visualization.plot_optimization_history(study).show()
        param_importance_fig = optuna.visualization.plot_param_importances(study).show()
        logger.info("Visualization plots generated")
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")
    
    return study


if __name__ == '__main__':
    study = run_optimization()