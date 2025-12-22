import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import math

import sys
sys.path.append('/home/ipk311l-user/ws_ros2humble-main_lab/ACT/act_origin')

from act_origin.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

def geodesic_loss(pred_6d, target_6d, reduction='mean'):
    """
    Geodesic loss for 6D rotation representation.
    Computes angular distance between predicted and target rotations.
    
    Args:
        pred_6d: (batch, seq, 6) or (batch, 6)
        target_6d: (batch, seq, 6) or (batch, 6)
    Returns:
        loss: scalar
    """
    # Normalize to unit vectors
    pred_normalized = F.normalize(pred_6d, p=2, dim=-1)
    target_normalized = F.normalize(target_6d, p=2, dim=-1)
    
    # Compute cosine similarity (dot product of normalized vectors)
    cos_sim = (pred_normalized * target_normalized).sum(dim=-1)
    
    # Clamp to avoid numerical issues with arccos
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    
    # Compute angular distance (geodesic distance on SO(3))
    angular_distance = torch.acos(cos_sim)
    
    if reduction == 'mean':
        return angular_distance.mean()
    elif reduction == 'none':
        return angular_distance
    else:
        return angular_distance.sum()


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.trans_weight = args_override['trans_weight']
        self.rot_weight = args_override['rot_weight']
        self.dist_weight = args_override['dist_weight']
        
        # --- CORRECTION START ---
        # Initialize a placeholder for the initial loss
        self.initial_loss = None 
        # --- CORRECTION END ---
        
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            
            # Split predictions (distance is now already sigmoid'd)
            a_hat_xyz = a_hat[:, :, :3]
            a_hat_rotation = a_hat[:, :, 3:9]
            a_hat_distance = a_hat[:, :, 9:]  # Already probabilities
            
            # Split targets
            actions_xyz = actions[:, :, :3]
            actions_rotation = actions[:, :, 3:9]
            actions_distance = actions[:, :, 9:]  # Binary [0, 1]
            
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            
            # L1 loss for xyz
            all_l1_xyz = F.l1_loss(a_hat_xyz, actions_xyz, reduction='none')
            l1_xyz = (all_l1_xyz * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1_xyz'] = l1_xyz
            
            # Geodesic loss for rotation
            all_geodesic = geodesic_loss(a_hat_rotation, actions_rotation, reduction='none')
            l_geodesic = (all_geodesic * ~is_pad).mean()
            loss_dict['geodesic_rotation'] = l_geodesic
            
            # BCE loss for distance (now using probabilities, not logits)
            bce_loss = F.binary_cross_entropy(
                a_hat_distance.squeeze(-1),
                actions_distance.squeeze(-1),
                reduction='none'
            )
            l_bce_distance = (bce_loss * ~is_pad).mean()
            loss_dict['bce_distance'] = l_bce_distance
            
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = (loss_dict['l1_xyz'] * self.trans_weight+ 
                            loss_dict['geodesic_rotation'] * self.rot_weight+ 
                            loss_dict['bce_distance'] * self.dist_weight+ 
                            loss_dict['kl'] * self.kl_weight)
            
            # --- CORRECTION START ---
            # If this is the first step, record the loss as the initial loss.
            # We use .item() to store the scalar value without keeping the gradient graph.
            if self.initial_loss is None:
                self.initial_loss = loss_dict['loss'].item()
            
            # Calculate ratio: Current Loss / Initial Loss
            # Guard against division by zero if initial loss happens to be 0
            if self.initial_loss != 0:
                loss_dict['loss_ratio'] = loss_dict['loss'] / self.initial_loss
            else:
                loss_dict['loss_ratio'] = torch.tensor(1.0, device=loss_dict['loss'].device)
            # --- CORRECTION END ---
            
            return loss_dict
            
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)
            return a_hat  # Distance component already has sigmoid applied
    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer