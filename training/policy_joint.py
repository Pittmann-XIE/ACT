import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

import sys
sys.path.append('/home/ipk311l-user/ws_ros2humble-main_lab/ACT/act_origin')

from act_origin.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed
import torch

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.dist_weight = args_override['dist_weight']

        self.initial_loss = None 
        print(f'KL Weight {self.kl_weight}, dist_weight {self.dist_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            noise_std = 0.02 
            qpos_noise = torch.randn_like(qpos[:, :6]) * noise_std
            corrupted_qpos = qpos.clone()
            corrupted_qpos[:, :6] += qpos_noise
            qpos = corrupted_qpos
            
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            
            # joint
            a_hat_joint = a_hat[:, :, :6] 
            actions_joint = actions[:, :, :6]
            all_l1 = F.l1_loss(actions_joint, a_hat_joint, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            # gripper
            actions_gripper = actions[:, :, 6:]   
            a_hat_gripper = a_hat[:, :, 6:]
            bce_loss = F.binary_cross_entropy(
                a_hat_gripper.squeeze(-1),
                actions_gripper.squeeze(-1),
                reduction='none'
            )
            l_bce_distance = (bce_loss * ~is_pad).mean()
           
            # total loss
            loss_dict = dict()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['bce_distance'] = l_bce_distance
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + loss_dict['bce_distance']*self.dist_weight
            
            # Calculate ratio: Current Loss / Initial Loss
            if self.initial_loss is None:
                self.initial_loss = loss_dict['loss'].item()
           
            # Guard against division by zero if initial loss happens to be 0
            if self.initial_loss != 0:
                loss_dict['loss_ratio'] = loss_dict['loss'] / self.initial_loss
            else:
                loss_dict['loss_ratio'] = torch.tensor(1.0, device=loss_dict['loss'].device)

            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

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