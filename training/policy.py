import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

import sys
sys.path.append('/home/ipk311l-user/ws_ros2humble-main_lab/ACT/act_origin')

from act_origin.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
            env_state = None
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            image = normalize(image)
            if actions is not None: # training time
                actions = actions[:, :self.model.num_queries]
                is_pad = is_pad[:, :self.model.num_queries]

                a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
                
                # Split predictions into action and distance components
                a_hat_action = a_hat[:, :, :self.action_dim]        # (batch, seq, action_dim)
                a_hat_distance = a_hat[:, :, self.action_dim:]       # (batch, seq, 1)
                
                # Split targets into action and distance components
                actions_action = actions[:, :, :self.action_dim]     # (batch, seq, action_dim)
                actions_distance = actions[:, :, self.action_dim:]   # (batch, seq, 1)
                
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                loss_dict = dict()
                
                # L1 loss for action part
                all_l1_action = F.l1_loss(actions_action, a_hat_action, reduction='none')
                l1_action = (all_l1_action * ~is_pad.unsqueeze(-1)).mean()
                loss_dict['l1_action'] = l1_action
                
                # Binary Cross Entropy loss for distance part
                # BCEWithLogitsLoss applies sigmoid internally (expects raw logits)
                bce_loss = F.binary_cross_entropy_with_logits(
                    a_hat_distance.squeeze(-1),           # (batch, seq)
                    actions_distance.squeeze(-1),         # (batch, seq)
                    reduction='none'
                )
                l1_distance = (bce_loss * ~is_pad).mean()
                loss_dict['bce_distance'] = l1_distance
                
                # Total loss
                loss_dict['kl'] = total_kld[0]
                loss_dict['loss'] = loss_dict['l1_action'] + loss_dict['bce_distance'] + loss_dict['kl'] * self.kl_weight
                return loss_dict
            else: # inference time
                a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
                return a_hat
            
    def configure_optimizers(self):
        return self.optimizer


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