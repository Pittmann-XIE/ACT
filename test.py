from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG # must import first

import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt

from training.utils import *
import sys 
sys.path.append('/home/ipk311l-user/ws_ros2humble-main_lab/ACT/act_origin')

from detr.main import build_ACT_model_and_optimizer

# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task1')
args = parser.parse_args()
task = args.task

# configs
task_cfg = TASK_CONFIG
train_cfg = TRAIN_CONFIG
policy_config = POLICY_CONFIG
checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task)

# device
device = os.environ['DEVICE']

policy = build_ACT_model_and_optimizer(policy_config)
# print(policy)

