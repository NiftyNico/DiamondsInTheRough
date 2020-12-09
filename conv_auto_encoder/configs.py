import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import pytorch_ssim
import json
import shutil
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import model_zoo
import ds
import visualization
import stats


def get_mse_pooled_config():
  return {
    'EXP_NAME' :     "modified_arch_mse_retrain",
    'EXP_MODEL':      model_zoo.ConvAutoEncoder, 
    'MINERL_GYM_ENV':"MineRLTreechopVectorObf-v0",
    'BATCH_SIZE':     512,
    'NUM_WORKERS':    16,
    'CENTER_CHANNELS': False,
    'TRAIN_CRITERIA': nn.MSELoss(),
    'TEST_CRITERIA':  SSIM(data_range=1., size_average=True, channel=12),
    'LEARNING_RATE':  0.0005,
    'TRAIN_VIS_I':   [0, 150],
    'VALID_VIS_I':   [0, 150],
    'TRAIN_EPOCHS':   70,
    'STACK_CHANNELS': True,
  }

def get_mse_4conv_config():
  return {
    'EXP_NAME' :     "4conv_bse_retrain",
    'EXP_MODEL':      model_zoo.FourConvAutoEncoder, 
    'MINERL_GYM_ENV':"MineRLTreechopVectorObf-v0",
    'BATCH_SIZE':     512,
    'NUM_WORKERS':    16,
    'TRAIN_CRITERIA': nn.BCELoss(),
    'CENTER_CHANNELS': False,
    'TEST_CRITERIA':  SSIM(data_range=1., size_average=True, channel=12),
    'LEARNING_RATE':  0.0005,
    'TRAIN_VIS_I':   [0, 150],
    'VALID_VIS_I':   [0, 150],
    'TRAIN_EPOCHS':   70,
    'STACK_CHANNELS': True,
  }

def get_baseline_config():
  return {
    'EXP_NAME' :     "baseline_retrain",
    'EXP_MODEL':      model_zoo.BaselineConvAutoEncoder, 
    'MINERL_GYM_ENV':"MineRLTreechopVectorObf-v0",
    'BATCH_SIZE':     512,
    'NUM_WORKERS':    16,
    'TRAIN_CRITERIA': nn.BCELoss(),
    'CENTER_CHANNELS': False,
    'TEST_CRITERIA':  SSIM(data_range=1., size_average=True, channel=12),
    'LEARNING_RATE':  0.0005,
    'TRAIN_VIS_I':   [0, 150],
    'VALID_VIS_I':   [0, 150],
    'TRAIN_EPOCHS':   70,
    'STACK_CHANNELS': True,
  }

