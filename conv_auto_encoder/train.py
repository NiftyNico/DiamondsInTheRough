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


def train_epoch(model, train_loader):
  model = model.train(True)
  epoch_loss = 0
  num_samples = 0
  for frames_batch in train_loader:
    x_in = model.preprocess(frames_batch)

    model.get_optimizer().zero_grad()
    x_out = model(x_in)
    loss = model.get_criteria()(x_out, x_in)
    loss.backward()
    model.get_optimizer().step()
    epoch_loss  += loss.detach().item() * len(x_in)
    num_samples += len(x_in)
    # break
  epoch_loss = epoch_loss / num_samples
  return epoch_loss

def valid_epoch(model, valid_loader, test_criteria=None):
  model = model.eval()
  epoch_loss = 0
  num_samples = 0
  for frames_batch in valid_loader:
    x_in = model.preprocess(frames_batch)
    x_out = model(x_in)

    criteria = model.get_criteria()
    if test_criteria is not None:
      # x_in  = visualization.ds_image_to_rgb(x_in.cpu().detach().numpy())
      # x_out = visualization.ds_image_to_rgb(x_out.cpu().detach().numpy())
      criteria = test_criteria
    loss = criteria(x_out, x_in)
  
    epoch_loss += loss.detach().item() * len(x_in)
    num_samples += len(x_in)
    # break
  epoch_loss = epoch_loss / num_samples
  return epoch_loss

def train(model, n_epochs, train_loader, valid_loader, stats, valid_epoch_freq=1, valid_callback=None, test_criteria=None, verbose=True):
  start_epoch = stats.get_num_epochs_completed()
  for epoch_i in tqdm(range(start_epoch, n_epochs)):
    train_loss = train_epoch(model, train_loader)

    valid_loss = None
    train_ssim = None
    valid_ssim = None
    if (epoch_i % valid_epoch_freq) == 0:
      valid_loss = valid_epoch(model, valid_loader)
      model.get_scheduler().step(valid_loss)

      if test_criteria is not None:
        train_ssim = valid_epoch(model, train_loader, test_criteria=test_criteria)
        valid_ssim = valid_epoch(model, valid_loader, test_criteria=test_criteria)

    stats.add_epoch(train_loss=train_loss, valid_loss=valid_loss, train_ssim=train_ssim, valid_ssim=valid_ssim)
    if valid_loss is not None and valid_callback is not None:
      valid_callback(epoch_i, model.eval(), stats)