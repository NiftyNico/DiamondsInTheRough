import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import pytorch_ssim
import json
import shutil

import model_zoo
import ds
import visualization
import stats


def train_epoch(model, crit, opt, train_loader):
  model = model.train(True)
  epoch_loss = 0
  for frames_batch in train_loader:
    x_in = model.preprocess(frames_batch)

    opt.zero_grad()
    x_out = model(x_in)
    loss = crit(x_out, x_in)
    loss.backward()
    opt.step()
    epoch_loss += loss.detach().item() * len(x_in)
    # break
  epoch_loss = epoch_loss / len(train_loader)
  return epoch_loss

def valid_epoch(model, crit, valid_loader):
  model = model.eval()
  epoch_loss = 0
  for frames_batch in valid_loader:
    x_in = model.preprocess(frames_batch)
    x_out = model(x_in)

    loss = crit(x_out, x_in)
    epoch_loss += loss.detach().item() * len(x_in)
    # break
  epoch_loss = epoch_loss / len(valid_loader)
  return epoch_loss

def train(model, crit, opt, n_epochs, train_loader, valid_loader, stats, valid_epoch_freq=1, valid_callback=None, verbose=True):
  start_epoch = stats.get_num_epochs_completed()
  for epoch_i in tqdm(range(start_epoch, n_epochs)):
    train_loss = train_epoch(model, crit, opt, train_loader)

    valid_loss = None
    if (epoch_i % valid_epoch_freq) == 0:
      valid_loss = valid_epoch(model, crit, valid_loader)

    stats.add_epoch(train_loss=train_loss, valid_loss=valid_loss)
    if valid_loss is not None and valid_callback is not None:
      valid_callback(epoch_i, model.eval(), stats)