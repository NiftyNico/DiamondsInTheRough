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
import train


config = {
  'EXP_NAME' :     "modified_arch",
  'EXP_MODEL':      model_zoo.ConvAutoEncoder, 
  'MINERL_GYM_ENV':"MineRLTreechopVectorObf-v0",
  'BATCH_SIZE':     512,
  'NUM_WORKERS':    16,
  'CRITERIA':       nn.BCELoss(), #pytorch_ssim.SSIM(window_size=8)
  'LEARNING_RATE':  0.001
}

LOG_ROOT         = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"logs/{config['EXP_NAME']}")
STATS_FILENAME   = "stats.csv"
MODEL_FILENAME   = 'model.pt'
ENCODER_FILENAME = 'encoder.pt'
FORCE_TRAIN      = True
# FORCE_TRAIN      = False
CONFIG_FILENAME  = "config.json"

if FORCE_TRAIN and os.path.exists(LOG_ROOT):
  shutil.rmtree(LOG_ROOT)
os.makedirs(LOG_ROOT, exist_ok=True)

stats_path       = os.path.join(LOG_ROOT, STATS_FILENAME)
model_path       = os.path.join(LOG_ROOT, MODEL_FILENAME)
encoder_path     = os.path.join(LOG_ROOT, ENCODER_FILENAME)
config_path = os.path.join(LOG_ROOT, CONFIG_FILENAME)
config_json = {str(key): str(value) for key, value in config.items()}
with open(config_path, 'w+') as config_json_fp:
  json.dump(config_json, config_json_fp, indent=2)

def visualize(epoch_i, nograd_model, out_dir=LOG_ROOT):
  x_in  = torch.tensor((np.array([train_ds[0]])))
  x_out = nograd_model(x_in)
  x_out_0 = x_out[0].cpu().detach().numpy()
  x_in_0  = x_in[0].cpu().detach().numpy()

  out_dir = os.path.join(out_dir, "train_0_images/")
  visualization.visualize_ds_image(out_dir, "expected", x_in_0)
  visualization.visualize_ds_image(out_dir, str(epoch_i), x_out_0)

def update_best(nograd_model, stats, out_path=model_path):
  if not stats.is_last_epoch_best():
    return
  torch.save(nograd_model, out_path)

def valid_callback(epoch_i, nograd_model, stats):
  visualize(epoch_i, nograd_model)
  update_best(nograd_model, stats)

# assert os.path.exists(stats_path) == os.path.exists(model_path)

train_ds = ds.MineRLFrameDataset(config["MINERL_GYM_ENV"], "train")
valid_ds = ds.MineRLFrameDataset(config["MINERL_GYM_ENV"], "valid")
test_ds  = ds.MineRLFrameDataset(config["MINERL_GYM_ENV"], "test")

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"], shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"], shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"], shuffle=True)

n_input_channels = train_ds[0].shape[0]
model = config["EXP_MODEL"](n_input_channels=n_input_channels)
if os.path.exists(model_path):
  print("Loading cached model...")
  model = torch.load(model_path)
if torch.cuda.is_available():
  print(f"Moving model to cuda...")
  model = model.cuda()
if os.path.exists(model_path) and not os.path.exists(encoder_path):
  print('Encoder does not exist, exporting...')
  model.export_encoder(encoder_path)

X_shape = tuple([1] + list(train_ds[0].shape))
visualization.visualize_model(LOG_ROOT, MODEL_FILENAME,   model,               X_shape=X_shape)
visualization.visualize_model(LOG_ROOT, ENCODER_FILENAME, model.get_encoder(), X_shape=X_shape)

crit  = config["CRITERIA"]
opt   = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
stats = stats.Stats(stats_path)
stats.plot()

train.train(model, crit, opt, 10, train_loader, valid_loader, stats, valid_epoch_freq=2, valid_callback=valid_callback)
