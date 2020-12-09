import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import pytorch_ssim
import json
import shutil
import glob
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import model_zoo
import ds
import visualization
import stats
import train
import configs


def print_banner(text):
  print('-' * 80)
  print(text)
  print('-' * 80)

config = configs.get_baseline_config()

LOG_ROOT          = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"logs/{config['EXP_NAME']}")
STATS_FILENAME    = "stats.csv"
MODEL_FILENAME    = 'model.pt'
ENCODER_FILENAME  = 'encoder.pt'
MAX_MERGE_COUNT   = 4
FORCE_TRAIN       = True
# FORCE_TRAIN       = False
CONFIG_FILENAME   = "config.json"
TEST_RES_FILENAME = "test_results.txt"   

if FORCE_TRAIN and os.path.exists(LOG_ROOT):
  shutil.rmtree(LOG_ROOT)
os.makedirs(LOG_ROOT, exist_ok=True)

test_res_path    = os.path.join(LOG_ROOT, TEST_RES_FILENAME)
stats_path       = os.path.join(LOG_ROOT, STATS_FILENAME)
model_path       = os.path.join(LOG_ROOT, MODEL_FILENAME)
encoder_path     = os.path.join(LOG_ROOT, ENCODER_FILENAME)
config_path      = os.path.join(LOG_ROOT, CONFIG_FILENAME)

if os.path.exists(test_res_path):
  print_banner(f'Test results already exist for {config["EXP_NAME"]} ... terminating')

config_json = {str(key): str(value) for key, value in config.items()}
with open(config_path, 'w+') as config_json_fp:
  json.dump(config_json, config_json_fp, indent=2)

def visualize_sample(epoch_i, dataset, sample_i, nograd_model, max_merge_count=MAX_MERGE_COUNT, out_dir=LOG_ROOT):
  x_in    = dataset[sample_i].unsqueeze(0)
  x_out   = nograd_model.postprocess(nograd_model(x_in))
  x_out_0 = x_out[0].cpu().detach().numpy()
  x_in_0  = x_in[0].cpu().detach().numpy()

  out_dir = os.path.join(out_dir, f"{dataset.get_name()}_{sample_i}_images/")
  visualization.visualize_epoch(epoch_i, out_dir, x_in_0, x_out_0, max_merge_count=max_merge_count)

def visualize_epoch(epoch_i, nograd_model, train_vis_i=config['TRAIN_VIS_I'], valid_vis_i=config['VALID_VIS_I']):
  for i in train_vis_i:
    visualize_sample(epoch_i, train_ds, i, nograd_model)
  for i in valid_vis_i:
    visualize_sample(epoch_i, valid_ds, i, nograd_model)

def update_best(nograd_model, stats, model_path=model_path, encoder_path=encoder_path):
  if not stats.is_last_epoch_best():
    return
  
  grad_model = model.train(True)
  torch.save(grad_model, model_path)
  torch.save(grad_model.get_encoder(), encoder_path)
  grad_model = model.train(False)

def valid_callback(epoch_i, nograd_model, stats):
  visualize_epoch(epoch_i, nograd_model)
  update_best(nograd_model, stats)

# assert os.path.exists(stats_path) == os.path.exists(model_path)

train_ds = ds.MineRLFrameDataset(config["MINERL_GYM_ENV"], "train", config['STACK_CHANNELS'])
valid_ds = ds.MineRLFrameDataset(config["MINERL_GYM_ENV"], "valid", config['STACK_CHANNELS'])
test_ds  = ds.MineRLFrameDataset(config["MINERL_GYM_ENV"], "test",  config['STACK_CHANNELS'])

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"], shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"], shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"], shuffle=True)

model = None
if not os.path.exists(model_path):
  print_banner("Initializing new model...")
  n_input_channels = train_ds[0].shape[0]
  model = config["EXP_MODEL"](n_input_channels=n_input_channels)

  crit  = config["TRAIN_CRITERIA"]
  opt   = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
  sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')
  model.set_criteria(crit)
  model.set_optimizer(opt)
  model.set_scheduler(sched)

  X_shape = tuple([1] + list(train_ds[0].shape))
  visualization.visualize_model(LOG_ROOT, MODEL_FILENAME,   model,               X_shape=X_shape)
  visualization.visualize_model(LOG_ROOT, ENCODER_FILENAME, model.get_encoder(), X_shape=X_shape)
else:
  print_banner("Loading cached model...")
  model = torch.load(model_path)

if torch.cuda.is_available():
  print_banner(f"Moving model to cuda...")
  model = model.cuda()


if config['CENTER_CHANNELS']:
  print_banner('Calculating channel centerings...')
  c_means, c_stds = ds.get_per_channel_mean_std(train_loader, model.preprocess)
  model.set_c_centering(c_means, c_stds)

stats = stats.Stats(stats_path)
stats.plot()

print_banner(f'Training {config["TRAIN_EPOCHS"]} epochs')
train.train(model, config['TRAIN_EPOCHS'], train_loader, valid_loader, stats, valid_epoch_freq=1, valid_callback=valid_callback, test_criteria=config['TEST_CRITERIA'])

print_banner(f'Calculating test results ...')
test_result = train.valid_epoch(model, test_loader, test_criteria=config['TEST_CRITERIA'])
with open(test_res_path, "w+") as test_fp:
  test_fp.write(f"{test_result}\n")

print_banner(f'Model created successfully...')