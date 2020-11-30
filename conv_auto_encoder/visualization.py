import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os 

import torch
from torch.autograd import Variable
from torchviz import make_dot


def plot_learning_curve(out_root, epoch_and_train_loss, epoch_and_valid_loss):
  if len(epoch_and_train_loss) == 0:
    return

  train_epochs, train_losses = list(zip(*epoch_and_train_loss))
  valid_epochs, valid_losses = list(zip(*epoch_and_valid_loss))

  fig, ax = plt.subplots()
  ax.scatter(train_epochs, train_losses, color='r')
  ax.scatter(valid_epochs, valid_losses, color='b')
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Loss')
  ax.set_title('Learning Curve')

  out_path = os.path.join(out_root, 'learning_curve.png')
  plt.savefig(out_path)
  plt.close('all')

def visualize_ds_image(out_root, image_name, frames, single_frame_channel_count=3):
  os.makedirs(out_root, exist_ok=True)
  frames = np.moveaxis(frames, 0, -1)

  orig_height = frames.shape[0]
  orig_width  = frames.shape[1]

  num_frames = int(frames.shape[-1] / single_frame_channel_count)
  new_height = orig_height
  new_width  = orig_width * num_frames
  new_frame  = np.zeros((new_height, new_width, single_frame_channel_count))
  for i in range(0, num_frames):
    start_h = 0
    end_h   = orig_height
    start_w = i * orig_width
    end_w   = start_w + orig_width
    start_c = i * single_frame_channel_count
    end_c   = start_c + single_frame_channel_count
    new_frame[start_h:end_h, start_w:end_w, :] = frames[:, :, start_c:end_c]

  if np.max(new_frame) <= 1.:
    new_frame *= 255
  new_frame = np.uint8(new_frame)
  im = Image.fromarray(new_frame)
  out_path = os.path.join(out_root, f"{image_name}.png")
  im.save(out_path)

def merge_ds_image_visualizations(out_root):

def visualize_model(out_root, model_name, model, X_shape):
  model.eval()
  os.makedirs(out_root, exist_ok=True)

  X   = Variable(torch.randn(X_shape))
  X   = X.to(next(model.parameters()).device)
  y   = model(X)
  dot = make_dot(y, params=dict(model.named_parameters()))
  
  dot.format = 'png'
  out_path = os.path.join(out_root, f"{model_name}")
  dot.render(out_path)