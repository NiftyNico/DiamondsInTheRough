import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os 
import glob

import torch
from torch.autograd import Variable
from torchviz import make_dot


def load_rgb(image_path):
  return np.array(Image.open(image_path))

def save_rgb(out_path, rgb):
  if np.max(rgb) <= 1.:
    rgb *= 255
  rgb = np.uint8(rgb)
  im = Image.fromarray(rgb)
  im.save(out_path)

def plot_internal(out_root, title, y_label, epoch_and_train_vals, epoch_and_valid_vals):
  if len(epoch_and_train_vals) == 0:
    return

  train_epochs, train_vals = list(zip(*epoch_and_train_vals))
  valid_epochs, valid_vals = list(zip(*epoch_and_valid_vals))

  fig, ax = plt.subplots()
  ax.plot(train_epochs, train_vals, color='r')
  ax.plot(valid_epochs, valid_vals, color='b')
  ax.set_xlabel('Epoch')
  ax.set_ylabel(y_label)
  ax.set_title(title)
  ax.legend(["Train", "Valid"])

  filename = title.lower().replace(" ", "_") + ".png"
  out_path = os.path.join(out_root, filename)
  plt.savefig(out_path)
  plt.close('all')

def plot_ssim(out_root, epoch_and_train_ssim, epoch_and_valid_ssim):
  plot_internal(out_root, "Performance", 'SSIM', epoch_and_train_ssim, epoch_and_valid_ssim)

def plot_learning_curve(out_root, epoch_and_train_loss, epoch_and_valid_loss):
  plot_internal(out_root, "Learning Curve", 'Loss', epoch_and_train_loss, epoch_and_valid_loss)

def ds_image_to_rgb(frames, single_frame_channel_count=3):
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
  return new_frame

def save_ds_image(out_root, image_name, frames):
  os.makedirs(out_root, exist_ok=True)
  new_frame = ds_image_to_rgb(frames)
  out_path = os.path.join(out_root, f"{image_name}.png")
  save_rgb(out_path, new_frame)
  return out_path

def merge_ds_image_visualizations(out_root, image_name, image_paths):
  if len(image_paths) == 0:
    return
  
  in_height, out_width, out_channels = load_rgb(image_paths[0]).shape
  out_height = in_height * len(image_paths)
  out_rgb = np.zeros((out_height, out_width, out_channels))
  for i, image_path in enumerate(image_paths):
    h_start, h_end = i * in_height, (i + 1) * in_height
    out_rgb[h_start:h_end, :, :] = load_rgb(image_path)

  out_path = os.path.join(out_root, f"{image_name}.png")
  save_rgb(out_path, out_rgb)
  return out_path

def visualize_epoch(epoch_i, out_dir, x_in, x_out, max_merge_count=4):
  expected_path = save_ds_image(out_dir, "expected", x_in)
  _             = save_ds_image(out_dir, str(epoch_i), x_out)

  def sort_by_int(filepath):
    return int(os.path.splitext(os.path.basename(filepath))[0])
  visualization_paths = glob.glob1(out_dir, "[0123456789]*.png")
  visualization_paths = [os.path.join(out_dir, filename) for filename in visualization_paths]
  visualization_paths.sort(key=sort_by_int)
  visualization_paths.append(expected_path)

  if len(visualization_paths) > max_merge_count:
    num_to_find = max_merge_count - 2 # one for final, one for expected
    include_i = np.arange(0, len(visualization_paths) - 2, int(len(visualization_paths) / num_to_find))
    new_visualization_paths = []
    for i in include_i:
      new_visualization_paths.append(visualization_paths[int(i)])
    new_visualization_paths.append(visualization_paths[-2])
    new_visualization_paths.append(visualization_paths[-1])
    visualization_paths = new_visualization_paths

  merge_ds_image_visualizations(out_dir, "combined", visualization_paths)

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