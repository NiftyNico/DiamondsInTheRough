# System Dependencies
import sys
import os
import shutil
import math

# External Dependencies
import torch
import torchvision
from torch.autograd import Variable
from torchviz import make_dot

import minerl
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm, trange
import numpy as np

from PIL import Image
from matplotlib import cm

# model = torchvision.models.squeezenet1_1(pretrained=True)

# X = Variable(torch.randn(64, 3, 300, 300))
# y = model(X)
# dot = make_dot(y, params=dict(model.named_parameters()))
# dot.format = 'png'
# dot.render('torchviz-sample')

BATCH_SIZE = 16

class SqueezenetKMeans:
  def preprocess(povs):
    SQUEEZENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])
    SQUEEZENET_STD  = torch.FloatTensor([0.229, 0.224, 0.225])

    povs  = povs.float()
    povs /= 255.
    povs  = (povs - SQUEEZENET_MEAN) / SQUEEZENET_STD
    povs  = povs.permute(0, 3, 1, 2)
    return Variable(povs)

  def postprocess(conv_X):
    conv_X   = torch.flatten(conv_X, start_dim=1)
    conv_X   = conv_X.detach().numpy()
    return conv_X

  def __init__(self, n_clusters=32):
    global BATCH_SIZE
    self.n_clusters = n_clusters
    self.kmeans     = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=0, batch_size=BATCH_SIZE)
    self.model      = torchvision.models.squeezenet1_1(pretrained=True).eval()

  def forward(self, X):
    final_conv_module = self.model.features[12]
    conv_outs = None
    def activation_hook(self, input, output):
      nonlocal conv_outs
      conv_outs = output
    final_conv_module.register_forward_hook(activation_hook)

    povs   = SqueezenetKMeans.preprocess(X)
    _      = self.model(povs)
    conv_X = SqueezenetKMeans.postprocess(conv_outs)
    return conv_X

  def fit(self, X):
    conv_X = self.forward(X)
    self.kmeans.partial_fit(conv_X)

  def eval(self, X):
    X = self.forward(X)
    return self.kmeans.predict(X)

def pov_batcher(num_batches, dataset):
  global BATCH_SIZE
  dat             = minerl.data.make(dataset)
  batch_iter      = dat.batch_iter(batch_size=BATCH_SIZE, seq_len=32, num_epochs=1, preload_buffer_size=20)
  yielded_batches = 0

  for current_state, action, reward, next_state, done in batch_iter:
    if yielded_batches == num_batches:
      return

    povs = current_state['pov']
    povs = torch.from_numpy(np.reshape(povs, (povs.shape[0] * povs.shape[1], povs.shape[2], povs.shape[3], povs.shape[4])))

    yield povs
    yielded_batches += 1

def train_test_batcher(n_train_batches, n_test_batches, dataset):
  num_batches = n_train_batches + n_test_batches
  batcher     = pov_batcher(num_batches, dataset)
  yielded_batches = 0

  pbar = tqdm(total=num_batches)
  for batch in batcher:
    is_train = yielded_batches < n_train_batches
    yield is_train, batch
    yielded_batches += 1
    pbar.update(1)

  pbar.close()

def export(X, y, out_root, rm=False):
  assert len(X) == len(y)
  if rm and os.path.exists(out_root):
    shutil.rmtree(out_root, ignore_errors=True)

  img_X = np.uint8(X.detach().numpy())
  for i in range(0, len(y)):
    containing_dir_path = os.path.join(out_root, f"{y[i]}/")
    if not os.path.exists(containing_dir_path):
      os.makedirs(containing_dir_path)
    file_path = os.path.join(containing_dir_path, f"{i}.png")
    Image.fromarray(img_X[i]).save(file_path)

def make_collages(out_root):
  for folder in os.listdir(out_root):
    out_folder = os.path.join(out_root, folder)
    collage = make_collage(out_folder)
    out_path = out_folder + ".png"
    collage.save(out_path)

def make_collage(out_root):
  files = os.listdir(out_root)
  if len(files) == 0:
    return

  def image_for_filename(filename):
    nonlocal out_root
    filepath = os.path.join(out_root, filename)
    return np.array(Image.open(filepath).convert('RGB'))

  first_image  = image_for_filename(files[0])
  width_height = math.ceil(math.sqrt(len(files))) * first_image.shape[0]
  collage      = np.zeros((width_height, width_height, 3))
  # print(collage.shape)
  row_start = 0
  col_start = 0
  for filename in files:
    image   = image_for_filename(filename)
    row_end = row_start + image.shape[0]
    col_end = col_start + image.shape[1]
    # print(f"{row_start}:{row_end}, {col_start}:{col_end}")
    collage[row_start:row_end, col_start:col_end, :] = image

    col_start += image.shape[1]
    if col_start == width_height:
      row_start += image.shape[0]
      col_start  = 0

  return Image.fromarray(np.uint8(collage))

def main():
  batcher = train_test_batcher(50, 50, 'MineRLNavigateVectorObf-v0')
  kmeans = SqueezenetKMeans(n_clusters=16)
  is_first_test = True
  out_root      = "./out/dataset_exploration/images"
  # for is_train, X in batcher:
  #   if is_train:
  #     # print('train')
  #     kmeans.fit(X)
  #   else:
  #     # print('eval')
  #     y = kmeans.eval(X)
  #     export(X, y, out_root, rm=is_first_test)
  #     is_first_test = False

  make_collages(out_root)

  # X_train, X_test = povs[:200], povs[200:]
  # print(X_test[0])
  # print(povs.shape)

  # kmeans.fit(X_train)
  # y_test = kmeans.eval(X_test)
  # print(X_test[0])
  # batch_iter = data.batch_iter(batch_size=1, num_epochs=1, seq_len=32)

  # povs = []
  # for current_state, action, reward, next_state, done in batch_iter:
  #   pov = current_state[pov]
  #   povs.append(pov)
  #   return
          # # Print the POV @ the first step of the sequence
          # print(current_state['pov'][0])

          # # Print the final reward pf the sequence!
          # print(reward[-1])

          # # Check if final (next_state) is terminal.
          # print(done[-1])

          # # ... do something with the data.
          # print("At the end of trajectories the length"
          #       "can be < max_sequence_len", len(reward))
  sys.exit(0)
if __name__ == "__main__":
  main()