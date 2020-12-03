import minerl
import torch.utils.data
import numpy as np
import os
import glob
import shutil
from tqdm import tqdm


def create_train_valid_test_split(env, train_percent, valid_percent):
  dat = minerl.data.make(env)
  episode_names = np.array(dat.get_trajectory_names())
  num_episodes  = len(episode_names)

  test_percent  = 1.0 - (train_percent + valid_percent)

  random_i = np.random.choice(np.arange(0, num_episodes), num_episodes, replace=False)
  train_max = train_percent * num_episodes
  valid_min = train_max
  valid_max = valid_min + valid_percent * num_episodes
  test_min  = valid_max

  train_i = (random_i <= train_max)
  valid_i = (random_i > valid_min) & (random_i <= valid_max)
  test_i  = random_i > test_min
  train_episode_names = episode_names[train_i]
  valid_episode_names = episode_names[valid_i]
  test_episode_names  = episode_names[test_i]
  assert (len(train_episode_names) + len(valid_episode_names) + len(test_episode_names)) == num_episodes

  return train_episode_names, valid_episode_names, test_episode_names

def get_ds_path(env, name):
  data_root = os.getenv('MINERL_DATA_ROOT')
  assert data_root is not None
  data_root = os.path.join(data_root, env)
  assert os.path.exists(data_root)
  data_root = os.path.join(data_root, "frame_dump", name)
  return data_root

def ensure_dataset(env, name, episodes, frame_skip=0, frame_stack=1, force_create=True, verbose=True):
  if verbose:
    print(f"Creating {name}...")
  
  out_ds_path = get_ds_path(env, name)
  if not force_create and os.path.exists(out_ds_path):
    return out_ds_path

  if os.path.exists(out_ds_path):
    shutil.rmtree(out_ds_path)
  os.makedirs(out_ds_path, exist_ok=True)

  out_frame_samples_i = 0
  frames_skipped = 0
  dat = minerl.data.make(env)
  for episode_name in episodes:
    traj = dat.load_data(episode_name)

    pov_queue = {}
    for i in range(0, frame_skip):
      pov_queue[i] = []

    done    = False
    frame_i = 0
    while True:
      if done:
        break

      ob, act, rw, next_ob, done = next(traj)
      assert 'pov' in ob
      pov = ob['pov']
      pov = np.moveaxis(pov, -1, 0)

      pov_i = frame_i % frame_skip
      pov_queue[pov_i].append(pov)
      assert len(pov_queue[pov_i]) <= frame_stack

      frame_i += 1
      if len(pov_queue[pov_i]) != frame_stack:
        continue
      assert len(pov_queue[pov_i]) == frame_stack

      frame_path = os.path.join(out_ds_path, str(out_frame_samples_i))
      frame = np.stack(pov_queue[pov_i], 0)
      np.save(frame_path, pov_queue[pov_i])
      out_frame_samples_i += 1
      pov_queue[pov_i] = pov_queue[pov_i][1:]
      assert len(pov_queue[pov_i]) <= frame_stack

  return out_ds_path

def get_per_channel_mean_std(data_loader, preprocess_fn):
  means, stds = None, None
  num_samples = 0
  
  for frames_batch in tqdm(data_loader):
    frame_samples = len(frames_batch)
    num_samples += frame_samples
    frames_batch = preprocess_fn(frames_batch)

    new_means = torch.mean(frames_batch, dim=[0, 2, 3], keepdim=True) * frame_samples
    new_stds  = torch.std(frames_batch,  dim=[0, 2, 3], keepdim=True) * frame_samples
    if means is None and stds is None:
      means, stds = preprocess_fn(torch.zeros(new_means.shape)), preprocess_fn(torch.zeros(new_stds.shape))
    means += new_means
    stds  += new_stds

  means /= num_samples
  stds  /= num_samples
  return means, stds
    

  # pixels = (pixels - mean) / std

class MineRLFrameDataset(torch.utils.data.Dataset):
  def __init__(self, env, name, stack_channels):
    self.__name         = name
    self.data_path      = get_ds_path(env, name)
    self.num_samples    = len(glob.glob1(self.data_path, "*.npy"))
    self.stack_channels = stack_channels

  def get_name(self):
    return self.__name

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    np_filename = f"{idx}.npy"
    np_path = os.path.join(self.data_path, np_filename)
    if not os.path.exists(np_path):
      print(f"Unable to find {np_path}")
      assert False

    frame = np.load(np_path)
    if self.stack_channels:
      frame = np.reshape(frame, (frame.shape[0] * frame.shape[1], frame.shape[2], frame.shape[3]), order='C')
    frame = torch.from_numpy(frame)
    return frame
