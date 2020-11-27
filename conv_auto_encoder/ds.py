import minerl
import torch.utils.data
import numpy as np
import os
import glob
import shutil

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


def ensure_dataset(name, episodes, root_ds_path, force_create=False, verbose=True):
  out_ds_path = os.path.join(root_ds_path, name)
  if not force_create and os.path.exists(out_ds_path):
    return out_ds_path

  if os.path.exists(out_ds_path):
    shutil.rmtree(out_ds_path)
  os.makedirs(out_ds_path, exist_ok=True)

  frame_i = 0
  frames_skipped = 0
  for episode_name in episodes:
    traj = dat.load_data(episode_name)

    prev_pov = None
    done = False
    while True:
      if done:
        break

      ob, act, rw, next_ob, done = next(traj)
      assert 'pov' in ob
      pov = ob['pov']
      if prev_pov is not None and np.allclose(pov, prev_pov):
        frames_skipped += 1
        continue

      frame_path = os.path.join(out_ds_path, str(frame_i))
      np.save(frame_path, pov)
      frame_i += 1
      prev_pov = pov

  if verbose:
    print(f"frames_skipped:{frames_skipped}")
  return out_ds_path

class MineRLFrameDataset(torch.utils.data.Dataset):
  def __init__(self, data_path):
    self.data_path   = data_path
    self.num_samples = len(glob.glob1(data_path, "*.npy"))

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    np_filename = f"{idx}.npy"
    np_path = os.path.join(self.data_path, np_filename)
    frame = np.load(np_path)
    frame = np.moveaxis(frame, -1, 0)
    return frame
