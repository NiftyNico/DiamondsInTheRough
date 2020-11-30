import pandas as pd
import numpy as np
import os
from pathlib import Path
import visualization


class Stats:
  def __init__(self, filepath):
    self.filepath = filepath
    if os.path.exists(filepath):
      self.df = pd.read_csv(filepath, index_col=0)
      return
    
    d = { 'Train_Loss': [], 'Valid_Loss': [] }
    self.df = pd.DataFrame(data=d)

  def add_epoch(self, train_loss=None, valid_loss=None, save=True):
    d = {}
    if train_loss is not None:
      d['Train_Loss'] = [train_loss]
    if valid_loss is not None:
      d['Valid_Loss'] = [valid_loss]

    new_df = pd.DataFrame(data=d)
    self.df = self.df.append(new_df, ignore_index=True)
    self.df.index.name = 'Epoch'
    if save:
      self.save()

  def get_num_epochs_completed(self):
    return len(self.df)

  def is_last_epoch_best(self):
    return np.min(self.df['Valid_Loss']) == self.df.iloc[-1]['Valid_Loss']

  def __str__(self):
    return str(self.df)

  def save(self, plot=True):
    path = Path(self.filepath)
    os.makedirs(path.parent, exist_ok=True)
    self.df.to_csv(self.filepath, index_label=self.df.index.name)

    if plot:
      self.plot()

  def plot(self):
    out_root = os.path.dirname(os.path.realpath(self.filepath))
    train_series = self.df["Train_Loss"]
    valid_series = self.df["Valid_Loss"]
    epoch_and_train_loss = list(zip(train_series.index, train_series))
    epoch_and_valid_loss = list(zip(valid_series.index, valid_series))
    visualization.plot_learning_curve(out_root, epoch_and_train_loss, epoch_and_valid_loss)



# stats = Stats()
# stats.add_epoch(train_loss=1, valid_loss=2)
# stats.add_epoch(train_loss=2)
# stats.add_epoch(train_loss=3)
# stats.add_epoch(train_loss=4, valid_loss=3)

# stats_path = "./logs/stats_test/stats.csv"
# stats.write(stats_path)
# stats = Stats.read(stats_path)


# print(stats)