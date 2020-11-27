import torch.nn as nn


class ConvAutoEncoder(nn.Module):
  def __init__(self, n_input_channels):
    super(ConvAutoEncoder, self).__init__()
    
    self.pool = nn.MaxPool2d(2, 2)

    self.enc_l1 = nn.Sequential(
      nn.Conv2d(n_input_channels, 32, 3, stride=1, padding=1),
      nn.ReLU(),
      self.pool
    )

    self.enc_l2 = nn.Sequential(
      nn.Conv2d(32, 16, 3, stride=1, padding=1),
      nn.ReLU(),
      self.pool
    )

    self.enc_l3 = nn.Sequential(
      nn.Conv2d(16, 8, 3, stride=1, padding=1),
      nn.ReLU(),
      self.pool
    )

    self.dec_l1 = nn.Sequential(
      nn.ConvTranspose2d(8, 16, 2, stride=2),
      nn.ReLU()
    )

    self.dec_l2 = nn.Sequential(
      nn.ConvTranspose2d(16, 32, 2, stride=2),
      nn.ReLU()
    )

    self.dec_l3 = nn.Sequential(
      nn.ConvTranspose2d(32, n_input_channels, 2, stride=2),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.enc_l1(x)
    x = self.enc_l2(x)
    x = self.enc_l3(x)
    x = self.dec_l1(x)
    x = self.dec_l2(x)
    x = self.dec_l3(x)
    return x