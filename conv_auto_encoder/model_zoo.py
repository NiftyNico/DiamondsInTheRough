import torch
import torch.nn as nn

class FloatNN(nn.Module):
  def __init__(self):
    super(FloatNN, self).__init__()

  def preprocess(self, x):
    x = x.to(next(self.parameters()).device)
    if x.dtype == torch.uint8:
      x = x.float() / 256.0
    return x

class ConvAutoEncoder(FloatNN):
  def __init__(self, n_input_channels):
    super(ConvAutoEncoder, self).__init__()
    
    self.pool = nn.MaxPool2d(2, 2)

    self.encoder_layers = nn.Sequential(
      nn.Conv2d(n_input_channels, 64, 3, stride=1, padding=1),
      nn.ReLU(),
      self.pool,

      nn.Conv2d(64, 32, 3, stride=1, padding=1),
      nn.ReLU(),
      self.pool,

      nn.Conv2d(32, 16, 3, stride=1, padding=1),
      nn.ReLU(),
      self.pool,

      nn.Conv2d(16, 8, 3, stride=1, padding=1),
      nn.ReLU(),
      self.pool
    )

    self.decoder_layers = nn.Sequential(
      nn.ConvTranspose2d(8, 16, 2, stride=2),
      nn.ReLU(),

      nn.ConvTranspose2d(16, 32, 2, stride=2),
      nn.ReLU(),

      nn.ConvTranspose2d(32, 64, 2, stride=2),
      nn.ReLU(),

      nn.ConvTranspose2d(64, n_input_channels, 2, stride=2),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.preprocess(x)
    x = self.encoder_layers(x)
    x = self.decoder_layers(x)
    return x

  def get_encoder(self):
    return self.encoder_layers

  def export_encoder(self, out_path):
    torch.save(out_path, self.get_encoder())


class BaselineConvAutoEncoder(FloatNN):
  def __init__(self, n_input_channels, activation=nn.ReLU()):
    super(BaselineConvAutoEncoder, self).__init__()

                # nn.Conv2d(n_input_channels, 32, 8, stride=4),
                # nn.Conv2d(32, 64, 4, stride=2),
                # nn.Conv2d(64, 64, 3, stride=1),

    self.enc_l1 = nn.Sequential(
      nn.Conv2d(n_input_channels, 32, 8, stride=4),
      activation
    )

    self.enc_l2 = nn.Sequential(
      nn.Conv2d(32, 64, 4, stride=2),
      activation
    )

    self.enc_l3 = nn.Sequential(
      nn.Conv2d(64, 64, 3, stride=1),
      activation
    )

    self.dec_l1 = nn.Sequential(
      nn.ConvTranspose2d(64, 64, 2, stride=2, padding=1),
      activation
    )

    self.dec_l2 = nn.Sequential(
      nn.ConvTranspose2d(64, 32, 2, stride=3, padding=1),
      activation
    )

    self.dec_l3 = nn.Sequential(
      nn.ConvTranspose2d(32, n_input_channels, 8, stride=4, padding=0),
      nn.Sigmoid()
    )

  def get_encoder(self):
    modules = list(self.children())[:-3]
    encoder = nn.Sequential(*modules)
    return encoder

  def export_encoder(self, out_path):
    torch.save(out_path, self.get_encoder())

  def forward(self, x):
    x = self.preprocess(x)
    # print(x.shape)
    x = self.enc_l1(x)
    # print(x.shape)
    x = self.enc_l2(x)
    # print(x.shape)
    x = self.enc_l3(x)
    # print(x.shape)
    x = self.dec_l1(x)
    # print(x.shape)
    x = self.dec_l2(x)
    # print(x.shape)
    x = self.dec_l3(x)
    # print(x.shape)
    return x