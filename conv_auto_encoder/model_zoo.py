import torch
import torch.nn as nn

class FloatNN(nn.Module):
  def __init__(self):
    super(FloatNN, self).__init__()

    self.c_means = None
    self.c_stds  = None

  def preprocess(self, x):
    x = x.to(next(self.parameters()).device)
    if x.dtype == torch.uint8:
      x = x.float() / 256.0
    
    if self.should_c_center():
      x -= self.c_means
      x /= self.c_stds

    return x

  def postprocess(self, x):
    if self.should_c_center():
      x *= self.c_stds
      x += self.c_means
    return x

  def should_c_center(self):
    return self.c_means is not None and self.c_stds is not None

  def set_c_centering(self, means, stds):
    self.c_means = means
    self.c_stds  = stds

  def set_criteria(self, crit):
    self.__crit = crit
  def get_criteria(self):
    return self.__crit

  def set_optimizer(self, opt):
    self.__opt  = opt
  def get_optimizer(self):
    return self.__opt

  def set_scheduler(self, sched):
    self.__sched = sched
  def get_scheduler(self):
    return self.__sched

class RNNConvAutoEncoder(FloatNN):
  def get_encoder_layer(in_c, out_c):
    return nn.Sequential(
      nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, 2)
    )

  def get_decoder_layer(in_c, out_c):
    return nn.Sequential(
      nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
      nn.ReLU(),
    )

  def __init__(self, n_input_channels):
    self.encoder_layers = nn.Sequential(
      LSTMConvAutoEncoder.get_encoder_layer(n_input_channels, 32),
      LSTMConvAutoEncoder.get_encoder_layer(32, 16),
      LSTMConvAutoEncoder.get_encoder_layer(16, 8)
    )

    self.decoder_layers = nn.Sequential(
      LSTMConvAutoEncoder.get_decoder_layer(8, 16),
      LSTMConvAutoEncoder.get_decoder_layer(16, 32),

      nn.ConvTranspose2d(32, n_input_channels, 2, stride=2),
      nn.Sigmoid()
    )

  def forward(self, x):
    assert len(x.shape) == 5 # (sample, frame, c, h, w)

class FourConvAutoEncoder(FloatNN):
  def __init__(self, n_input_channels):
    super(FourConvAutoEncoder, self).__init__()
    
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

class ConvAutoEncoder(FloatNN):
  def __init__(self, n_input_channels):
    super(ConvAutoEncoder, self).__init__()
    
    self.pool = nn.MaxPool2d(2, 2)

    self.encoder_layers = nn.Sequential(
      nn.Conv2d(n_input_channels, 32, 3, stride=1, padding=1),
      nn.ReLU(),
      self.pool,

      # nn.Conv2d(64, 32, 3, stride=1, padding=1),
      # nn.ReLU(),
      # self.pool,

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

      # nn.ConvTranspose2d(32, 64, 2, stride=2),
      # nn.ReLU(),

      nn.ConvTranspose2d(32, n_input_channels, 2, stride=2),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.preprocess(x)
    x = self.encoder_layers(x)
    x = self.decoder_layers(x)
    return x

  def get_encoder(self):
    return self.encoder_layers


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