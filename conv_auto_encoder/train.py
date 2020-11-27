import torch
from tqdm import tqdm

def preprocess(model, x_in):
  x_in = x_in.to(next(model.parameters()).device)
  if x_in.dtype == torch.uint8:
    x_in = x_in.float() / 256.0
  return x_in

def train_epoch(model, crit, opt, train_loader):
  model = model.train(True)
  epoch_loss = 0
  for frames_batch in train_loader:
    x_in  = preprocess(model, frames_batch)
    x_out = x_in

    opt.zero_grad()
    x_out = model(x_in)
    loss = crit(x_out, x_in)
    loss.backward()
    opt.step()
    epoch_loss += loss.detach().item() * len(x_in)
    break
        
  epoch_loss = epoch_loss / len(train_loader)
  return epoch_loss

def valid_epoch(model, crit, valid_loader):
  model = model.eval()
  epoch_loss = 0
  for frames_batch in valid_loader:
    x_in  = preprocess(model, frames_batch)
    x_out = model(x_in)

    loss = crit(x_out, x_in)
    epoch_loss += loss.detach().item() * len(x_in)
    break

  epoch_loss = epoch_loss / len(valid_loader)
  return epoch_loss

def train(model, crit, opt, n_epochs, train_loader, valid_loader, valid_epoch_freq=1, valid_callback=None, verbose=True):
  for epoch_i in tqdm(range(0, n_epochs)):
    train_loss = train_epoch(model, crit, opt, train_loader)

    valid_loss = float("-inf")
    if (epoch_i % valid_epoch_freq) == 0:
      valid_loss = valid_epoch(model, crit, train_loader)
      if valid_callback is not None:
        valid_callback(epoch_i, model.eval(), x_in)

    if verbose:
      print('{}, {:.6f}, {:.6f}'.format(epoch_i, train_loss, valid_loss))
