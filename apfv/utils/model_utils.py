import numpy as np
import torch
import torchvision

import pdb


class Normalize(torch.nn.Module):
  def __init__(self, mean: tuple, std: tuple):
    super(Normalize, self).__init__()
    self._mean = mean
    self._std = std

  def forward(self, tensor: torch.tensor) -> torch.tensor:
    N, C, H, W = tensor.shape
    mean = np.expand_dims(np.expand_dims(np.expand_dims(np.array(self._mean).astype(np.float32), axis=0), axis=-1), axis=-1)
    mean_tile = torch.tensor(np.tile(mean, (N, 1, H, W)))
    std = np.expand_dims(np.expand_dims(np.expand_dims(np.array(self._std).astype(np.float32), axis=0), axis=-1), axis=-1)
    std_tile = torch.tensor(np.tile(std, (N, 1, H, W)))
    if tensor.is_cuda:
      mean_tile = mean_tile.cuda()
      std_tile = std_tile.cuda()

    tensor = (tensor - mean_tile) / std_tile
    return tensor

class UnNormalize(torch.nn.Module):
  def __init__(self, mean: tuple, std: tuple):
    super(UnNormalize, self).__init__()
    self.mean = mean
    self.std = std

  def forward(self, tensor: torch.tensor) -> torch.tensor:
    N, C, H, W = tensor.shape
    mean = np.expand_dims(np.expand_dims(np.expand_dims(np.array(self._mean).astype(np.float32), axis=0), axis=-1), axis=-1)
    mean_tile = torch.tensor(np.tile(mean, (N, 1, H, W)))
    std = np.expand_dims(np.expand_dims(np.expand_dims(np.array(self._std).astype(np.float32), axis=0), axis=-1), axis=-1)
    std_tile = torch.tensor(np.tile(std, (N, 1, H, W)))
    if tensor.is_cuda:
      mean_tile = mean_tile.cuda()
      std_tile = std_tile.cuda()

    tensor = tensor * std_tile + mean_tile
    return tensor