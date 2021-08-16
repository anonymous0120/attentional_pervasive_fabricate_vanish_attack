""" Adapted Resnet152 pytorch model that used as surrogate. """
import torchvision.models as models
import torch

from apfv21.utils.imagenet_utils import get_imagenet_normalize
from apfv21.utils.model_utils import Normalize

import pdb


class RESNET152(torch.nn.Module):
  def __init__(self, is_normalize: bool=True):
    super(RESNET152, self).__init__()
    self._is_normalize = is_normalize
    img_mean, img_std = get_imagenet_normalize()
    self._normalize = Normalize(img_mean, img_std)
    self._model = models.resnet152(pretrained=True).cuda().eval()
    features = list(self._model.children())
    #print(len(features))
    #for ii, model in enumerate(features):
    #    print(ii, model)
    self._features = torch.nn.ModuleList(features)

  def forward(self, input_t, internal: tuple=()):
    if self._is_normalize:
      x = self._normalize(input_t)
    else:
      x = input_t
    pred = self._model(x)

    hit_cnt = 0
    if len(internal) == 0:
      return [], pred
    
    layers = []
    for ii, model in enumerate(self._features):
      x = model(x)
      if ii in internal:
        hit_cnt += 1
        layers.append(x)
      if hit_cnt==len(internal):
        break
    return layers, pred

if __name__ == "__main__":
  Resnet152()