# Copyright 2021 Yantao Lu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torchvision.models as models
import torch

from apfv21.utils.imagenet_utils import get_imagenet_normalize
from apfv21.utils.model_utils import Normalize

import pdb


class MOBILENET_V2(torch.nn.Module):
  def __init__(self, is_normalize: bool=True):
    super(MOBILENET_V2, self).__init__()
    self._is_normalize = is_normalize
    img_mean, img_std = get_imagenet_normalize()
    self._normalize = Normalize(img_mean, img_std)
    self._model = models.mobilenet_v2(pretrained=True).cuda().eval()
    features = list(self._model.features)
    self._features = torch.nn.ModuleList(features).cuda().eval()

  def forward(self, input_t, internal=[]):
    if self._is_normalize:
      x = self._normalize(input_t)
    else:
      x = input_t
    pred = self._model(x)
    if len(internal) == 0:
      return [], pred
    
    layers = []
    for ii, model in enumerate(self._features):
      x = model(x)
      if(ii in internal):
      # if isinstance(model, torch.nn.modules.conv.Conv2d):
        layers.append(x)
    return layers, pred
