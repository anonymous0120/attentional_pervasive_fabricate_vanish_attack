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
""" Script for attention model, which is used for AFV attack.
"""
import torch
import torch.nn as nn
import torchvision

from apfv21.attacks.models import fpn_hm

import pdb


class ForegroundAttentionFPN(nn.Module):
  """ Class for attention model of AFV attack.
  """
  def __init__(self, 
               num_classes: int,
               feature_channels: int,
               resize: tuple=None) -> None:
    """
    Parameters:
    ----------
    num_classes : int
      Number of classes
    feature_channels : int
      Number of feature channels
    resize : (int, int)
      interpolation input size for FPN matching.
    """
    super(ForegroundAttentionFPN, self).__init__()
    self._num_classes = num_classes
    self._feature_channels = feature_channels
    if resize != None:
      assert isinstance(resize, tuple)
      assert len(resize) == 2
      assert isinstance(resize[0], int)
      assert isinstance(resize[1], int)
    self._resize = resize
    self._hm_model = fpn_hm.resnet50(
        self._num_classes, is_deconv=False, pretrained=True, input_channels=3)
    self._sigmoid = nn.Sigmoid()

  def forward(self, input_tensor):
    """ forward function for torch modules.
    Parameters:
    ----------
    input : 4-D tensor (B,C,H,W)
      input tensor.
    """
    ori_h, ori_w = input_tensor.shape[2:]
    if self._resize != None:
      input_resize = torch.nn.functional.interpolate(
          input_tensor, size=self._resize, mode='bilinear') # mode='nearest'
    else:
      input_resize = input_tensor
    _, _, heatmap = self._hm_model(input_resize)
    attention_map = self._sigmoid(heatmap)
    if self._resize != None:
      attention_map_orisize = torch.nn.functional.interpolate(
          attention_map, size=(ori_h, ori_w), mode='bilinear') # mode='nearest'
    else:
      attention_map_orisize = attention_map
    return attention_map_orisize


class ForegroundAttentionFCN(nn.Module):
  """ Class for attention model of AFV attack.
  """
  def __init__(self, 
               num_classes: int,
               feature_channels: int,
               resize: tuple=None,
               use_pretrained_foreground: bool=False) -> None:
    """
    Parameters:
    ----------
    num_classes : int
      Number of classes
    feature_channels : int
      Number of feature channels
    resize : (int, int)
      Interpolation input size for FCN matching.
    use_pretrained_foreground : bool
      Use |1 - background properbility| of the pre-trained
      senmentor as heatmap.
    """
    super(ForegroundAttentionFCN, self).__init__()
    self._num_classes = num_classes
    self._feature_channels = feature_channels
    self._use_pretrained_foreground = use_pretrained_foreground
    if resize != None:
      assert isinstance(resize, tuple)
      assert len(resize) == 2
      assert isinstance(resize[0], int)
      assert isinstance(resize[1], int)
    self._resize = resize
    self._hm_model = torchvision.models.segmentation.fcn_resnet50(
        pretrained=True, progress=True, num_classes=21)
    if self._use_pretrained_foreground:
      self._softmax = nn.Softmax(dim=1)
      for p in self._hm_model.parameters():
        p.requires_grad = False
    else:
      self._conv = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=1)
      self._sigmoid = nn.Sigmoid()

  def forward(self, input_tensor):
    """ forward function for torch modules.
    Parameters:
    ----------
    input : 4-D tensor (B,C,H,W)
      input tensor.
    """
    ori_h, ori_w = input_tensor.shape[2:]
    if self._resize != None:
      input_resize = torch.nn.functional.interpolate(
          input_tensor, size=self._resize, mode='bilinear') # mode='nearest'
    else:
      input_resize = input_tensor
    heatmap_dict = self._hm_model(input_resize)
    if not self._use_pretrained_foreground:
      heatmap = self._conv(heatmap_dict['out'])
      attention_map = self._sigmoid(heatmap)
    else:
      heatmap = self._softmax(heatmap_dict['out'])
      attention_map = 1 - heatmap[:, 0 : 1, :, :]

    if self._resize != None:
      attention_map_orisize = torch.nn.functional.interpolate(
          attention_map, size=(ori_h, ori_w), mode='bilinear') # mode='nearest'
    else:
      attention_map_orisize = attention_map
    return attention_map_orisize


def test_attention():
  """ Test function for |ForegroundAttention| class.
  """
  dummy_img = torch.rand(16, 3, 224, 224)
  attention = ForegroundAttentionFCN(
      1, 3, resize=(256, 256),
      use_pretrained_foreground=False)
  att_map = attention(dummy_img)
  rint(att_map.shape)

  attention = ForegroundAttentionFCN(
      1, 3, resize=(256, 256),
      use_pretrained_foreground=True)
  att_map = attention(dummy_img)
  print(att_map.shape)


if __name__ == "__main__":
  test_attention()
