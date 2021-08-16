import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms

import sys
sys.path.append("/home/yantaolu/workspace/projects/Deformable-Convolution-V2-PyTorch")
# from modules.deform_conv import DeformConvPack as DCN
from apfv21.attacks.models.fpn_hm_utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes

import pdb

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
  def __init__(self, C3_size, C4_size, C5_size, is_deconv, feature_size=256):
    super(PyramidFeatures, self).__init__()

    if is_deconv:
      # resister_conv = DCN
      raise ValueError("Deformable Conv is not enabled yet.")
    else:
      resister_conv = nn.Conv2d

    # upsample C5 to get P5 from the FPN paper
    self.P5_1 = nn.Conv2d(
        C5_size, feature_size, kernel_size=1, stride=1, padding=0)
    self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
    self.P5_2 = resister_conv(
        feature_size, feature_size, kernel_size=3, stride=1, padding=1)
    self.bn5_2 = nn.BatchNorm2d(feature_size)

    # add P5 elementwise to C4
    self.P4_1 = nn.Conv2d(
        C4_size, feature_size, kernel_size=1, stride=1, padding=0)
    self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
    self.P4_2 = resister_conv(
        feature_size, feature_size, kernel_size=3, stride=1, padding=1)
    self.bn4_2 = nn.BatchNorm2d(feature_size)

    # add P4 elementwise to C3
    self.P3_1 = nn.Conv2d(
        C3_size, feature_size, kernel_size=1, stride=1, padding=0)
    self.P3_2 = resister_conv(
        feature_size, feature_size, kernel_size=3, stride=1, padding=1)
    self.bn3_2 = nn.BatchNorm2d(feature_size)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    self.P6 = resister_conv(
        C5_size, feature_size, kernel_size=3, stride=2, padding=1)
    self.bn6 = nn.BatchNorm2d(feature_size)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    self.P7_1 = nn.ReLU()
    self.P7_2 = resister_conv(
        feature_size, feature_size, kernel_size=3, stride=2, padding=1)
    self.bn7_2 = nn.BatchNorm2d(feature_size)

  def forward(self, inputs):
    C3, C4, C5 = inputs
    
    P5_x = self.P5_1(C5)
    P5_upsampled_x = self.P5_upsampled(P5_x)
    P5_x = self.bn5_2(self.P5_2(P5_x))

    P4_x = self.P4_1(C4)
    P4_x = P5_upsampled_x + P4_x
    P4_upsampled_x = self.P4_upsampled(P4_x)
    P4_x = self.bn4_2(self.P4_2(P4_x))

    P3_x = self.P3_1(C3)
    P3_x = P3_x + P4_upsampled_x
    P3_x = self.bn3_2(self.P3_2(P3_x))

    P6_x = self.bn6(self.P6(C5))

    P7_x = self.P7_1(P6_x)
    P7_x = self.bn7_2(self.P7_2(P7_x))

    return [P3_x, P4_x, P5_x, P6_x, P7_x]


class HeatmapModel(nn.Module):
  def __init__(self, num_classes, feature_channels):
    super(HeatmapModel, self).__init__()

    self.P7_upsampled = nn.ConvTranspose2d(
        feature_channels, feature_channels, 32,
        stride=16, padding=8, bias=False)
    self.bn1 = nn.BatchNorm2d(feature_channels)
    self.P3_score = nn.Conv2d(feature_channels, num_classes, 1)
    self.P3_upsampled = nn.ConvTranspose2d(
        num_classes, num_classes, 16,
        stride=8, padding=4, bias=False)

  def forward(self, fpn_features):
    [P3_x, P4_x, P5_x, P6_x, P7_x] = fpn_features
    # P3_x:(48,160) 8x, P4_x:(24,80) 16x, P5_x:(12,40) 32x, P6_x:(6,20) 64x, P7_x:(3,10) 128x
    P7_up = self.P7_upsampled(P7_x)
    P3 = self.bn1(P3_x + P7_up)
    heatmap_feature = self.P3_score(P3)
    heatmap_orisize = self.P3_upsampled(heatmap_feature)
    return heatmap_feature, P3, heatmap_orisize


class ResNet(nn.Module):

  def __init__(self, num_classes, block, layers, input_channels, is_deconv):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self._is_deconv = is_deconv
    self.conv1 = nn.Conv2d(
        input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    if block == BasicBlock:
      fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels,
                   self.layer3[layers[2] - 1].conv2.out_channels,
                   self.layer4[layers[3] - 1].conv2.out_channels]
    elif block == Bottleneck:
      fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels,
                   self.layer3[layers[2] - 1].conv3.out_channels,
                   self.layer4[layers[3] - 1].conv3.out_channels]
    else:
      raise ValueError("Block type {block} not understood")

    self.fpn = PyramidFeatures(
        fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], self._is_deconv)
    self.heatmap = HeatmapModel(
        num_classes, self.layer1[-1].conv3.out_channels)

    for m in self.modules():
      if isinstance(m, nn.Conv2d): #isinstance(m, DCN) or isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    # self.freeze_bn()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(planes * block.expansion),
      )

    layers = [block(self.inplanes, planes, stride, downsample)]
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def freeze_bn(self):
    '''Freeze BatchNorm layers.'''
    for layer in self.modules():
      if isinstance(layer, nn.BatchNorm2d):
        layer.eval()

  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x1 = self.layer1(x)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)

    fpn_features = self.fpn([x2, x3, x4])
    heatmap, bottleneck, heatmap_orisize = self.heatmap(fpn_features)
    return heatmap, bottleneck, heatmap_orisize



def resnet18(num_classes, pretrained=False, **kwargs):
  """Constructs a ResNet-18 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
  if pretrained:
    model.load_state_dict(
        model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
  return model


def resnet34(num_classes, pretrained=False, **kwargs):
  """Constructs a ResNet-34 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(
        model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
  return model


def resnet50(num_classes, pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(
        model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
  return model


def resnet101(num_classes, pretrained=False, **kwargs):
  """Constructs a ResNet-101 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
  if pretrained:
    model.load_state_dict(
        model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
  return model


def resnet152(num_classes, pretrained=False, **kwargs):
  """Constructs a ResNet-152 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
  if pretrained:
    model.load_state_dict(
        model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
  return model
