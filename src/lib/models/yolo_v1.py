import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Callable, Any
import struct
import numpy as np

from .misc import BasicConv2d, BasicLinear

__all__ = ['Yolo1']

class YoloInception(nn.Module):
  def __init__(self, in_channels, ch3x3red, ch3x3):
    super().__init__()
    self.conv1 = BasicConv2d(in_channels, ch3x3red, kernel_size=1)
    self.conv2 = BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    return x
  
  def load_darknet_weights(self, data, offset):
    _offset = offset
    _offset += self.conv1.load_darknet_weights(data, _offset)
    _offset += self.conv2.load_darknet_weights(data, _offset)
    return offset

class YoloGoogleNet(nn.Module):
  _layers='conv1 maxpool1 conv2 maxpool2 \
          inception3a inception3b maxpool3 \
          inception4a inception4b inception4c inception4d inception4e maxpool4 \
          inception5a inception5b conv5a conv5b \
          conv6a conv6b'
  _layers = [e for e in _layers.split() if len(e.strip()) > 0]
  def __init__(self):
    super().__init__()
    self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
    self.maxpool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
    
    self.conv2 = BasicConv2d(64, 192, kernel_size=3, padding=1)
    self.maxpool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    self.inception3a = YoloInception(192, 128, 256)
    self.inception3b = YoloInception(256, 256, 512)
    self.maxpool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    self.inception4a = YoloInception(512, 256, 512)
    self.inception4b = YoloInception(512, 256, 512)
    self.inception4c = YoloInception(512, 256, 512)
    self.inception4d = YoloInception(512, 256, 512)
    self.inception4e = YoloInception(512, 512, 1024)
    self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    self.inception5a = YoloInception(1024, 512, 1024)
    self.inception5b = YoloInception(1024, 512, 1024)
    self.conv5a = BasicConv2d(1024, 1024, kernel_size=3, padding=1)
    self.conv5b = BasicConv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
    
    self.conv6a = BasicConv2d(1024, 1024, kernel_size=3, padding=1)
    self.conv6b = BasicConv2d(1024, 1024, kernel_size=3, padding=1)

  def forward(self, x, print_layer_info=False):
    for name in self._layers:
      layer = getattr(self, name, None)
      if print_layer_info:
        info = f'{name:<15}: {" x ".join([str(e) for e in list(x.shape)]): <25}'
      x = layer(x)
      if print_layer_info:
        info += f' --->  {" x ".join([str(e) for e in list(x.shape)])}'
        print(info)
    return x

  def load_darknet_weights(self, data, offset):
    _offset = offset
    for name in self._layers:
      print(name)
      layer = getattr(self, name, None)
      if hasattr(layer, 'load_darknet_weights'):
        _offset += layer.load_darknet_weights(data, _offset)
    return _offset - offset
  
class Yolo1(nn.Module):
  name = 'yolov1'
  image_size = 448
  _layers = 'conv dropout fc'.split()
  def __init__(self, num_classes, num=3, dropout=0.5):
    super().__init__()
    self.num_classes = num_classes
    self.backbone = YoloGoogleNet()
    self.conv = BasicConv2d(1024, 256, with_bn=False, kernel_size=3, padding=1)
    self.dropout = nn.Dropout(p=dropout)
    self.fc = BasicLinear(12544, (num_classes + 5 * num) * 49)
    
  def forward(self, x, print_layer_info=False):
    b = x.size(0)
    x = self.backbone(x, print_layer_info=print_layer_info)
    for name in self._layers:
      layer = getattr(self, name, None)
      if layer:
        if name == 'fc':
          x = x.view(b, -1)
        if print_layer_info:
          info = f'{name:<15}: {" x ".join([str(e) for e in list(x.shape)]): <25}'
        x = layer(x)
        if print_layer_info:
          info += f' --->  {" x ".join([str(e) for e in list(x.shape)])}'
          print(info)
    return x
  
  def load_darknet_weights(self, weight_file):
    with open(weight_file, 'rb') as f:
      data = f.read()
      offset = 12
      major, minor, revision = struct.unpack('iii', data[:offset])
      if ((major * 10 + minor) >= 2) and (major < 1000) and (minor < 1000):
        seen, = struct.unpack('i', data[offset:offset+8])
        offset += 8
      else:
        seen, = struct.unpack('i', data[offset:offset+4])
        offset += 4
      print(major, minor, revision, seen)
      offset += self.backbone.load_darknet_weights(data, offset)
      offset += self.conv.load_darknet_weights(data, offset)
      offset += self.fc.load_darknet_weights(data, offset)
      print(offset)
