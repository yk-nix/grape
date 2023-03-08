from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from .misc import BasicConv2d, BasicLinear, BasicModule

__all__ = ['Yolo1Tiny']

class Yolo1Tiny(BasicModule):
  name='yolov1'
  _layers='conv1,conv2,conv3,conv4,conv5,conv6,conv7,conv8,fc'.split(',')
  def __init__(self, num_classes:int, bbox_per_cell=2, grid_size:int|tuple=7, image_size=448) -> None:
    super().__init__()
    mp = nn.MaxPool2d(kernel_size=2, stride=2)
    act = nn.LeakyReLU(0.02)
    self.image_size = image_size
    self.num_classes = num_classes
    self.grid_size = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)
    self.cells = reduce(lambda w, h: w * h, self.grid_size)
    self.bbox_per_cell = bbox_per_cell
    self.conv1 = BasicConv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, activation=act, pool=mp)
    self.conv2 = BasicConv2d(in_channels=self.conv1.out_channels, out_channels=32,   kernel_size=3, padding=1, activation=act, pool=mp)
    self.conv3 = BasicConv2d(in_channels=self.conv2.out_channels, out_channels=64,   kernel_size=3, padding=1, activation=act, pool=mp)
    self.conv4 = BasicConv2d(in_channels=self.conv3.out_channels, out_channels=128,  kernel_size=3, padding=1, activation=act, pool=mp)
    self.conv5 = BasicConv2d(in_channels=self.conv4.out_channels, out_channels=256,  kernel_size=3, padding=1, activation=act, pool=mp)
    self.conv6 = BasicConv2d(in_channels=self.conv5.out_channels, out_channels=512,  kernel_size=3, padding=1, activation=act, pool=mp)
    self.conv7 = BasicConv2d(in_channels=self.conv6.out_channels, out_channels=1024, kernel_size=3, padding=1, activation=act)
    self.conv8 = BasicConv2d(in_channels=self.conv7.out_channels, out_channels=256,  kernel_size=3, padding=1, activation=act)
    self.fc = BasicLinear(in_features=self.cells * self.conv8.out_channels, out_features=self.cells * (num_classes + bbox_per_cell * 5))