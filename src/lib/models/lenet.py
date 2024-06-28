from torch import Tensor
from typing import NoReturn, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LeNet']

class LeNet(nn.Module):
  version = "5.0"
  name = 'lenet'
  
  def __init__(self, classes: int) -> NoReturn:
    super().__init__()
    self.classes = classes
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, padding = 2)    
    self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)
    self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size= 5)
    self.linear1 = nn.Linear(in_features = 120, out_features = 84)
    self.linear2 = nn.Linear(in_features = 84, out_features = classes)
      
  def forward(self, images : Tensor) -> Tensor:
    """
    images: b x 1 x 28 x 28
    ouput:  b x classes
    """
    # b x 1 x 28 x 28 --> b x 6 x 28 x 28
    x = F.sigmoid(self.conv1(images))
    # b x 6 x 28 x 28 --> b x 6 x 14 x 14
    x = F.avg_pool2d(x, 2, 2)
    # b x 6 x 14 x 14 -> b x 16 x 10 x 10
    x = F.sigmoid(self.conv2(x))
    # b x 16 x 10 x 10 --> b x 16 x 5 x 5
    x = F.avg_pool2d(x, 2, 2)
    # b x 16 x 5 x 5 --> b x 120 x 1 x 1
    x = self.conv3(x)
    # b x 120 -> b x 84
    x = self.linear1(x.reshape(-1, 120))
    # b x 84 -> b x classes
    return self.linear2(x)
  
  @staticmethod
  def eval(x: Tensor)-> Tensor:
    """
      x (shape: b x classes):     the output of forward
    """
    return torch.softmax(x, dim = -1).max(dim = -1)