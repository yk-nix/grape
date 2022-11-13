# -*- coding: utf-8 -*-
"""
Created on 2022.04.01

@author: yoka
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Any, Callable
from torch import Tensor


__all__ = ['load_weight_from_buffer', 'BasicConv2d', 'BasicLinear']


def load_weight_from_buffer(params:Tensor|Parameter,  data:bytearray, offset:int) -> int:
  start, end = offset, offset
  if not hasattr(params, '__iter__'):
    params = [params]
  for param in params:
    if param is not None:
      end += param.numel() * 4
      weight = torch.frombuffer(data[start:end], dtype=torch.float32).view(param.shape)
      if isinstance(param, Parameter):
        param.data.copy_(weight)
      else:
        param.copy_(weight)
      print(int((end-start)/4))
      start = end
  return end - offset

class BasicConv2d(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, activation: Callable=F.relu, with_bn: bool=True, **kwargs: Any) -> None:
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
    self.bn = None
    if with_bn:
      self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    self.act = activation

  def forward(self, x: Tensor) -> Tensor:
    x = self.conv(x)
    if self.bn:
      x = self.bn(x)
    if self.act:
      x = self.act(x, inplace=True)
    return x
  
  def load_darknet_weights(self, data:bytearray, offset:int) -> int:
    _offset = offset
    params = [self.conv.bias]
    if self.bn:
      params += [self.bn.weight, self.bn.running_mean, self.bn.running_var]
    params += [self.conv.weight]
    _offset += load_weight_from_buffer(params, data, _offset)
    return _offset - offset

class BasicLinear(nn.Module):
  def __init__(self, in_features: int, out_features: int, activation: Callable=F.relu, with_bn: bool=False, **kwargs: Any) -> None:
    super().__init__()
    self.linear = nn.Linear(in_features=in_features, out_features=out_features, **kwargs)
    self.act = activation
    self.bn = None
    if with_bn:
      self.bn = nn.BatchNorm1d(out_features, eps=0.00001)

  def forward(self, x:Tensor) -> Tensor:
    x = self.linear(x)
    if self.act is not None:
      x = self.act(x)
    return x

  def load_darknet_weights(self, data:bytearray, offset:int) -> int:
    _offset = offset
    params = [self.linear.bias, self.linear.weight]
    if self.bn:
      params += [self.bn.weight, self.bn.running_mean, self.bn.running_var]
    _offset += load_weight_from_buffer(params, data, _offset)
    return _offset - offset