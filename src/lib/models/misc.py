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
import struct

__all__ = ['load_weight_from_buffer', 'BasicConv2d', 'BasicLinear', 'BasicModule']


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
      #print(int((end-start)/4))
      start = end
  return end - offset

class BasicConv2d(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, activation: Callable=F.relu, with_bn: bool=True, pool: Callable=None, **kwargs: Any) -> None:
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
    self.bn = None
    if with_bn:
      self.bn = nn.BatchNorm2d(out_channels, momentum=0.001)
    self.act = activation
    self.pool = pool
    self.out_channels = out_channels

  def forward(self, x: Tensor) -> Tensor:    
    x = self.conv(x)
    if self.bn:
      x = self.bn(x)
    if self.act:
      x = self.act(x)
    if self.pool is not None:
      x = self.pool(x)
    return x
  
  def load_darknet_weights(self, data:bytearray, offset:int) -> int:
    _offset = offset
    if self.bn:
      torch.nn.init.zeros_(self.conv.bias)
      params = [self.bn.bias, self.bn.weight, self.bn.running_mean, self.bn.running_var]
    else:
      params = [self.conv.bias]
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
 
class BasicModule(nn.Module):
  _layers=''.split()
  def __init__(self):
    super().__init__()
    
  def forward(self, x, print_layer_info: bool=False):
    b = x.size(0)
    for layer_name in self._layers:
      layer = getattr(self, layer_name)
      if layer_name == "fc":
        x = x.view(b, -1)
      if print_layer_info:
        info = f'{layer_name:<15} {" x ".join([str(e) for e in list(x.shape)]): <25}'
      x  = layer(x)
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
      for layer_name in self._layers:
        layer = getattr(self, layer_name)
        if hasattr(layer, 'load_darknet_weights'):
          offset += layer.load_darknet_weights(data, offset)
      return major, minor, revision, seen
  
class LocalLayer(nn.Module):
  def __init__(self, width:int, height: int, kernel_size, in_channels: int, out_channels:int, activation: Callable=F.relu, **kwargs: Any) -> None:
    super().__init__()
    self.act = activation
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=kernel_size, **kwargs)
    self.width = width
    self.height = height
    self.size = kernel_size
    
  def forward(self, x:Tensor) -> Tensor:
    raise NotImplementedError()
    