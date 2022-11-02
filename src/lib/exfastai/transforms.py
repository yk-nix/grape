from __future__ import annotations

from fastai.data.all import *
from fastai.vision.all import *
from torch import isin

__all__= ['PointScalerReverse', 'ExpandBatch']


#--------------------------------------------------------
# reverse version of PointScaler
class PointScalerReverse(Transform):
    "Scale a tensor representing points"
    order = 1
    def __init__(self, do_scale=True, y_first=False, order=None): 
      self.scaler = PointScaler(do_scale=do_scale, y_first=y_first)
      if order:
        self.order = order
      
    def encodes(self, x:PILBase|TensorImageBase): 
      return self.scaler.encodes(x)
    
    def decodes(self, x:PILBase|TensorImageBase): 
      return self.scaler.decodes(x)

    def encodes(self, x:TensorPoint):
      if len(x.size()) > 2:
        shape = x.size()
        x = x.view(-1, 4)
        return self.scaler.decodes(x).view(shape)
      else:
        return self.scaler.decodes(x)
      
    def decodes(self, x:TensorPoint):
      if len(x.size()) > 2:
        shape = x.size()
        x = x.view(-1, 4)
        return self.scaler.encodes(x).view(shape)
      else:
        return self.scaler.decodes(x)

#--------------------------------------------------------------------
# # expand a batch [item] into [item, item, ....], only for training dataset
class ExpandBatch(Transform):
  def __init__(self, times=8, split_idx=0):
    self.split_idx = split_idx
    self.times = times
      
  def encodes(self, x):
    return x * self.times

#--------------------------------------------------------------------
# # PointScaler ovrride
@patch
def __init__(self:PointScaler, do_scale=True, y_first=False, img_size=None):
  self.do_scale, self.y_first, self.sz = do_scale, y_first, img_size

#--------------------------------------------------------------------
# # DeterministicFlip ovrride
@patch
def __init__(self:DeterministicFlip,
             size:int|tuple=None, # Output size, duplicated if one value is specified
             mode:str='bilinear', # PyTorch `F.grid_sample` interpolation
             pad_mode=PadMode.Reflection, # A `PadMode`
            align_corners=True, # PyTorch `F.grid_sample` align_corners
             **kwargs):
        Flip.__init__(self, p=1., size=size, draw=DeterministicDraw([0,1]), mode=mode, pad_mode=pad_mode, align_corners=align_corners, **kwargs)