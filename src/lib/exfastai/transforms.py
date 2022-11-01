from __future__ import annotations

from fastai.data.all import *
from fastai.vision.all import *
from torch import isin

__all__= ['PointScalerReverse']


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

