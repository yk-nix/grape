from __future__ import annotations

from fastai.data.all import *
from fastai.vision.all import *

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

#--------------------------------------------------------
# Override on MutliCategorize
@patch
def encodes(self:MultiCategorize, o:LabeledBBox):
  if not all(elem in self.vocab.o2i.keys() for elem in o.lbl):
    diff = [elem for elem in o.lbl if elem not in self.vocab.o2i.keys()]
    diff_str = "', '".join(diff)
    raise KeyError(f"Labels '{diff_str}' were not included in the training dataset")
  return LabeledBBox(o.bbox, TensorMultiCategory([self.vocab.o2i[e] for e in o.lbl]))
@patch
def decodes(self:MultiCategorize, o:LabeledBBox):
  lbl = [self.vocab[e] for e in o.lbl]
  return LabeledBBox(o.bbox, lbl)


#--------------------------------------------------------
# Override on Resize
