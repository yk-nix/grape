import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.boxes import box_convert

from fastai.losses import *

from lib.models.ssd import SSDLoss

class SSDLossFlat(BaseLoss):
  y_int = True # y interpolation
  def __init__(self, ssd, *args, axis=-1, **kwargs):
    super().__init__(SSDLoss, ssd=ssd, *args, axis=axis, **kwargs)
    
  def __call__(self, preds, bboxes, labels):
    bboxes = bboxes.clamp(min=0, max=self.func.ssd.image_size)
    return self.func.__call__(preds, bboxes, labels)
    
  def activation(self, preds):
    conf, loc = preds
    scores = F.softmax(conf, dim=-1)
    bboxes = self.func.decode(loc)
    raise NotImplemented()
  
  def decodes(self, preds):
    raise NotImplemented()