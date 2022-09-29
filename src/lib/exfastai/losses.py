import torch.nn as nn
import torch.nn.functional as F

from fastai.losses import *

from lib.models.ssd import SSDLoss
from lib.models.utils import decode

class SSDLossFLat(BaseLoss):
  y_int = True # y interpolation
  def __init__(self, *args, axis=-1, **kwargs):
    super().__init__(SSDLoss, *args, axis=axis, **kwargs)
    
  def __call__(self, preds, bboxes, labels):
    return self.func.__call__(preds, bboxes, labels)
    
  def activation(self, preds):
    conf, loc = preds
    scores = F.softmax(conf, dim=-1)
    priors, variance = self.func.ssd.priors, self.func.variance
    bboxes = decode(loc, priors, variance)
    raise NotImplemented()
  
  def decodes(self, preds):
    raise NotImplemented()