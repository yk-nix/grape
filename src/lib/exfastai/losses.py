import torch
import torch.nn.functional as F

from fastai.losses import *
from fastai.vision.all import *

from lib.models.ssd import SSDLoss
from .misc import post_predict

class SSDLossFlat(BaseLoss):
  y_int = True # y interpolation
  def __init__(self, ssd, *args, axis=-1, nms_iou_threshold=0.3, score_threshold=0.95, top_n=200, **kwargs):
    self.nms_iou_threshold = nms_iou_threshold
    self.score_threshold = score_threshold
    self.top_n = top_n
    super().__init__(SSDLoss, ssd=ssd, *args, axis=axis, **kwargs)
    
  def __call__(self, preds, bboxes, labels):
    bboxes = bboxes.clamp(min=0, max=self.func.ssd.image_size)
    return self.func.__call__(preds, bboxes, labels)
    
  def activation(self, preds):
    conf, loc = preds
    scores = F.softmax(conf, dim=-1)
    bboxes_list = []
    for _bboxes in loc:
      bboxes_list.append(self.func.decode(_bboxes))
    bboxes = torch.stack(bboxes_list)
    return scores, bboxes
  
  def decodes(self, preds):
    bboxes, labels, scores = post_predict(self, *preds, self.top_n, self.score_threshold)
    return TensorBBox(bboxes), TensorMultiCategory(labels)