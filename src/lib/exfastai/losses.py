import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms as nms

from fastai.losses import *
from fastai.vision.all import *

from lib.models.ssd import SSDLoss

class SSDLossFlat(BaseLoss):
  y_int = True # y interpolation
  def __init__(self, ssd, *args, axis=-1, nms_iou_threshold=0.3, top_n=200, **kwargs):
    self.nms_iou_threshold = nms_iou_threshold
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
    _scores, _bboxes = preds
    idx = _scores.max(dim=-1)[0].argsort(dim=-1, descending=True).unsqueeze(dim=-1)
    scores = _scores.gather(1, idx.broadcast_to(_scores.shape))[:,:self.top_n,:]
    bboxes = _bboxes.gather(1, idx.broadcast_to(_bboxes.shape))[:,:self.top_n,:]   
    scores, labels = scores.max(dim=self.axis)
    label_list, bbox_list, score_list = [], [], []
    for b, s, l in zip(bboxes, scores, labels):
      i = nms(b, s, l, self.nms_iou_threshold)
      label_list.append(l[i])
      bbox_list.append(b[i])
      score_list.append(s[i])
    return TensorBBox(torch.stack(bbox_list)), TensorMultiCategory(torch.stack(label_list))